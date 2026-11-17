# melmod - arecanut models for the NEW striped background (v3)

- weights/detector_v3.pt (default): YOLO11s detector, new background: mAP50 0.992, recall 0.991
- weights/vit_areca_v3.pt (default): ViT gottu/kole/pure, new background test 100%, old dataset 99.3%
- Run (USB camera, GPU): python line_count.py --source 1 --device 0
- Retrain for a new background: python prepare_new_bg.py --src ..\final_data_set\captured_images (then train_mixed.py / train_vit_cls.py --init)

# Arecanut detection, grading and two-camera sorting (ESP32 servo)

Live camera system that:

1. **Detects every arecanut** in the frame (YOLO11s detector) and draws a box on each.
2. **Tracks** each nut with a persistent ID (ByteTrack).
3. **Counts** a nut the first time its box touches a line you draw on the camera view
   (1, 2, 3 …), with its number shown on top of its box.
4. **Grades** each nut as **gottu / kole / pure** with a Vision Transformer (ViT) that looks
   only inside the box, and keeps a per-type tally.
5. **Sorts** them: a second camera counts the same nuts and tells an ESP32 servo arm to turn to
   0 deg (bette / pure), +40 deg (kole) or -40 deg (gottu).

Dataset: [Melroy-Sahyadri-ECE/areca_dataset](https://github.com/Melroy-Sahyadri-ECE/areca_dataset)
(1,408 photos of gottu / kole / pure with segmentation masks).

## Models (in `weights/`)

| File | What | Test results (140 held-out photos) |
|---|---|---|
| `detector_v2.pt` | YOLO11s, one class `arecanut`, 512 px | mAP50 0.983 · mAP50-95 0.789 · recall 0.993 |
| `vit_areca_v2.pt` | ViT-Tiny (timm `vit_tiny_patch16_224`), 3 classes | 99.3 % on real backgrounds · 97.1 % on unseen backgrounds |
| `receiver_yolo11n.pt` | YOLO11n box-only detector for the sorting camera, 416 px | mAP50 0.982 · mAP50-95 0.819 · recall 0.993 · ~13 ms per frame on the CPU |

The ViT was trained on nuts cut out with the segmentation masks and pasted onto many
backgrounds, so it judges the nut itself, not the floor.

---

## 1. Setup (Windows: PowerShell or the Antigravity terminal)

```powershell
git clone https://github.com/Melroy-Sahyadri-ECE/melroy_final_-model.git
cd melroy_final_-model
python -m venv venv
.\venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
# NVIDIA GPU (optional, needed for ~30 fps):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Check the GPU (prints `True` if CUDA works; the CPU also works, just slower):

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

On Linux / macOS use the same commands, but activate with `source venv/bin/activate`.

## 2. Run the line counter (main app)

```powershell
python line_count.py --source 1 --device cpu      # USB camera = 1, laptop camera = 0
python line_count.py --source 1 --device 0        # use the NVIDIA GPU (faster)
```

In the camera window:

| Action | How |
|---|---|
| Draw / move the counting line | click, drag and release with the mouse (saved to `line_config.json`) |
| Redraw the line | `r` |
| Reset the count | `c` |
| Quit (prints the final count per type) | `q` or `Esc` |

Each nut shows `number | type confidence` on top of its box, e.g. `3 | pure 97%`.
The top bar shows `COUNT`, nuts in frame, fps and the `gottu / kole / pure` totals.

Useful options:

```powershell
python line_count.py --source 1 --conf 0.4               # fewer false boxes (default 0.3)
python line_count.py --source 1 --vit-every 5            # classify less often = faster
python line_count.py --source video.mp4 --save out.mp4   # run on a video and save the result
python line_count.py --source 1 --vit ""                 # detector + counting only (no ViT)
```

## 3. Two-camera sorting with the ESP32 servo

```
 camera 1 (main)  line_count.py          camera 2 (sorting)  receiver.py           ESP32 (pin 19 servo)
 counts nut #n, ViT finds its type  -->  counts the same nuts in the same  -->   bette/pure ->   0 deg
 sends {"n":5,"cls":"kole"} (UDP)        order; when ITS #n touches ITS line,    kole       -> +40 deg
                                          sends /sort?type=kole (WiFi)            gottu      -> -40 deg
```

1. Flash `esp32/areca_sorter/areca_sorter.ino` (Arduino IDE, board "ESP32 Dev Module",
   library **ESP32Servo**). Pins are unchanged: servo 1 = 18, sorting servo = 19, HOME sensor = 21,
   stepper PUL/DIR = 26/27.
2. Put the arm at HOME and power the ESP32. It makes its own WiFi **ESP32-Servo**
   (password `12345678`); connect the PC to it. The pin 19 arm stays at HOME and moves
   **only** when receiver.py sends an arecanut (servo 1 and the stepper run as before).
3. Start both cameras, each in its own terminal:

```powershell
# terminal 1 - main camera (counting + ViT grading), sends to the receiver
python line_count.py --source 1 --device cpu
# terminal 2 - sorting camera (box-only model, line in the middle), drives the ESP32
python receiver.py --source 0 --device 0 --esp 192.168.4.1   # uses the same detector_v3.pt as line_count.py
```

- The receiver's line starts in the **middle** of its view (`--line vertical` / `horizontal`);
  drag with the mouse to move it (saved to `receiver_line.json`). Put it where a nut is just
  before the sorting arm.
- Both cameras must see the nuts **in the same order**. Press `c` in the main window to reset
  both counts together.
- If a nut reaches the receiver before its type arrived, it waits up to `--wait 3` s, then uses
  `--unknown pure` (or `kole`, `gottu`, `skip`).
- No ESP32 yet? `python receiver.py --source 0 --esp ""` (dry run, shows what it would send).
- Test the arm by hand from a browser on the ESP32 WiFi:
  `http://192.168.4.1/sort?type=kole` · `?type=gottu` · `?type=bette` · `?pos=-40` ·
  status: `http://192.168.4.1/status`.

## 4. Tracking only (no line, no grading)

```powershell
python track.py --source 1 --show                  # live camera with IDs and trails
python track.py --source video.mp4 --line 0.5      # count nuts crossing a horizontal line
```

## 5. Retrain (optional)

Put the dataset next to this folder:

```powershell
git clone https://github.com/Melroy-Sahyadri-ECE/areca_dataset.git ..\areca_dataset

# Detector data: tight boxes from the masks + multi-nut composite images
python prepare_dataset.py --src ..\areca_dataset --out ..\areca_yolo --refine
python make_composites.py --data ..\areca_yolo --n 1500 --kole-weight 2 --synth-bg-frac 0.45
# Detector training (use --device cpu without a GPU; --workers 2 avoids Windows memory errors)
python train.py --data ..\areca_yolo\data.yaml --model weights\detector_v2.pt --epochs 30 --imgsz 512 --batch 8 --workers 2 --device 0 --name my_detector
# Which dataset nuts does it miss?
python find_missed.py --weights weights\detector_v2.pt --data ..\areca_yolo --device cpu

# ViT data: nuts cut out with the masks and pasted onto random backgrounds
python make_vit_seg_crops.py --data ..\areca_yolo --out ..\areca_vit --per-nut 5
python train_vit_cls.py --data ..\areca_vit --device cpu --epochs 8 --workers 2 --out weights\vit_new.pt
```

New detector weights are saved to `runs\detect\runs\areca\<name>\weights\best.pt`.
If training stops (crash, sleep), `python resume.py` continues from the last saved epoch.

GPU with automatic CPU fallback (resumes from the last epoch if the GPU crashes):

```powershell
python train_mixed.py --data ..\areca_yolo\data.yaml --model yolo11n.pt --epochs 25 --imgsz 416 --batch 16 --name receiver_yolo11n
```

## Files

| File | Purpose |
|---|---|
| `line_count.py` | Main app: camera + detector + tracking + line counter + ViT grading |
| `receiver.py` | Sorting camera: box-only detector + middle line + ESP32 servo commands |
| `esp32/areca_sorter/areca_sorter.ino` | ESP32 sketch (WiFi AP): sorting arm 0 / +40 / -40 deg on pin 19, moves only on received arecanuts |
| `train_mixed.py` | Training on the GPU with automatic CPU fallback if the GPU crashes |
| `track.py` | Detector + tracking only |
| `areca_bytetrack.yaml` | Tracker settings |
| `prepare_dataset.py` | Masks → YOLO labels, train/val/test split |
| `make_composites.py`, `backgrounds.py` | Multi-nut training images on varied surfaces |
| `make_vit_seg_crops.py` | Mask cut-outs for the ViT (GrabCut-refined) |
| `train.py`, `train_vit_cls.py` | Train the detector / the ViT |
| `find_missed.py` | Lists nuts the detector misses |
| `make_test_video.py` | Synthetic multi-nut test video |
| `resume.py` | Resume interrupted detector training |

## Notes

- Speed: about 9 fps on a CPU (Ryzen AI 9 365); use `--device 0` on an NVIDIA GPU for ~30 fps.
- The accuracy numbers are on dataset photos. On a new camera or surface, record a short clip
  and add frames to the dataset if you see missed nuts or wrong types.
