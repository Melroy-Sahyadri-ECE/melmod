"""
GPU training with automatic CPU fallback.

Runs YOLO training in a child process on the GPU. If the GPU crashes (driver reset, CUDA
error, process killed), it resumes from the last saved epoch - on the GPU again if CUDA is
still available, otherwise (or after 2 GPU crashes) on the CPU. Keeps the PC awake.

    python train_mixed.py --data ..\\areca_yolo\\data.yaml --model yolo11n.pt --epochs 25 --imgsz 416 --batch 16 --name receiver_yolo11n
"""
import argparse
import ctypes
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def last_ckpt(name):
    c = [p for p in glob.glob(os.path.join(HERE, "runs", "**", name, "weights", "last.pt"), recursive=True)]
    return max(c, key=os.path.getmtime) if c else None


def cuda_ok():
    r = subprocess.run([sys.executable, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"])
    return r.returncode == 0


def worker(a):
    """Child process: start new training, or resume from last.pt on the given device."""
    import torch
    from ultralytics import YOLO
    last = last_ckpt(a.name)
    if last:
        ck = torch.load(last, map_location="cpu", weights_only=False)
        ck["train_args"]["device"] = a.device          # switch GPU <-> CPU on resume
        ck["train_args"]["workers"] = a.workers
        ck["train_args"]["batch"] = a.batch if a.device != "cpu" else min(a.batch, 8)
        torch.save(ck, last)
        print(f"=== resuming {last} (epoch {ck['epoch'] + 1}) on device {a.device}", flush=True)
        try:
            YOLO(last).train(resume=True)
        except Exception as e:
            if "finished" in str(e).lower() or "nothing to resume" in str(e).lower():
                print("=== TRAINING FINISHED", flush=True)
                return
            raise
    else:
        print(f"=== new training on device {a.device}", flush=True)
        YOLO(a.model).train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                            workers=a.workers, project="runs/areca", name=a.name, exist_ok=True,
                            single_cls=True, patience=15, cos_lr=True, fliplr=0.5, flipud=0.5,
                            mosaic=1.0, close_mosaic=5, mixup=0.1, scale=0.5, translate=0.1,
                            hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, max_det=300, seed=0, plots=True)
    print("=== TRAINING FINISHED", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--name", default="receiver_yolo11n")
    ap.add_argument("--device", default="0")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        return worker(a)

    if sys.platform == "win32":  # keep the PC awake while training
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    gpu_crashes = 0
    for attempt in range(1, 12):
        dev = "0" if (gpu_crashes < 2 and cuda_ok()) else "cpu"
        print(f"\n##### attempt {attempt}: using {'GPU' if dev == '0' else 'CPU'} #####", flush=True)
        args = [sys.executable, "-u", os.path.abspath(__file__), "--worker", "--data", a.data, "--model", a.model,
                "--epochs", str(a.epochs), "--imgsz", str(a.imgsz), "--batch", str(a.batch),
                "--workers", str(a.workers), "--name", a.name, "--device", dev]
        r = subprocess.run(args, cwd=HERE)
        if r.returncode == 0:
            print("##### done #####")
            break
        if dev == "0":
            gpu_crashes += 1
        print(f"##### training stopped (exit {r.returncode}); resuming in 10 s "
              f"({'GPU crashed ' + str(gpu_crashes) + 'x' if dev == '0' else 'CPU run stopped'}) #####", flush=True)
        time.sleep(10)
    w = last_ckpt(a.name)
    if w:
        print(f"best weights: {os.path.join(os.path.dirname(w), 'best.pt')}")


if __name__ == "__main__":
    main()
