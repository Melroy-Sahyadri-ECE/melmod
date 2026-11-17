"""
Live arecanut LINE COUNTER.

1. Opens the camera and shows the live view with every arecanut boxed.
2. Draw the counting line with the mouse: click, drag, release. (Saved to line_config.json.)
3. The first time an arecanut's box touches the line it gets the next number
   (1, 2, 3, ...). The number is shown on top of its box and stays with that nut.
4. A ViT classifier looks only inside each box and labels the nut pure / kole / gottu
   (shown on top of the box, e.g. "3 | pure 97%"). Counted nuts are tallied per class.
5. Every counted nut is sent as UDP JSON {"type":"nut","n":3,"cls":"pure","conf":0.97}
   to the receiver camera (receiver.py), which sorts it with the ESP32 servo.
   It is re-sent if the ViT changes its mind; "c" also sends {"type":"reset"}.

Keys (click the video window first):
    r  - clear the line and draw a new one     c  - reset the count to 0
    q / Esc - quit

    python line_count.py --weights best.pt --source 1 --device 0
"""
import argparse
import json
import os
import socket
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
WIN = "Arecanut line counter  (drag = draw line | r = redraw | c = reset | q = quit)"


class LineDrawer:
    """Mouse 'pen': click-drag-release draws the counting line."""

    def __init__(self, line=None):
        self.line = line          # ((x1, y1), (x2, y2)) or None
        self.start = None
        self.preview = None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start, self.preview = (x, y), (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.start is not None:
            self.preview = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start is not None:
            if abs(x - self.start[0]) + abs(y - self.start[1]) > 10:  # ignore plain clicks
                self.line = (self.start, (x, y))
            self.start = self.preview = None


def box_touches_line(box, line):
    """True if the rectangle box (x1,y1,x2,y2) touches/crosses the line segment."""
    x1, y1, x2, y2 = [int(v) for v in box]
    ok, _, _ = cv2.clipLine((x1, y1, max(1, x2 - x1), max(1, y2 - y1)), line[0], line[1])
    return ok


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u else 0.0


CLASS_COLORS = {"pure": (0, 220, 0), "kole": (40, 40, 230), "gottu": (0, 200, 255)}  # BGR


class ViTClassifier:
    """ViT that looks only at the inside of each arecanut box -> pure / kole / gottu."""

    def __init__(self, path, device):
        import timm
        import torch
        self.torch = torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.classes = ck["classes"]
        self.size = ck.get("img_size", 224)
        self.dev = torch.device(f"cuda:{device}" if str(device) != "cpu" and torch.cuda.is_available() else "cpu")
        self.half = self.dev.type == "cuda"
        m = timm.create_model(ck["model"], pretrained=False, num_classes=len(self.classes))
        m.load_state_dict(ck["state_dict"])
        self.model = m.to(self.dev).eval()
        if self.half:
            self.model.half()
        self.mean = torch.tensor(ck.get("mean", (0.5,) * 3), device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor(ck.get("std", (0.5,) * 3), device=self.dev).view(1, 3, 1, 1)

    def crop(self, frame, b, pad=0.15):
        """Square crop around the box, same as the training crops (make_cls_crops.py)."""
        H, W = frame.shape[:2]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        side = max(b[2] - b[0], b[3] - b[1]) * (1 + 2 * pad)
        x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x2, y2 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
        c = frame[y1:y2, x1:x2]
        return cv2.resize(c, (self.size, self.size), interpolation=cv2.INTER_AREA) if c.size else None

    def __call__(self, frame, boxes):
        """Classify many boxes in ONE batch. Returns an (N, n_classes) numpy array of probabilities."""
        torch = self.torch
        crops = [self.crop(frame, b) for b in boxes]
        keep = [i for i, c in enumerate(crops) if c is not None]
        out = np.full((len(boxes), len(self.classes)), 1.0 / len(self.classes), np.float32)
        if not keep:
            return out
        x = np.stack([crops[i][:, :, ::-1] for i in keep])  # BGR -> RGB
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(x)).to(self.dev).permute(0, 3, 1, 2)
            t = t.half() if self.half else t.float()
            t = (t / 255.0 - self.mean.to(t.dtype)) / self.std.to(t.dtype)
            p = torch.softmax(self.model(t).float(), 1).cpu().numpy()
        out[keep] = p
        return out


class Camera:
    """Reads frames in a background thread so the newest frame is always ready."""

    def __init__(self, cap):
        self.cap, self.frame, self.ok, self.stop = cap, None, True, False
        self.lock = threading.Lock()
        self.seq = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stop:
            ok, f = self.cap.read()
            with self.lock:
                self.ok = ok
                if ok:
                    self.frame, self.seq = f, self.seq + 1
            if not ok:
                break

    def read(self, last_seq):
        while True:  # wait for a frame newer than the one we already processed
            with self.lock:
                if not self.ok:
                    return False, None, last_seq
                if self.frame is not None and self.seq != last_seq:
                    return True, self.frame, self.seq
            time.sleep(0.001)


def label(img, text, x, y, bg, fg=(0, 0, 0), scale=0.8, thick=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    y = max(th + 8, y)
    cv2.rectangle(img, (x, y - th - 10), (x + tw + 10, y), bg, -1)
    cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(HERE, "weights/detector_v3.pt"))
    ap.add_argument("--source", default="1", help="camera index (0, 1, ...) or a video file")
    ap.add_argument("--device", default="cpu", help="cpu or 0 for the NVIDIA GPU")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--tracker", default=os.path.join(HERE, "areca_bytetrack.yaml"))
    ap.add_argument("--config", default=os.path.join(HERE, "line_config.json"), help="where the line is saved")
    ap.add_argument("--save", default=None, help="optional output video, e.g. count_run.mp4")
    ap.add_argument("--vit", default=os.path.join(HERE, "weights/vit_areca_v3.pt"), help="ViT classifier weights ('' = off)")
    ap.add_argument("--vit-every", type=int, default=3, help="re-classify each nut every N frames (higher = faster)")
    ap.add_argument("--fps", type=int, default=30, help="camera frame rate to request")
    ap.add_argument("--send", default="127.0.0.1:5005",
                    help="host:port of receiver.py to send counted nuts to ('' = off)")
    args = ap.parse_args()

    # ---- link to the receiver camera (UDP, fire-and-forget: never slows this loop)
    udp, dest, sent_cls = None, None, {}
    if args.send:
        host, port = args.send.rsplit(":", 1)
        dest = (host, int(port))
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"sending counted nuts to receiver at {host}:{port}")

    def send(msg):
        if udp is not None:
            try:
                udp.sendto(json.dumps(msg).encode(), dest)
            except OSError:
                pass

    # ---- camera
    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if isinstance(src, int) else cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # MJPG = full fps on most USB cams
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read from camera/source {args.source!r}. Close other apps using the camera, or try --source 0")
    H, W = frame.shape[:2]
    print(f"camera {args.source}: {W}x{H} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps requested")

    # ---- saved line (calibration)
    saved = None
    if os.path.exists(args.config):
        try:
            cfg = json.load(open(args.config))
            if cfg.get("size") == [W, H]:
                saved = (tuple(cfg["line"][0]), tuple(cfg["line"][1]))
                print(f"loaded saved line {saved} from {args.config}  (press r to redraw)")
        except Exception:
            pass
    pen = LineDrawer(saved)
    last_saved_line = saved

    model = YOLO(args.weights)
    half = str(args.device) != "cpu"
    vit = None
    if args.vit and os.path.exists(args.vit):
        vit = ViTClassifier(args.vit, args.device)
        print(f"ViT classifier loaded: {vit.classes} on {vit.dev}")
    elif args.vit:
        print(f"(no ViT weights at {args.vit} - running without pure/kole/gottu labels)")
    probs_of = {}          # track id -> running average of class probabilities
    seen_frames = {}       # track id -> frames since it was last classified
    cam = Camera(cap) if isinstance(src, int) else None
    seq = 0
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(W, 1280), min(H, 720))
    cv2.setMouseCallback(WIN, pen.on_mouse)

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))

    count = 0
    class_of_number = {}  # count number -> latest ViT class of that nut
    number_of = {}        # track id -> count number given when it touched the line
    recent = []           # (time, box, number) of counted nuts, to survive ID switches
    last_box = {}
    t_prev, fps = time.time(), 0.0

    while True:
        if frame is None:
            if cam:
                ok, frame, seq = cam.read(seq)
            else:
                ok, frame = cap.read()
            if not ok:
                break
        res = model.track(frame, persist=True, tracker=args.tracker, conf=args.conf,
                          imgsz=args.imgsz, device=args.device, max_det=300, verbose=False)[0]
        vis = frame.copy()
        now = time.time()
        n_in_frame = 0

        if res.boxes is not None and len(res.boxes) and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy().astype(int)
            n_in_frame = len(ids)
            active = set(ids.tolist())
            owner = {num: t for t, num in number_of.items()}
            if vit is not None:  # ViT: classify only new tracks and every Nth frame per track, in one batch
                need = [i for i, t in enumerate(ids) if t not in probs_of or seen_frames.get(t, 0) >= args.vit_every]
                if need:
                    p = vit(frame, [boxes[i] for i in need])
                    for k, i in enumerate(need):
                        t = ids[i]
                        probs_of[t] = p[k] if t not in probs_of else 0.85 * probs_of[t] + 0.15 * p[k]  # steadier label
                        seen_frames[t] = 0
                for t in ids:
                    seen_frames[t] = seen_frames.get(t, 0) + 1
            for box, tid in zip(boxes, ids):
                b = tuple(int(v) for v in box)
                if tid not in number_of:
                    # same nut that just lost its tracker ID? -> keep its old number (never count twice)
                    inherited = None
                    for t, rb, num in reversed(recent):
                        if now - t < 2.0 and iou(b, rb) > 0.3 and owner.get(num) not in active:
                            inherited = num
                            break
                    if inherited is None and pen.line is not None and box_touches_line(b, pen.line):
                        count += 1
                        inherited = count
                        print(f"arecanut #{count} touched the line")
                    if inherited is not None:
                        number_of[tid] = inherited
                        owner[inherited] = tid
                last_box[tid] = b
                kind = ""
                if vit is not None and tid in probs_of:
                    j = int(np.argmax(probs_of[tid]))
                    kind = f"{vit.classes[j]} {probs_of[tid][j] * 100:.0f}%"
                    class_of_number[number_of.get(tid, -1)] = vit.classes[j]
                    n = number_of.get(tid)
                    if n is not None and sent_cls.get(n) != vit.classes[j]:  # new count, or ViT changed its mind
                        send({"type": "nut", "n": n, "cls": vit.classes[j], "conf": round(float(probs_of[tid][j]), 3)})
                        sent_cls[n] = vit.classes[j]
                elif vit is None and tid in number_of and number_of[tid] not in sent_cls:
                    send({"type": "nut", "n": number_of[tid], "cls": None, "conf": 0.0})
                    sent_cls[number_of[tid]] = None
                if tid in number_of:
                    recent.append((now, b, number_of[tid]))
                    cv2.rectangle(vis, b[:2], b[2:], (0, 220, 0), 3)
                    txt = f"{number_of[tid]} | {kind}" if kind else f"{number_of[tid]}"
                    bg = CLASS_COLORS.get(kind.split(" ")[0], (0, 220, 0)) if kind else (0, 220, 0)
                    label(vis, txt, b[0], b[1], bg, scale=1.0, thick=3)
                else:
                    cv2.rectangle(vis, b[:2], b[2:], (0, 200, 255), 2)
                    if kind:
                        label(vis, kind, b[0], b[1], (0, 200, 255), scale=0.7, thick=2)
        recent = [r for r in recent if now - r[0] < 2.0][-200:]
        if len(probs_of) > 500:  # forget old tracks
            live = set(number_of) | set(res.boxes.id.cpu().numpy().astype(int).tolist() if res.boxes is not None and res.boxes.id is not None else [])
            probs_of = {k: v for k, v in probs_of.items() if k in live}

        # ---- line + overlay
        if pen.line is not None:
            cv2.line(vis, pen.line[0], pen.line[1], (0, 0, 255), 4, cv2.LINE_AA)
            if pen.line != last_saved_line:
                json.dump({"line": [list(pen.line[0]), list(pen.line[1])], "size": [W, H]}, open(args.config, "w"))
                last_saved_line = pen.line
                print(f"line saved: {pen.line}")
        if pen.start is not None:
            cv2.line(vis, pen.start, pen.preview, (0, 0, 255), 2, cv2.LINE_AA)

        dt = now - t_prev
        t_prev = now
        fps = 0.9 * fps + 0.1 * (1 / dt if dt > 0 else 0)
        cv2.rectangle(vis, (0, 0), (W, 48), (0, 0, 0), -1)
        head = f"COUNT: {count}     in frame: {n_in_frame}     {fps:4.1f} fps"
        if vit is not None:
            tally = {c: 0 for c in vit.classes}
            for num in range(1, count + 1):
                c = class_of_number.get(num)
                if c in tally:
                    tally[c] += 1
            head += "     " + "  ".join(f"{c}: {n}" for c, n in tally.items())
        hs = min(1.0, (W - 24) / max(1, cv2.getTextSize(head, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0]))
        cv2.putText(vis, head, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, hs, (255, 255, 255), 2, cv2.LINE_AA)
        if pen.line is None:
            msg = "Draw the counting line: click, drag and release with the mouse"
            label(vis, msg, 12, H - 15, (0, 0, 255), fg=(255, 255, 255), scale=0.8)

        cv2.imshow(WIN, vis)
        if writer:
            writer.write(vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            pen.line = None
        if key == ord("c"):
            count, number_of, recent, class_of_number, sent_cls = 0, {}, [], {}, {}
            send({"type": "reset"})
            print("count reset")
        try:  # window closed with the X button
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_AUTOSIZE) < 0:
                break
        except cv2.error:
            break
        frame = None

    if cam:
        cam.stop = True
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"FINAL COUNT: {count}")
    if vit is not None:
        for c in vit.classes:
            print(f"  {c}: {sum(1 for n in range(1, count + 1) if class_of_number.get(n) == c)}")


if __name__ == "__main__":
    main()
