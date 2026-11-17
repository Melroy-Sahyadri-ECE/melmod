"""
RECEIVER camera: box-only detector + line in the middle + ESP32 sorting servo.

How it works
    main camera (line_count.py)  --UDP-->  receiver.py  --WiFi HTTP-->  ESP32 servo (pin 19)
       counts nut #n, finds its type          counts the same nuts in the same order;
       (pure / kole / gottu) and sends        when ITS nut #n touches ITS line, it looks up
       {"type":"nut","n":n,"cls":...}         the type of #n and sends the servo position:
                                                pure (bette) ->   0 deg
                                                kole         -> +40 deg
                                                gottu        -> -40 deg

Run (PC must be connected to the ESP32's WiFi "ESP32-Servo" for the servo part):
    python receiver.py --source 0 --esp 192.168.4.1
    python receiver.py --source 0 --esp ""          # dry run, no ESP32

Keys (click the window first): drag = move line | r = redraw line | c = reset count | q / Esc = quit
"""
import argparse
import json
import os
import queue
import socket
import threading
import time
import urllib.request

import cv2
import numpy as np
from ultralytics import YOLO

from line_count import Camera, LineDrawer, box_touches_line, iou, label

HERE = os.path.dirname(os.path.abspath(__file__))
WIN = "Receiver - sorting camera  (drag = line | r = redraw | c = reset | q = quit)"
POS = {"pure": 0, "bette": 0, "kole": 40, "gottu": -40}           # servo position per type
COLORS = {"pure": (0, 220, 0), "bette": (0, 220, 0), "kole": (40, 40, 230), "gottu": (0, 200, 255)}


class Inbox:
    """Receives {"type":"nut","n":..,"cls":..} / {"type":"reset"} from the main camera."""

    def __init__(self, listen):
        host, port = listen.rsplit(":", 1)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, int(port)))
        self.sock.settimeout(0.5)
        self.types = {}          # n -> (cls, conf)
        self.last_n = 0
        self.reset_flag = False
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()
        print(f"listening for the main camera on {host}:{port}")

    def _run(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
                msg = json.loads(data.decode())
            except (socket.timeout, ValueError, OSError):
                continue
            with self.lock:
                if msg.get("type") == "reset":
                    self.types.clear()
                    self.last_n = 0
                    self.reset_flag = True
                elif msg.get("type") == "nut":
                    n = int(msg["n"])
                    self.types[n] = (msg.get("cls"), float(msg.get("conf", 0)))
                    self.last_n = max(self.last_n, n)

    def get(self, n):
        with self.lock:
            return self.types.get(n)


class ESP32:
    """Sends sort commands in a background thread so the camera loop never waits on WiFi."""

    def __init__(self, host):
        self.host = host
        self.q = queue.Queue()
        self.status = "dry run (no ESP32)" if not host else "waiting"
        if host:
            threading.Thread(target=self._run, daemon=True).start()

    def sort(self, n, cls):
        if not self.host:
            self.status = f"dry run: #{n} {cls} {POS.get(cls, 0):+d} deg"
            return
        self.q.put((n, cls))

    def _run(self):
        while True:
            n, cls = self.q.get()
            url = f"http://{self.host}/sort?type={cls}&n={n}"
            try:
                with urllib.request.urlopen(url, timeout=1.5) as r:
                    self.status = f"ESP32 OK: #{n} {cls} {POS.get(cls, 0):+d} deg"
                    r.read()
            except Exception as e:  # noqa: BLE001 - show any network problem on screen
                self.status = f"ESP32 ERROR #{n}: {type(e).__name__} (connected to ESP32-Servo WiFi?)"
            print(self.status)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(HERE, "weights/receiver_yolo11n.pt"))
    ap.add_argument("--source", default="0", help="camera index of the RECEIVER camera, or a video file")
    ap.add_argument("--device", default="cpu", help="cpu or 0 for the NVIDIA GPU")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--tracker", default=os.path.join(HERE, "areca_bytetrack.yaml"))
    ap.add_argument("--config", default=os.path.join(HERE, "receiver_line.json"), help="where the line is saved")
    ap.add_argument("--line", choices=["vertical", "horizontal"], default="vertical",
                    help="default line through the middle of the frame (before you draw your own)")
    ap.add_argument("--listen", default="0.0.0.0:5005", help="where line_count.py sends counted nuts")
    ap.add_argument("--esp", default="192.168.4.1", help="ESP32 address ('' = dry run)")
    ap.add_argument("--wait", type=float, default=3.0,
                    help="seconds to wait for a nut's type from the main camera before using --unknown")
    ap.add_argument("--unknown", default="pure", choices=["pure", "kole", "gottu", "skip"],
                    help="what to do when a nut's type never arrives")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    inbox = Inbox(args.listen)
    esp = ESP32(args.esp)

    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if isinstance(src, int) else cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read receiver camera {args.source!r} (try --source 1 / 2)")
    H, W = frame.shape[:2]

    # line in the middle by default; a line you draw is saved and reused
    line = ((W // 2, 0), (W // 2, H - 1)) if args.line == "vertical" else ((0, H // 2), (W - 1, H // 2))
    if os.path.exists(args.config):
        try:
            cfg = json.load(open(args.config))
            if cfg.get("size") == [W, H]:
                line = (tuple(cfg["line"][0]), tuple(cfg["line"][1]))
        except Exception:
            pass
    pen = LineDrawer(line)
    saved_line = line
    print(f"receiver camera {args.source}: {W}x{H}, line {line}")

    model = YOLO(args.weights)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(W, 1280), min(H, 720))
    cv2.setMouseCallback(WIN, pen.on_mouse)
    cam = Camera(cap) if isinstance(src, int) else None
    seq = 0

    count = 0
    number_of, recent = {}, []
    pending = {}          # n -> time it touched the line (type not yet known)
    action_of = {}        # n -> (cls, pos) that was sent to the servo
    t_prev, fps = time.time(), 0.0

    while True:
        if frame is None:
            ok, frame, seq = cam.read(seq) if cam else (*cap.read(), 0)
            if not ok:
                break
        if inbox.reset_flag:  # main camera pressed "c"
            inbox.reset_flag = False
            count, number_of, recent, pending, action_of = 0, {}, [], {}, {}
            print("reset from main camera")

        res = model.track(frame, persist=True, tracker=args.tracker, conf=args.conf, imgsz=args.imgsz,
                          device=args.device, max_det=300, verbose=False)[0]
        vis, now, n_in = frame.copy(), time.time(), 0

        if res.boxes is not None and len(res.boxes) and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy().astype(int)
            n_in = len(ids)
            active = set(ids.tolist())
            owner = {num: t for t, num in number_of.items()}
            for box, tid in zip(boxes, ids):
                b = tuple(int(v) for v in box)
                if tid not in number_of:
                    got = None
                    for t, rb, num in reversed(recent):  # same nut after a tracker ID switch
                        if now - t < 2.0 and iou(b, rb) > 0.3 and owner.get(num) not in active:
                            got = num
                            break
                    if got is None and pen.line is not None and box_touches_line(b, pen.line):
                        count += 1
                        got = count
                        pending[count] = now
                        print(f"receiver: nut #{count} at the line")
                    if got is not None:
                        number_of[tid] = got
                        owner[got] = tid
                if tid in number_of:
                    n = number_of[tid]
                    recent.append((now, b, n))
                    if n in action_of:
                        cls, pos = action_of[n]
                        txt, col = f"{n} | {cls} {pos:+d} deg", COLORS.get(cls, (0, 220, 0))
                    else:
                        txt, col = f"{n} | waiting for type", (200, 200, 200)
                    cv2.rectangle(vis, b[:2], b[2:], col, 3)
                    label(vis, txt, b[0], b[1], col, scale=0.9, thick=2)
                else:
                    cv2.rectangle(vis, b[:2], b[2:], (0, 200, 255), 2)
        recent = [r for r in recent if now - r[0] < 2.0][-200:]

        # send servo commands: as soon as the main camera has told us this nut's type
        for n in sorted(list(pending)):
            info = inbox.get(n)
            if info and info[0]:
                cls = info[0]
            elif now - pending[n] > args.wait:
                if args.unknown == "skip":
                    print(f"receiver: no type for #{n} after {args.wait}s - skipped")
                    del pending[n]
                    continue
                cls = args.unknown
                print(f"receiver: no type for #{n} after {args.wait}s - using {cls}")
            else:
                continue
            action_of[n] = (cls, POS.get(cls, 0))
            esp.sort(n, cls)
            del pending[n]

        # line + header
        if pen.line is not None:
            cv2.line(vis, pen.line[0], pen.line[1], (0, 0, 255), 4, cv2.LINE_AA)
            if pen.line != saved_line:
                json.dump({"line": [list(pen.line[0]), list(pen.line[1])], "size": [W, H]}, open(args.config, "w"))
                saved_line = pen.line
        if pen.start is not None:
            cv2.line(vis, pen.start, pen.preview, (0, 0, 255), 2, cv2.LINE_AA)
        dt, t_prev = now - t_prev, now
        fps = 0.9 * fps + 0.1 * (1 / dt if dt > 0 else 0)
        head = f"RX COUNT: {count}   main sent: {inbox.last_n}   in frame: {n_in}   {fps:4.1f} fps"
        cv2.rectangle(vis, (0, 0), (W, 80), (0, 0, 0), -1)
        hs = min(1.0, (W - 24) / max(1, cv2.getTextSize(head, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0]))
        cv2.putText(vis, head, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, hs, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, esp.status, (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 80, 255) if "ERROR" in esp.status else (120, 255, 120), 2, cv2.LINE_AA)

        cv2.imshow(WIN, vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            pen.line = None
        if key == ord("c"):
            count, number_of, recent, pending, action_of = 0, {}, [], {}, {}
            print("receiver count reset")
        try:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_AUTOSIZE) < 0:
                break
        except cv2.error:
            break
        frame = None

    if cam:
        cam.stop = True
    cap.release()
    cv2.destroyAllWindows()
    print(f"receiver: {count} nuts, sorted: " +
          ", ".join(f"{c}={sum(1 for v in action_of.values() if v[0] == c)}" for c in ("pure", "kole", "gottu")))


if __name__ == "__main__":
    main()
