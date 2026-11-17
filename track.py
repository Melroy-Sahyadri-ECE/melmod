"""
Detect + track every arecanut in a video / webcam / RTSP stream.

Each nut gets a box and a persistent ID (#1, #2, ...) that stays with it while it
moves. Shows how many nuts are in the frame and how many unique nuts were seen.
Optional counting line for conveyor belts.

Examples
    python track.py --weights best.pt --source test_video.mp4            # saves runs/track/test_video_tracked.mp4
    python track.py --weights best.pt --source 0 --show                  # webcam, live window
    python track.py --weights best.pt --source rtsp://... --show
    python track.py --weights best.pt --source belt.mp4 --line 0.5       # count nuts crossing the middle line
    python track.py --weights best.pt --source in.mp4 --csv tracks.csv   # log every box per frame
"""
import argparse
import colorsys
import csv
import os
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))


def id_color(track_id):
    """Stable, distinct BGR color per ID."""
    h = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_label(img, text, x, y, color):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(HERE, "weights/detector_v3.pt"))
    ap.add_argument("--source", default="0", help="video file, folder, webcam index (0), or rtsp/http URL")
    ap.add_argument("--tracker", default=os.path.join(HERE, "areca_bytetrack.yaml"),
                    help="areca_bytetrack.yaml (fast) or botsort.yaml (better with camera motion)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--device", default=None)
    ap.add_argument("--line", type=float, default=None,
                    help="horizontal counting line at this fraction of frame height (e.g. 0.5)")
    ap.add_argument("--trail", type=int, default=30, help="length of the motion trail (0 = off)")
    ap.add_argument("--show", action="store_true", help="show a live window (needs a display)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--out", default="runs/track")
    ap.add_argument("--csv", default=None, help="write frame,id,x1,y1,x2,y2,conf rows here")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.weights)

    fps = 30.0
    if isinstance(source, str) and os.path.isfile(source):
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

    trails = defaultdict(lambda: deque(maxlen=max(1, args.trail)))
    seen_ids = set()
    crossed = set()
    last_cy = {}
    writer = None
    csv_file = csv.writer(open(args.csv, "w", newline="")) if args.csv else None
    if csv_file:
        csv_file.writerow(["frame", "id", "x1", "y1", "x2", "y2", "conf"])

    results = model.track(source=source, stream=True, persist=True, tracker=args.tracker,
                          conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device,
                          max_det=300, verbose=False)

    frame_idx = -1
    for frame_idx, r in enumerate(results):
        frame = r.orig_img.copy()
        H, W = frame.shape[:2]
        line_y = int(args.line * H) if args.line is not None else None
        boxes = r.boxes
        n_now = 0

        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
            n_now = len(xyxy)
            for (x1, y1, x2, y2), c, tid in zip(xyxy, confs, ids):
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if tid is None:  # detected but not yet confirmed as a track
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
                    continue
                color = id_color(tid)
                seen_ids.add(tid)
                trails[tid].append((cx, cy))
                if line_y is not None and tid in last_cy and tid not in crossed:
                    if (last_cy[tid] - line_y) * (cy - line_y) <= 0 and last_cy[tid] != cy:
                        crossed.add(tid)
                last_cy[tid] = cy

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_label(frame, f"#{tid} {c:.2f}", x1, y1, color)
                if args.trail and len(trails[tid]) > 1:
                    pts = list(trails[tid])
                    for a, b in zip(pts[:-1], pts[1:]):
                        cv2.line(frame, a, b, color, 2)
                if csv_file:
                    csv_file.writerow([frame_idx, tid, x1, y1, x2, y2, round(float(c), 3)])

        # overlay
        panel = f"In frame: {n_now}   Unique nuts: {len(seen_ids)}"
        if line_y is not None:
            cv2.line(frame, (0, line_y), (W, line_y), (0, 255, 255), 2)
            panel += f"   Crossed line: {len(crossed)}"
        cv2.rectangle(frame, (0, 0), (W, 34), (0, 0, 0), -1)
        cv2.putText(frame, panel, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if not args.no_save:
            if writer is None:
                os.makedirs(args.out, exist_ok=True)
                stem = os.path.splitext(os.path.basename(str(args.source)))[0] or "stream"
                out_path = os.path.join(args.out, f"{stem}_tracked.mp4")
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
                print(f"saving -> {out_path}")
            writer.write(frame)
        if args.show:
            cv2.imshow("arecanut tracking (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"done: {frame_idx + 1} frames, {len(seen_ids)} unique arecanuts"
          + (f", {len(crossed)} crossed the line" if args.line is not None else ""))


if __name__ == "__main__":
    main()
