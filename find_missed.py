"""
Run the detector on every ORIGINAL dataset photo (not the composites) and report which
nuts it misses or boxes poorly, per type (gottu / kole / pure).

    python find_missed.py --weights runs\\detect\\runs\\areca\\arecanut_yolo11\\weights\\best.pt --data ..\\areca_yolo
Writes missed_<tag>.jpg (picture of the misses) and missed_<tag>.txt (file list).
"""
import argparse
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

CLASSES = ["gottu", "kole", "pure"]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="../areca_yolo")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="before")
    args = ap.parse_args()

    model = YOLO(args.weights)
    stats = {c: [0, 0] for c in CLASSES}  # found, total
    missed = []
    for split in ("train", "val", "test"):
        for lab in sorted(glob.glob(os.path.join(args.data, "labels", split, "*.txt"))):
            name = os.path.splitext(os.path.basename(lab))[0]
            cls = name.split("_", 1)[0]
            if cls not in CLASSES:
                continue
            img = cv2.imread(os.path.join(args.data, "images", split, name + ".jpg"))
            H, W = img.shape[:2]
            gts = []
            for line in open(lab):
                _, cx, cy, w, h = map(float, line.split())
                gts.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
            r = model.predict(img, conf=args.conf, device=args.device, verbose=False)[0]
            dets = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
            for g in gts:
                stats[cls][1] += 1
                best = max([iou(g, d) for d in dets], default=0.0)
                if best >= 0.3:  # found (box overlaps the nut well enough)
                    stats[cls][0] += 1
                else:
                    missed.append((split, cls, name, best, img, g))

    print(f"\nDetector check on {sum(t for _, t in stats.values())} dataset nuts (conf {args.conf}):")
    for c, (f, t) in stats.items():
        print(f"  {c:<6} found {f}/{t}  ({100 * f / max(1, t):.1f}%)   missed {t - f}")
    with open(f"missed_{args.tag}.txt", "w") as fh:
        for split, cls, name, best, _, _ in missed:
            fh.write(f"{split}\t{cls}\t{name}\tbest_iou={best:.2f}\n")
    if missed:
        tiles = []
        for split, cls, name, best, img, g in missed[:40]:
            t = img.copy()
            cv2.rectangle(t, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 0, 255), 4)
            cv2.putText(t, f"{cls} {split}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
            tiles.append(cv2.resize(t, (240, 240)))
        while len(tiles) % 8:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)])
        cv2.imwrite(f"missed_{args.tag}.jpg", sheet)
        print(f"  pictures of the misses: missed_{args.tag}.jpg   list: missed_{args.tag}.txt")


if __name__ == "__main__":
    main()
