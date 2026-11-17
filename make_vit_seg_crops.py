"""
Segmented ViT dataset: cut every nut out with its mask and paste it on many different
backgrounds, so the classifier learns the nut (husk / skin / shell) and not the floor.

Same train/val/test split as the detector (areca_yolo). Output (ImageFolder layout):
    <out>/train/{gottu,kole,pure}/      nut on random backgrounds (several per nut) + original crop
    <out>/val/...                       original crops (real background)  -> picks the best epoch
    <out>/test/...                      original crops (real background)
    <out>/test_swap/...                 test nuts on NEW random backgrounds -> proves it ignores the floor

    python make_vit_seg_crops.py --data ..\\areca_yolo --out ..\\areca_vit
"""
import argparse
import glob
import os
import random
import shutil

import cv2
import numpy as np

from backgrounds import Backgrounds

CLASSES = ["gottu", "kole", "pure"]


def refine(img, blob, box):
    """GrabCut inside the hand-drawn circle -> the nut's real outline (circles are often too big)."""
    x1, y1, x2, y2 = box
    s = max(x2 - x1, y2 - y1)
    m = int(0.25 * s)
    H, W = blob.shape
    X1, Y1, X2, Y2 = max(0, x1 - m), max(0, y1 - m), min(W, x2 + m), min(H, y2 + m)
    sub, b = img[Y1:Y2, X1:X2], blob[Y1:Y2, X1:X2]
    gc = np.full(b.shape, cv2.GC_BGD, np.uint8)
    gc[cv2.dilate(b, np.ones((9, 9), np.uint8)) > 0] = cv2.GC_PR_BGD   # just around the circle
    gc[b > 0] = cv2.GC_PR_FGD                                           # inside the circle
    core = cv2.erode(b, np.ones((max(3, int(0.45 * s)) | 1,) * 2, np.uint8))
    gc[core > 0] = cv2.GC_FGD                                           # centre is surely nut
    try:
        cv2.grabCut(sub, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return blob
    fg = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)                 # keep the biggest piece
    if n < 2:
        return blob
    fg = (lab == 1 + int(np.argmax(st[1:, 4]))).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    area, circle = (fg > 0).sum(), (b > 0).sum()
    if not (0.3 * circle < area <= 1.05 * circle):                       # GrabCut failed -> fallback
        return blob
    out = np.zeros_like(blob)
    out[Y1:Y2, X1:X2] = fg
    return out


def nut_cutouts(img, mask):
    """Yield (bgr, alpha, box) for every nut blob in the mask."""
    binary = (mask > 127).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 0.0005 * mask.size:
            continue
        blob = (lab == i).astype(np.uint8) * 255
        blob = refine(img, blob, (x, y, x + w, y + h))
        ys, xs = np.nonzero(blob)
        x, y, w, h = xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        k = max(3, int(0.02 * max(w, h)) | 1)
        blob = cv2.erode(blob, np.ones((k, k), np.uint8))
        alpha = cv2.GaussianBlur(blob, (0, 0), max(1.0, 0.015 * max(w, h))).astype(np.float32) / 255
        yield img, alpha, (x, y, x + w, y + h)


def square_crop(img, box, pad):
    H, W = img.shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = max(box[2] - box[0], box[3] - box[1]) * (1 + 2 * pad)
    x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x2, y2 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    return x1, y1, x2, y2


def on_background(img, alpha, box, bgs, rng, size=224):
    """Nut (from img, cut by alpha) on a random background, framed like a live detector crop."""
    pad = rng.uniform(0.05, 0.3)                       # loose / tight boxes
    x1, y1, x2, y2 = square_crop(img, box, pad)
    nut, a = img[y1:y2, x1:x2].astype(np.float32), alpha[y1:y2, x1:x2]
    h, w = a.shape
    bg = bgs.sample(h, w).astype(np.float32)
    if rng.random() < 0.7:                              # soft contact shadow under the nut
        sh = cv2.GaussianBlur(a, (0, 0), max(2.0, w / 18))
        sh = np.roll(np.roll(sh, int(h * rng.uniform(0.02, 0.07)), 0), int(w * rng.uniform(-0.04, 0.04)), 1)
        bg *= 1 - rng.uniform(0.2, 0.45) * sh[..., None]
    nut *= rng.uniform(0.75, 1.25)                     # lighting on the nut
    out = bg * (1 - a[..., None]) + nut * a[..., None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:                             # camera blur / out of focus
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.6, 2.2))
    if rng.random() < 0.3:                              # sensor noise
        out = np.clip(out + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 8), out.shape), 0, 255).astype(np.uint8)
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)


def original(img, box, size=224, pad=0.15):
    x1, y1, x2, y2 = square_crop(img, box, pad)
    return cv2.resize(img[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../areca_yolo")
    ap.add_argument("--out", default="../areca_vit")
    ap.add_argument("--per-nut", type=int, default=5, help="random-background copies of each training nut")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    # backgrounds for TRAIN come from train floors; test_swap uses its own seed (and val/test floors)
    bg_train = Backgrounds(Backgrounds.floors_from_dataset(f"{args.data}/images/train", f"{args.data}/masks/train", 150, 1), seed=2)
    bg_test = Backgrounds(Backgrounds.floors_from_dataset(f"{args.data}/images/test", f"{args.data}/masks/test", 60, 3), seed=4)

    counts = {}
    for split in ("train", "val", "test"):
        outs = ["train"] if split == "train" else ([split, "test_swap"] if split == "test" else [split])
        for o in outs:
            for c in CLASSES:
                os.makedirs(os.path.join(args.out, o, c), exist_ok=True)
        for mp in sorted(glob.glob(f"{args.data}/masks/{split}/*.png")):
            name = os.path.splitext(os.path.basename(mp))[0]
            cls = name.split("_", 1)[0]
            if cls not in CLASSES:
                continue
            img = cv2.imread(f"{args.data}/images/{split}/{name}.jpg")
            mask = cv2.imread(mp, 0)
            if img is None or mask is None:
                continue
            for k, (im, alpha, box) in enumerate(nut_cutouts(img, mask)):
                def save(o, tag, crop):
                    cv2.imwrite(os.path.join(args.out, o, cls, f"{name}_{k}_{tag}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 93])
                    counts[(o, cls)] = counts.get((o, cls), 0) + 1
                if split == "train":
                    save("train", "orig", original(im, box))
                    for j in range(args.per_nut):
                        save("train", f"bg{j}", on_background(im, alpha, box, bg_train, rng))
                else:
                    save(split, "orig", original(im, box))
                    if split == "test":
                        save("test_swap", "bg", on_background(im, alpha, box, bg_test, rng))

    print(f"{'split':<10}" + "".join(f"{c:>8}" for c in CLASSES))
    for o in ("train", "val", "test", "test_swap"):
        print(f"{o:<10}" + "".join(f"{counts.get((o, c), 0):>8}" for c in CLASSES))
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
