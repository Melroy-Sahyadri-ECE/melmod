"""
Create synthetic multi-arecanut training images.

The original dataset has exactly ONE nut per photo. A detector trained only on that
tends to under-fire when many nuts share a frame (piles, conveyor, tray). This script
cuts each training nut out using its circle mask and pastes 2-8 nuts (random class,
size, rotation, brightness, soft shadow) onto real floor backgrounds, writing tight
YOLO boxes for every pasted nut.

Only the TRAIN split is used as a source, so val/test stay untouched (no leakage).

Usage (after prepare_dataset.py):
    python make_composites.py --data areca_yolo --n 1200
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np

from backgrounds import Backgrounds
from make_vit_seg_crops import refine

SOURCE_FOLDERS = ["gottu", "kole", "pure"]  # only used to mix nut types evenly
CLASSES = ["arecanut"]  # single class


# ---------------------------------------------------------------- helpers

def refined_mask(img, mask):
    """Hand-drawn circles -> each nut's real outline (GrabCut), as a 0/1 mask."""
    binary = (mask > 127).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary)
    out = np.zeros_like(binary)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 0.0005 * mask.size:
            continue
        out |= (refine(img, (lab == i).astype(np.uint8) * 255, (x, y, x + w, y + h)) > 0).astype(np.uint8)
    return out


def load_sprites(data_dir, split="train"):
    """Return list of (folder_id, bgr_crop, alpha_float) cut out with each nut's mask."""
    sprites = []
    for mask_path in sorted(glob.glob(f"{data_dir}/masks/{split}/*.png")):
        name = os.path.splitext(os.path.basename(mask_path))[0]
        cls_id = SOURCE_FOLDERS.index(name.split("_", 1)[0])
        img = cv2.imread(f"{data_dir}/images/{split}/{name}.jpg")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        binary = refined_mask(img, mask)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(binary)
        for i in range(1, n):
            x, y, w, h, _ = stats[i]
            pad = 6
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
            blob = (lab[y1:y2, x1:x2] == i).astype(np.uint8) * 255
            # hand-drawn circles are often a bit bigger than the nut -> shrink slightly
            # so we don't paste a ring of the old floor, then feather the edge
            blob = cv2.erode(blob, np.ones((3, 3), np.uint8), iterations=1)
            alpha = cv2.GaussianBlur(blob, (7, 7), 0).astype(np.float32) / 255.0
            sprites.append((cls_id, img[y1:y2, x1:x2].copy(), alpha))
    return sprites


def clean_background(img, mask, rng=None):
    """Remove the nut (and its shadow) from a photo -> empty floor.

    Copies a nut-free floor patch from elsewhere in the same photo over the nut and
    blends it with Poisson (seamless) cloning, so the floor texture stays real.
    Falls back to inpainting only if the nut is too big to find a free patch.
    """
    rng = rng or random.Random(0)
    H, W = img.shape[:2]
    hole = cv2.dilate((mask > 127).astype(np.uint8) * 255, np.ones((31, 31), np.uint8))
    x, y, w, h = cv2.boundingRect(hole)
    m = 6  # margin so the clone mask never touches the patch border
    x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)
    pw, ph = x1 - x0, y1 - y0
    cands = []
    for sy in range(0, H - ph + 1, 16):
        for sx in range(0, W - pw + 1, 16):
            if hole[sy:sy + ph, sx:sx + pw].any():
                continue
            cands.append((sx, sy))
    if not cands:
        return cv2.inpaint(img, hole, 7, cv2.INPAINT_TELEA)
    sx, sy = rng.choice(cands)
    patch = img[sy:sy + ph, sx:sx + pw]
    clone_mask = np.zeros((ph, pw), np.uint8)
    clone_mask[2:-2, 2:-2] = 255  # whole patch, minus a 2px rim (required by seamlessClone)
    center = (x0 + pw // 2, y0 + ph // 2)
    return cv2.seamlessClone(patch, img, clone_mask, center, cv2.NORMAL_CLONE)


def transform_sprite(bgr, alpha, rng, scale):
    """Random rotation + scale + brightness. Returns new (bgr, alpha)."""
    h, w = alpha.shape
    angle = rng.uniform(0, 360)
    diag = int(np.ceil(np.hypot(h, w) * scale)) + 2
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += diag / 2 - w / 2
    M[1, 2] += diag / 2 - h / 2
    bgr_t = cv2.warpAffine(bgr, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    a_t = cv2.warpAffine(alpha, M, (diag, diag), flags=cv2.INTER_LINEAR, borderValue=0)
    gain = rng.uniform(0.85, 1.15)
    bgr_t = np.clip(bgr_t.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    # crop to the visible nut
    ys, xs = np.where(a_t > 0.02)
    if len(xs) == 0:
        return None, None
    return bgr_t[ys.min():ys.max() + 1, xs.min():xs.max() + 1], a_t[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def paste(canvas, bgr, alpha, x, y, shadow=True):
    """Alpha-blend sprite at top-left (x, y). Returns tight box of the nut (alpha>0.5)."""
    h, w = alpha.shape
    roi = canvas[y:y + h, x:x + w].astype(np.float32)
    if shadow:  # soft drop shadow, offset down-right like the real photos
        sh = cv2.GaussianBlur(alpha, (0, 0), max(2.0, w / 25))
        sh = np.roll(np.roll(sh, max(2, h // 18), 0), max(1, w // 30), 1) * 0.35
        roi *= (1 - sh[..., None])
    a = alpha[..., None]
    roi = roi * (1 - a) + bgr.astype(np.float32) * a
    canvas[y:y + h, x:x + w] = roi.astype(np.uint8)
    ys, xs = np.where(alpha > 0.5)
    return (x + xs.min(), y + ys.min(), x + xs.max() + 1, y + ys.max() + 1)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def overlap(a, b):
    """max(IoU, intersection / smaller-box-area) — catches a small nut inside a big one."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return max(iou(a, b), inter / small if small else 0.0)


def read_yolo(label_path, w, h):
    out = []
    if os.path.exists(label_path):
        for line in open(label_path):
            c, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            out.append((int(c), (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)))
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="areca_yolo")
    ap.add_argument("--n", type=int, default=1200, help="number of composite images")
    ap.add_argument("--min-nuts", type=int, default=2)
    ap.add_argument("--max-nuts", type=int, default=8)
    ap.add_argument("--neg-frac", type=float, default=0.04, help="fraction of empty-floor images (no nuts)")
    ap.add_argument("--max-iou", type=float, default=0.25, help="allowed overlap between nuts (touching piles)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kole-weight", type=float, default=2.0, help="how much more often kole nuts are pasted")
    ap.add_argument("--synth-bg-frac", type=float, default=0.4, help="fraction of images on new backgrounds (red ridges etc.)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sprites = load_sprites(args.data)
    by_folder = {c: [s for s in sprites if s[0] == c] for c in range(len(SOURCE_FOLDERS))}
    folder_w = [args.kole_weight if f == "kole" else 1.0 for f in SOURCE_FOLDERS]
    bgs = Backgrounds(Backgrounds.floors_from_dataset(f"{args.data}/images/train", f"{args.data}/masks/train", 150, args.seed), seed=args.seed + 1)
    names = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(f"{args.data}/masks/train/*.png"))]
    print(f"{len(sprites)} nut sprites from train split")

    img_dir, lab_dir = f"{args.data}/images/train", f"{args.data}/labels/train"
    bg_cache = {}
    total_boxes = 0
    for k in range(args.n):
        base = rng.choice(names)
        img = cv2.imread(f"{img_dir}/{base}.jpg")
        H, W = img.shape[:2]
        negative = rng.random() < args.neg_frac
        keep_original = (not negative) and rng.random() < 0.5
        if keep_original:  # keep the real nut and its label, add more around it
            canvas = img.copy()
            objs = read_yolo(f"{lab_dir}/{base}.txt", W, H)
        else:
            if base not in bg_cache:
                bg_cache[base] = clean_background(img, cv2.imread(f"{args.data}/masks/train/{base}.png", 0), rng)
            canvas = bg_cache[base].copy()
            if rng.random() < args.synth_bg_frac:  # new surface: red ridges, blue edge, plain colours ...
                canvas = bgs.sample(H, W)
            objs = []
        if rng.random() < 0.5:
            canvas = cv2.flip(canvas, 1)
            objs = [(c, (W - b[2], b[1], W - b[0], b[3])) for c, b in objs]

        if not negative:
            target = rng.randint(args.min_nuts, args.max_nuts)
            # smaller nuts when the frame is crowded so they fit
            scale_hi = 1.25 if target <= 4 else 0.95
            attempts = 0
            while len(objs) < target and attempts < 60:
                attempts += 1
                folder = rng.choices(range(len(SOURCE_FOLDERS)), folder_w)[0]  # extra kole
                _, bgr, alpha = rng.choice(by_folder[folder])
                bgr_t, a_t = transform_sprite(bgr, alpha, rng, rng.uniform(0.55, scale_hi))
                if bgr_t is None:
                    continue
                h, w = a_t.shape
                if w >= W - 4 or h >= H - 4:
                    continue
                x, y = rng.randint(0, W - w), rng.randint(0, H - h)
                cand = (x, y, x + w, y + h)
                if any(overlap(cand, b) > args.max_iou for _, b in objs):
                    continue
                box = paste(canvas, bgr_t, a_t, x, y)
                objs.append((0, box))

        name = f"comp_{k:05d}"
        cv2.imwrite(f"{img_dir}/{name}.jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(f"{lab_dir}/{name}.txt", "w") as f:
            for c, (x1, y1, x2, y2) in objs:
                f.write("%d %.6f %.6f %.6f %.6f\n" % (c, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
                total_boxes += 1
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{args.n}")

    print(f"wrote {args.n} composites with {total_boxes} arecanuts")


if __name__ == "__main__":
    main()
