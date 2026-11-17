"""
Make a synthetic test video with many moving arecanuts (from the TEST split only,
which the model never trained on) to check detection + ID tracking.

Nuts drift, cross paths, bounce off walls and some enter/leave the frame, so you can
see whether IDs stay stable.

    python make_test_video.py --data areca_yolo --out test_video.mp4
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np

from make_composites import clean_background, load_sprites, transform_sprite


def paste_clipped(canvas, bgr, alpha, x, y):
    """Alpha-blend a sprite whose top-left may be partly outside the frame."""
    H, W = canvas.shape[:2]
    h, w = alpha.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    sx0, sy0 = x0 - x, y0 - y
    a = alpha[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)][..., None]
    s = bgr[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)].astype(np.float32)
    roi = canvas[y0:y1, x0:x1].astype(np.float32)
    sh = cv2.GaussianBlur(a[..., 0], (0, 0), 4)[..., None] * 0.3  # soft shadow
    roi = roi * (1 - sh)
    canvas[y0:y1, x0:x1] = (roi * (1 - a) + s * a).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="areca_yolo")
    ap.add_argument("--out", default="test_video.mp4")
    ap.add_argument("--nuts", type=int, default=9)
    ap.add_argument("--seconds", type=float, default=12)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sprites = load_sprites(args.data, split="test")
    masks = sorted(glob.glob(f"{args.data}/masks/test/*.png"))
    base = os.path.splitext(os.path.basename(rng.choice(masks)))[0]
    img = cv2.imread(f"{args.data}/images/test/{base}.jpg")
    bg = clean_background(img, cv2.imread(f"{args.data}/masks/test/{base}.png", 0), rng)
    bg = cv2.resize(bg, (args.size, args.size))
    S = args.size

    nuts = []
    for i in range(args.nuts):
        _, bgr, alpha = rng.choice(sprites)
        bgr, alpha = transform_sprite(bgr, alpha, rng, rng.uniform(0.55, 0.85))
        h, w = alpha.shape
        speed = rng.uniform(1.5, 4.0)
        ang = rng.uniform(0, 2 * np.pi)
        enter = i >= args.nuts - 2  # last two nuts enter from outside the frame
        x = -w if enter else rng.uniform(0, S - w)
        y = rng.uniform(0, S - h)
        nuts.append(dict(bgr=bgr, alpha=alpha, x=x, y=y, vx=abs(speed * np.cos(ang)) if enter else speed * np.cos(ang),
                         vy=speed * np.sin(ang), enter=enter, start=rng.randint(30, 150) if enter else 0))

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (S, S))
    n_frames = int(args.seconds * args.fps)
    for f in range(n_frames):
        frame = bg.copy()
        for n in nuts:
            if f < n["start"]:
                continue
            h, w = n["alpha"].shape
            n["x"] += n["vx"]
            n["y"] += n["vy"]
            if not n["enter"] or n["x"] > 0:  # bounce once fully inside
                n["enter"] = False
                if n["x"] < 0 or n["x"] > S - w:
                    n["vx"] *= -1
                if n["y"] < 0 or n["y"] > S - h:
                    n["vy"] *= -1
                n["x"] = min(max(n["x"], 0), S - w)
                n["y"] = min(max(n["y"], 0), S - h)
            paste_clipped(frame, n["bgr"], n["alpha"], int(n["x"]), int(n["y"]))
        # mild camera noise so frames aren't identical
        noise = np.random.default_rng(f).normal(0, 2.0, frame.shape).astype(np.int16)
        writer.write(np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8))
    writer.release()
    print(f"wrote {args.out}: {n_frames} frames, {args.nuts} nuts ({args.fps} fps, {S}x{S})")


if __name__ == "__main__":
    main()
