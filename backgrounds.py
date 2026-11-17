"""
Random backgrounds for training, so models learn the NUT and not the floor.

Kinds: real floor patches from the dataset (nut removed), red/brown corrugated
ridges (like the live camera surface), plain colours, gradients, noise textures,
light-blue painted edges, and mixes of two of these. Everything is randomised.
"""
import glob
import os
import random

import cv2
import numpy as np


def _ridges(h, w, rng):
    """Corrugated sheet: shaded sine ridges, any angle, red/brown/grey/blue tints."""
    period = rng.uniform(10, 45)
    angle = rng.uniform(-35, 35) + (90 if rng.random() < 0.3 else 0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = np.deg2rad(angle)
    u = xx * np.sin(t) + yy * np.cos(t)
    wave = 0.5 + 0.5 * np.sin(2 * np.pi * u / period)
    wave = wave ** rng.uniform(0.6, 1.8)  # sharper / softer ridges
    base = np.array(rng.choice([
        (40, 60, 170), (50, 70, 150), (35, 45, 120), (60, 90, 190),    # reds/browns (BGR)
        (90, 90, 95), (120, 120, 125), (150, 140, 110), (170, 150, 120),  # greys / blue-grey
    ]), np.float32) * rng.uniform(0.7, 1.25)
    img = base[None, None, :] * (0.55 + 0.6 * wave[..., None])
    return img


def _noise_floor(h, w, rng):
    """Concrete / asphalt-like grainy floor."""
    g = rng.uniform(60, 190)
    tint = np.array([rng.uniform(-15, 15) for _ in range(3)], np.float32)
    img = np.full((h, w, 3), g, np.float32) + tint
    for s, a in ((1, 18), (3, 14), (9, 10), (25, 8)):
        n = np.random.default_rng(rng.randrange(1 << 30)).normal(0, a, (max(2, h // s), max(2, w // s), 1))
        img += cv2.resize(n.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)[..., None]
    return img


def _plain(h, w, rng):
    c = np.array([rng.uniform(20, 235) for _ in range(3)], np.float32)
    img = np.full((h, w, 3), c, np.float32)
    if rng.random() < 0.6:  # gradient / vignette
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        g = (xx / w * rng.uniform(-1, 1) + yy / h * rng.uniform(-1, 1))
        img *= (1 + 0.35 * g)[..., None]
    return img


def _blue_edge(h, w, rng):
    """Light-blue painted strip next to a darker surface (like the camera's right side)."""
    img = _ridges(h, w, rng) if rng.random() < 0.6 else _noise_floor(h, w, rng)
    x = int(rng.uniform(0.3, 0.8) * w)
    blue = np.array([rng.uniform(200, 240), rng.uniform(200, 235), rng.uniform(120, 170)], np.float32)
    img[:, x:] = blue * rng.uniform(0.85, 1.1)
    return img


class Backgrounds:
    """bg = Backgrounds(floor_dir); bg.sample(h, w) -> uint8 BGR image."""

    def __init__(self, floor_images=(), seed=0):
        self.rng = random.Random(seed)
        self.floors = list(floor_images)  # list of BGR uint8 images (nut-free floor)

    @staticmethod
    def floors_from_dataset(image_dir, mask_dir, limit=120, seed=0):
        """Nut-free floor patches cut from dataset photos (area far from the nut)."""
        rng = random.Random(seed)
        out = []
        paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        rng.shuffle(paths)
        for mp in paths[: limit * 2]:
            name = os.path.splitext(os.path.basename(mp))[0]
            img = cv2.imread(os.path.join(image_dir, name + ".jpg"))
            m = cv2.imread(mp, 0)
            if img is None or m is None:
                continue
            hole = cv2.dilate((m > 127).astype(np.uint8), np.ones((41, 41), np.uint8))
            H, W = m.shape
            for _ in range(6):
                s = rng.randint(min(H, W) // 4, min(H, W) // 2)
                x, y = rng.randint(0, W - s), rng.randint(0, H - s)
                if not hole[y:y + s, x:x + s].any():
                    out.append(img[y:y + s, x:x + s].copy())
                    break
            if len(out) >= limit:
                break
        return out

    def sample(self, h, w):
        r = self.rng
        kinds = [_ridges, _noise_floor, _plain, _blue_edge]
        weights = [0.32, 0.2, 0.13, 0.1]
        if self.floors:
            kinds.append("floor")
            weights.append(0.25)
        k = r.choices(kinds, weights)[0]
        if k == "floor":
            img = cv2.resize(r.choice(self.floors), (w, h)).astype(np.float32)
            if r.random() < 0.5:
                img = img[:, ::-1] if r.random() < 0.5 else img[::-1]
        else:
            img = k(h, w, r)
        if r.random() < 0.25:  # blend two backgrounds
            other = self.sample(h, w).astype(np.float32)
            a = r.uniform(0.2, 0.5)
            img = img * (1 - a) + other * a
        img *= r.uniform(0.7, 1.3)  # global lighting
        if r.random() < 0.4:
            img = cv2.GaussianBlur(img, (0, 0), r.uniform(0.5, 2.5))
        return np.clip(img, 0, 255).astype(np.uint8)
