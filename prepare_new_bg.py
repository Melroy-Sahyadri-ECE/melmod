"""
Retraining data for a NEW background (github.com/Thilakraj116/final_data_set):
class folders of camera photos, one arecanut per photo, no labels.

1. Boxes every photo automatically with the current detector (one nut per photo -> best box).
2. Splits per class in blocks of consecutive frames (train / val / test), so near-identical
   frames never end up on both sides.
3. Detector data  (<out-yolo>):  new photos + multi-nut frames made on the new background
   (nuts from other photos pasted in) + part of the old dataset (keeps old backgrounds working).
   val / test = new background only.
4. ViT data (<out-vit>):  square crops per class (gottu / kole / pure) + the old ViT training set.
   val / test = new background only; test_old = old test set (regression check).
5. review_boxes.jpg: contact sheet of the automatic boxes to check by eye.

    python prepare_new_bg.py --src ..\\final_data_set\\captured_images --det weights\\detector_v2.pt
"""
import argparse
import glob
import os
import random
import shutil

import cv2
import numpy as np
from ultralytics import YOLO

FOLDER_TO_CLASS = {"batte": "pure", "bette": "pure", "pure": "pure",
                   "gotu": "gottu", "gottu": "gottu", "kole": "kole"}
CLASSES = ["gottu", "kole", "pure"]


def best_box(model, img, device):
    """Highest-confidence box; retries larger / lower threshold if nothing is found."""
    for imgsz, conf in ((640, 0.05), (960, 0.01)):
        r = model.predict(img, imgsz=imgsz, conf=conf, device=device, verbose=False, max_det=5)[0]
        if r.boxes is not None and len(r.boxes):
            i = int(r.boxes.conf.argmax())
            return [int(v) for v in r.boxes.xyxy[i].tolist()], float(r.boxes.conf[i])
    return None, 0.0


def refine_box(img, b, grow=1.0, max_area_ratio=3.0):
    """Expand a (possibly partial) nut box to the whole light-coloured nut.
    Nut = pixels that differ from the floor colour, are not dark (shadow) and not cyan (tape),
    connected to the detected box. Returns the union box (capped)."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    s = max(bw, bh)
    X1, Y1 = max(0, int(x1 - grow * s)), max(0, int(y1 - grow * s))
    X2, Y2 = min(W, int(x2 + grow * s)), min(H, int(y2 + grow * s))
    sub = img[Y1:Y2, X1:X2]
    lab = cv2.cvtColor(sub, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    inb = np.zeros(sub.shape[:2], bool)
    inb[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1] = True
    # floor colour = median of the border of the search window
    border = np.zeros_like(inb); border[:6] = border[-6:] = True; border[:, :6] = border[:, -6:] = True
    fl = np.median(lab[border], axis=0)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    cyan = (hsv[..., 0] > 75) & (hsv[..., 0] < 105) & (hsv[..., 1] > 80)
    # husk / skin is greener (lower a) and yellower (higher b) than the reddish floor; shadow is dark
    husk = (A < fl[1] - 5) & (B > fl[2] + 6) & (L > 60)
    white = (L > 205) & (A < fl[1] - 3) & (np.abs(A - 128) < 9) & (np.abs(B - 130) < 14)  # over-exposed skin
    fg = (husk | white) & ~cyan
    fg = fg.astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labm, st, _ = cv2.connectedComponentsWithStats(fg)
    best, best_ov = 0, 0
    near = cv2.dilate(inb.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0   # inside or touching the box
    for i in range(1, n):
        ov = (labm[near] == i).sum()
        if ov > best_ov:
            best, best_ov = i, ov
    if best == 0 or best_ov < 0.02 * inb.sum():
        return b, False
    x, y, w, h = st[best][:4]
    nb = [min(x1, X1 + x), min(y1, Y1 + y), max(x2, X1 + x + w), max(y2, Y1 + y + h)]
    if (nb[2] - nb[0]) * (nb[3] - nb[1]) > max_area_ratio * bw * bh:
        return b, False
    return nb, nb != list(b)


def to_yolo(b, w, h):
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h


def square_crop(img, b, pad=0.15):
    x1, y1, x2, y2 = b
    s = max(x2 - x1, y2 - y1) * (1 + 2 * pad)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    H, W = img.shape[:2]
    X1, Y1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    X2, Y2 = int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
    return img[Y1:Y2, X1:X2]


def paste_nut(dst, src, b, cx, cy, scale):
    """Paste the nut in box b of src centred at (cx, cy) with a soft elliptical edge. Returns its box."""
    x1, y1, x2, y2 = b
    m = int(0.12 * max(x2 - x1, y2 - y1))
    H, W = src.shape[:2]
    X1, Y1, X2, Y2 = max(0, x1 - m), max(0, y1 - m), min(W, x2 + m), min(H, y2 + m)
    patch = src[Y1:Y2, X1:X2]
    ph, pw = patch.shape[:2]
    alpha = np.zeros((ph, pw), np.float32)
    cv2.ellipse(alpha, (((x1 + x2) / 2 - X1, (y1 + y2) / 2 - Y1), ((x2 - x1) * 1.12, (y2 - y1) * 1.12), 0), 1.0, -1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(1.0, 0.06 * max(pw, ph)))
    k = random.choice([0, 1, 2, 3])                      # random 90 deg turn + flip
    patch, alpha = np.rot90(patch, k), np.rot90(alpha, k)
    if random.random() < 0.5:
        patch, alpha = patch[:, ::-1], alpha[:, ::-1]
    patch = cv2.resize(np.ascontiguousarray(patch), None, fx=scale, fy=scale)
    alpha = cv2.resize(np.ascontiguousarray(alpha), None, fx=scale, fy=scale)[..., None]
    ph, pw = patch.shape[:2]
    DH, DW = dst.shape[:2]
    ox, oy = int(cx - pw / 2), int(cy - ph / 2)
    if ox < 0 or oy < 0 or ox + pw > DW or oy + ph > DH:
        return None
    roi = dst[oy:oy + ph, ox:ox + pw].astype(np.float32)
    dst[oy:oy + ph, ox:ox + pw] = (alpha * patch + (1 - alpha) * roi).astype(np.uint8)
    # box of the nut inside the pasted patch
    bw, bh = (x2 - x1), (y2 - y1)
    if k % 2:
        bw, bh = bh, bw
    bw, bh = bw * scale, bh * scale
    return [int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)]


def overlaps(b, boxes, gap=6):
    return any(not (b[2] + gap < o[0] or o[2] + gap < b[0] or b[3] + gap < o[1] or o[3] + gap < b[1]) for o in boxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../final_data_set/captured_images")
    ap.add_argument("--det", default="weights/detector_v2.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--old-yolo", default="../areca_yolo")
    ap.add_argument("--old-vit", default="../areca_vit")
    ap.add_argument("--out-yolo", default="../areca_yolo_v3")
    ap.add_argument("--out-vit", default="../areca_vit_v3")
    ap.add_argument("--composites", type=int, default=800, help="multi-nut frames on the new background")
    ap.add_argument("--old-train", type=int, default=1200, help="old-background training images to keep")
    ap.add_argument("--old-vit-per-class", type=int, default=500, help="old ViT training crops kept per class")
    ap.add_argument("--block", type=int, default=10, help="consecutive frames kept together in one split")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)

    for d in (a.out_yolo, a.out_vit):
        if os.path.exists(d):
            shutil.rmtree(d)
    for s in ("train", "val", "test"):
        os.makedirs(f"{a.out_yolo}/images/{s}")
        os.makedirs(f"{a.out_yolo}/labels/{s}")
        for c in CLASSES:
            os.makedirs(f"{a.out_vit}/{s}/{c}")
    for c in CLASSES:
        os.makedirs(f"{a.out_vit}/test_old/{c}")

    model = YOLO(a.det)
    items, missed, review = [], [], []
    n_grown = 0
    for folder in sorted(os.listdir(a.src)):
        cls = FOLDER_TO_CLASS.get(folder.lower())
        if cls is None or not os.path.isdir(os.path.join(a.src, folder)):
            continue
        files = sorted(glob.glob(os.path.join(a.src, folder, "*.jpg")) + glob.glob(os.path.join(a.src, folder, "*.png")))
        print(f"{folder} -> {cls}: {len(files)} photos, boxing...", flush=True)
        for i, f in enumerate(files):
            img = cv2.imread(f)
            if img is None:
                continue
            b, conf = best_box(model, img, a.device)
            H, W = img.shape[:2]
            ok = b is not None and 0.0005 < (b[2] - b[0]) * (b[3] - b[1]) / (W * H) < 0.25
            if not ok:
                missed.append(f)
                continue
            if cls == "gottu":                 # the detector often boxes only the bright part
                b, _grown = refine_box(img, b)  # of a husked nut -> grow the box over the husk
                b = [int(v) for v in b]
                n_grown += int(_grown)
            blk = (i // a.block) % 10
            split = "val" if blk == 8 else "test" if blk == 9 else "train"
            items.append(dict(f=f, img=img, b=b, conf=conf, cls=cls, split=split,
                              name=f"new_{cls}_{os.path.splitext(os.path.basename(f))[0]}"))
            if i % max(1, len(files) // (24 if cls == "gottu" else 8)) == 0:
                review.append((img, b, cls, conf))

    # ---- detector + ViT data from the new photos ----
    counts = {}
    for it in items:
        img, b, s = it["img"], it["b"], it["split"]
        H, W = img.shape[:2]
        cv2.imwrite(f"{a.out_yolo}/images/{s}/{it['name']}.jpg", img)
        with open(f"{a.out_yolo}/labels/{s}/{it['name']}.txt", "w") as fh:
            fh.write("0 %.6f %.6f %.6f %.6f\n" % to_yolo(b, W, H))
        cv2.imwrite(f"{a.out_vit}/{s}/{it['cls']}/{it['name']}.jpg", square_crop(img, b))
        counts[(s, it["cls"])] = counts.get((s, it["cls"]), 0) + 1

    # ---- multi-nut frames on the new background (train only) ----
    train = [it for it in items if it["split"] == "train"]
    for k in range(a.composites if train else 0):
        base = random.choice(train)
        img = base["img"].copy()
        H, W = img.shape[:2]
        boxes = [base["b"]]
        for _ in range(random.randint(1, 6)):
            donor = random.choice(train)
            for _try in range(15):
                scale = random.uniform(0.8, 1.25)
                bw = (donor["b"][2] - donor["b"][0]) * scale
                bh = (donor["b"][3] - donor["b"][1]) * scale
                cx = random.uniform(bw, W - bw)
                cy = random.uniform(bh, H - bh)
                guess = [int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)]
                if overlaps(guess, boxes, gap=int(0.25 * max(bw, bh))):
                    continue
                nb = paste_nut(img, donor["img"], donor["b"], cx, cy, scale)
                if nb:
                    boxes.append(nb)
                break
        name = f"comp_{k:04d}"
        cv2.imwrite(f"{a.out_yolo}/images/train/{name}.jpg", img)
        with open(f"{a.out_yolo}/labels/train/{name}.txt", "w") as fh:
            for b in boxes:
                fh.write("0 %.6f %.6f %.6f %.6f\n" % to_yolo(b, W, H))
        if k < 3:
            review.append((img, None, f"{len(boxes)} nuts", 0))

    # ---- part of the old dataset (train only) so old backgrounds keep working ----
    old = sorted(glob.glob(os.path.join(a.old_yolo, "images", "train", "*.jpg")))
    random.shuffle(old)
    for p in old[:a.old_train]:
        stem = os.path.splitext(os.path.basename(p))[0]
        lab = os.path.join(a.old_yolo, "labels", "train", stem + ".txt")
        if os.path.exists(lab):
            shutil.copy(p, f"{a.out_yolo}/images/train/old_{stem}.jpg")
            shutil.copy(lab, f"{a.out_yolo}/labels/train/old_{stem}.txt")
    with open(f"{a.out_yolo}/data.yaml", "w") as fh:
        fh.write(f"path: {os.path.abspath(a.out_yolo)}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                 "names:\n  0: arecanut\n")

    # ---- old ViT data: training set + old test as a regression check ----
    for src_split, dst_split, limit in (("train", "train", a.old_vit_per_class), ("test", "test_old", None)):
        for c in CLASSES:
            fs = sorted(glob.glob(os.path.join(a.old_vit, src_split, c, "*")))
            random.shuffle(fs)
            for p in fs[:limit]:
                shutil.copy(p, f"{a.out_vit}/{dst_split}/{c}/old_{os.path.basename(p)}")

    # ---- review sheet ----
    tiles = []
    for img, b, cls, conf in review[:40]:
        t = img.copy()
        if b:
            cv2.rectangle(t, tuple(b[:2]), tuple(b[2:]), (0, 0, 255), 3)
        cv2.putText(t, f"{cls} {conf:.2f}" if b else str(cls), (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        tiles.append(cv2.resize(t, (320, 240)))
    while len(tiles) % 8:
        tiles.append(np.zeros((240, 320, 3), np.uint8))
    sheet = np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)])
    cv2.imwrite("review_boxes.jpg", sheet)
    if missed:
        mt = [cv2.resize(cv2.imread(f), (320, 240)) for f in missed[:24]]
        while len(mt) % 8:
            mt.append(np.zeros((240, 320, 3), np.uint8))
        cv2.imwrite("review_missed.jpg", np.vstack([np.hstack(mt[i:i + 8]) for i in range(0, len(mt), 8)]))

    print("\nnew photos per split / class:")
    for s in ("train", "val", "test"):
        print(f"  {s:<5} " + "  ".join(f"{c} {counts.get((s, c), 0)}" for c in CLASSES))
    confs = np.array([it["conf"] for it in items]) if items else np.zeros(1)
    print(f"boxed {len(items)} photos (median conf {np.median(confs):.2f}, {int((confs < 0.3).sum())} below 0.3), "
          f"no box: {len(missed)}")
    print(f"gotu boxes grown over the husk: {n_grown}")
    print(f"composites {a.composites}, old images {min(len(old), a.old_train)}")
    print(f"detector data: {os.path.abspath(a.out_yolo)}\\data.yaml   ViT data: {os.path.abspath(a.out_vit)}")
    print("PREP DONE")


if __name__ == "__main__":
    main()
