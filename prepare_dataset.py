"""
Convert the areca_dataset repo (class folders + circle masks) into a YOLO detection dataset.

Source layout (github.com/Melroy-Sahyadri-ECE/areca_dataset):
    gottu/IMG_xxx.jpg   gottu_mask/IMG_xxx.png
    kole/IMG_xxx.jpg    kole_mask/IMG_xxx.png
    pure/IMG_xxx.jpg    pure_mask/IMG_xxx.png
Each mask is white circles on black; every separate white blob = one arecanut.

Output layout:
    <out>/images/{train,val,test}/*.jpg
    <out>/labels/{train,val,test}/*.txt   (class cx cy w h, normalised)
    <out>/masks/{train,val,test}/*.png    (for make_composites.py / make_test_video.py)
    <out>/data.yaml

Usage:
    python prepare_dataset.py --src areca_dataset --out areca_yolo
"""
import argparse
import glob
import os
import random
import shutil

import cv2
import numpy as np

# All three folders are just arecanuts -> ONE class. Folders are only used for a
# balanced (stratified) train/val/test split.
SOURCE_FOLDERS = ["gottu", "kole", "pure"]
CLASSES = ["arecanut"]


def mask_to_boxes(mask, min_area_frac=0.0005):
    """Return list of (x1, y1, x2, y2) — one per connected white blob."""
    binary = (mask > 127).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h, w = mask.shape[:2]
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area_frac * h * w:  # ignore stray specks
            continue
        boxes.append((x, y, x + bw, y + bh))
    return boxes


def to_yolo(box, w, h):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="areca_dataset", help="path to cloned areca_dataset repo")
    ap.add_argument("--out", default="areca_yolo")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--refine", action="store_true", help="tight boxes from the nut's real outline (GrabCut) instead of the drawn circle")
    args = ap.parse_args()

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    for split in ("train", "val", "test"):
        os.makedirs(f"{args.out}/images/{split}")
        os.makedirs(f"{args.out}/labels/{split}")
        os.makedirs(f"{args.out}/masks/{split}")

    rng = random.Random(args.seed)
    summary = {}
    skipped = []
    cls_id = 0  # everything is "arecanut"
    for cls in SOURCE_FOLDERS:
        items = []
        for img_path in sorted(glob.glob(os.path.join(args.src, cls, "*.jpg"))):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(args.src, f"{cls}_mask", f"{stem}.png")
            if not os.path.exists(mask_path):
                skipped.append(f"{cls}/{stem}.jpg (no mask)")
                continue
            items.append((img_path, mask_path, stem))

        # stratified split: each class is split with the same ratios
        rng.shuffle(items)
        n = len(items)
        n_test = round(n * args.test)
        n_val = round(n * args.val)
        splits = {"test": items[:n_test], "val": items[n_test:n_test + n_val], "train": items[n_test + n_val:]}

        for split, split_items in splits.items():
            for img_path, mask_path, stem in split_items:
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if img is None or mask is None:
                    skipped.append(f"{cls}/{stem} (unreadable)")
                    continue
                if mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                if args.refine:
                    from make_composites import refined_mask
                    boxes = mask_to_boxes(refined_mask(img, mask) * 255)
                else:
                    boxes = mask_to_boxes(mask)
                if not boxes:
                    skipped.append(f"{cls}/{stem} (empty mask)")
                    continue
                h, w = img.shape[:2]
                name = f"{cls}_{stem}"  # prefix: file names repeat across class folders
                shutil.copy(img_path, f"{args.out}/images/{split}/{name}.jpg")
                with open(f"{args.out}/labels/{split}/{name}.txt", "w") as f:
                    for b in boxes:
                        f.write("%d %.6f %.6f %.6f %.6f\n" % ((cls_id,) + to_yolo(b, w, h)))
                cv2.imwrite(f"{args.out}/masks/{split}/{name}.png", mask)
                summary[(split, cls)] = summary.get((split, cls), 0) + 1

    with open(f"{args.out}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write("names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))

    print("all images labelled as single class 'arecanut'; counts by source folder:")
    print(f"{'split':<6}" + "".join(f"{c:>8}" for c in SOURCE_FOLDERS) + f"{'total':>8}")
    for split in ("train", "val", "test"):
        counts = [summary.get((split, c), 0) for c in SOURCE_FOLDERS]
        print(f"{split:<6}" + "".join(f"{x:>8}" for x in counts) + f"{sum(counts):>8}")
    if skipped:
        print(f"skipped {len(skipped)}: " + ", ".join(skipped))
    print(f"wrote {args.out}/data.yaml")


if __name__ == "__main__":
    main()
