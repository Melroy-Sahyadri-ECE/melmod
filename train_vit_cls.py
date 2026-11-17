"""
Fine-tune a small pretrained Vision Transformer (ViT) to classify arecanut crops
as gottu / kole / pure.

    python make_cls_crops.py --data ..\\areca_yolo --out ..\\areca_cls
    python train_vit_cls.py --data ..\\areca_cls --device 0

Saves weights/vit_areca.pt  (state dict + class names + image size + model name)
and prints validation/test accuracy and a confusion matrix.
"""
import argparse
import os
import time

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # what timm's augreg ViTs were trained with


def loaders(root, size, batch, workers):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(0.35, 0.35, 0.3, 0.03),
        transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.0))], p=0.3),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    sets = {s: datasets.ImageFolder(os.path.join(root, s), train_tf if s == "train" else eval_tf)
            for s in ("train", "val", "test", "test_swap", "test_old") if os.path.isdir(os.path.join(root, s))}
    dl = {s: DataLoader(d, batch_size=batch, shuffle=(s == "train"), num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
          for s, d in sets.items()}
    return sets, dl


@torch.no_grad()
def evaluate(model, dl, device, n_cls):
    model.eval()
    cm = torch.zeros(n_cls, n_cls, dtype=torch.long)
    for x, y in dl:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            p = model(x.to(device, non_blocking=True)).argmax(1).cpu()
        for t, q in zip(y, p):
            cm[t, q] += 1
    return cm.diag().sum().item() / max(1, cm.sum().item()), cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../areca_cls")
    ap.add_argument("--model", default="vit_tiny_patch16_224.augreg_in21k_ft_in1k",
                    help="any timm ViT, e.g. vit_small_patch16_224.augreg_in21k_ft_in1k for more accuracy")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)  # keep low on Windows (paging-file limit)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="weights/vit_areca.pt")
    ap.add_argument("--init", default="", help="start from an earlier vit_areca*.pt (fine-tune) instead of ImageNet")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    sets, dl = loaders(args.data, 224, args.batch, args.workers)
    classes = sets["train"].classes
    print(f"classes: {classes} | train {len(sets['train'])}  val {len(sets['val'])}  test {len(sets['test'])}")

    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        if list(ck["classes"]) != list(classes):
            raise SystemExit(f"--init classes {ck['classes']} != data classes {classes}")
        args.model = ck["model"]
        model = timm.create_model(args.model, pretrained=False, num_classes=len(classes))
        model.load_state_dict(ck["state_dict"])
        model = model.to(device)
        print(f"fine-tuning from {args.init}")
    else:
        model = timm.create_model(args.model, pretrained=True, num_classes=len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl["train"]),
                                                pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, correct, n = time.time(), 0.0, 0, 0
        for x, y in dl["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(x)
                loss = loss_fn(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
        val_acc, _ = evaluate(model, dl["val"], device, len(classes))
        flag = ""
        if val_acc > best:
            best = val_acc
            torch.save({"model": args.model, "state_dict": model.state_dict(), "classes": classes,
                        "img_size": 224, "mean": MEAN, "std": STD}, args.out)
            flag = "  <- saved best"
        print(f"epoch {ep:2d}/{args.epochs}  loss {tot / n:.3f}  train acc {correct / n:.3f}  "
              f"val acc {val_acc:.3f}  ({time.time() - t0:.0f}s){flag}", flush=True)

    ckpt = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_acc, cm = evaluate(model, dl["test"], device, len(classes))
    print(f"\nTEST accuracy: {test_acc:.3f}  (best val {best:.3f})")
    if "test_old" in dl:
        old, _ = evaluate(model, dl["test_old"], device, len(classes))
        print(f"TEST on the OLD dataset (test_old): {old:.3f}")
    if "test_swap" in dl:
        sw, _ = evaluate(model, dl["test_swap"], device, len(classes))
        print(f"TEST on NEW backgrounds (test_swap): {sw:.3f}   <- close to TEST = it looks at the nut, not the floor")
    print("confusion matrix (rows = true, cols = predicted):")
    print("          " + "".join(f"{c:>8}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:>8}  " + "".join(f"{v:>8}" for v in cm[i].tolist()))
    print(f"\nweights: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
