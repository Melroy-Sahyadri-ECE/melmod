"""
Train a YOLO11 arecanut detector (single class: "arecanut").

Run on a GPU (Google Colab T4 is enough):
    pip install ultralytics
    python train.py --data areca_yolo/data.yaml

Outputs runs/areca/<name>/weights/best.pt (+ best.onnx).
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="areca_yolo/data.yaml")
    ap.add_argument("--model", default="yolo11s.pt",
                    help="yolo11n.pt = fastest (Raspberry Pi / Jetson), yolo11s.pt = best balance, yolo11m.pt = most accurate")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="0 for GPU, cpu for CPU")
    ap.add_argument("--name", default="arecanut_yolo11")
    ap.add_argument("--workers", type=int, default=2, help="keep low on Windows (paging-file limit)")
    args = ap.parse_args()

    model = YOLO(args.model)  # COCO-pretrained weights -> transfer learning
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/areca",
        name=args.name,
        exist_ok=True,
        single_cls=True,     # every nut is "arecanut"
        workers=args.workers,
        patience=20,         # early stop if val mAP stops improving
        cos_lr=True,
        # augmentation — nuts are seen top-down, so vertical flips are valid too
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,          # 4 images per sample -> many nuts per training frame
        close_mosaic=10,     # last 10 epochs on plain images for precise boxes
        mixup=0.1,
        scale=0.5,           # nuts closer to / further from the camera
        translate=0.1,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,  # lighting changes
        max_det=300,         # allow lots of nuts per frame
        seed=0,
        plots=True,
    )

    # newer Ultralytics versions nest runs under runs/detect/, so ask the trainer
    best_path = str(model.trainer.best)
    print(f"best weights: {best_path}")
    best = YOLO(best_path)
    print("\n=== held-out TEST split ===")
    m = best.val(data=args.data, split="test", imgsz=args.imgsz, plots=True,
                 project="runs/areca", name=f"{args.name}_test", exist_ok=True)
    print(f"mAP50 = {m.box.map50:.3f}   mAP50-95 = {m.box.map:.3f}   "
          f"precision = {m.box.mp:.3f}   recall = {m.box.mr:.3f}")

    # ONNX for deployment (OpenCV DNN / onnxruntime / TensorRT on Jetson)
    try:
        best.export(format="onnx", imgsz=args.imgsz, simplify=True)
    except Exception as e:  # export is optional; never fail the run over it
        print(f"(ONNX export skipped: {e})")
    print(f"\nweights: {best_path}")


if __name__ == "__main__":
    main()
