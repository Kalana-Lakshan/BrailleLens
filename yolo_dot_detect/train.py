"""Step 2 — Transfer-learn YOLOv8 on Braille embossed dots with augmentation.

Loads COCO-pretrained YOLOv8 weights (yolov8n.pt by default) and fine-tunes
on the YOLO-format dataset produced by prepare_dataset.py. Ultralytics applies
HSV / geometric / mosaic / mixup / random-erase augmentations during training
(see configs/default.yaml).

Usage (from repo root, after prepare_dataset):
    py -3.11 -m yolo_dot_detect.train
    py -3.11 -m yolo_dot_detect.train --epochs 30 --model yolov8s.pt --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train YOLOv8 Braille-dot detector")
    parser.add_argument(
        "--config",
        type=Path,
        default=here / "configs" / "default.yaml",
    )
    parser.add_argument("--data", type=Path, default=None, help="Override data.yaml")
    parser.add_argument("--model", type=str, default=None, help="e.g. yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu | 0 | 0,1")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    data_yaml = Path(args.data) if args.data else here / cfg["data"]
    if not data_yaml.is_absolute():
        # resolve relative to this package, then to CWD
        candidates = [here / data_yaml, Path.cwd() / data_yaml, data_yaml]
        data_yaml = next((p for p in candidates if p.exists()), candidates[0])
    if not data_yaml.exists():
        raise SystemExit(
            f"Dataset config not found: {data_yaml}\n"
            "Run first: py -3.11 -m yolo_dot_detect.prepare_dataset"
        )

    model_name = args.model or cfg.get("model", "yolov8n.pt")
    epochs = args.epochs if args.epochs is not None else cfg.get("epochs", 50)
    batch = args.batch if args.batch is not None else cfg.get("batch", 8)
    imgsz = args.imgsz if args.imgsz is not None else cfg.get("imgsz", 640)
    device = args.device if args.device is not None else cfg.get("device", "cpu")
    run_name = args.name or cfg.get("name", "braille_dot_yolov8")
    project = here / cfg.get("project", "runs/detect")

    print("=" * 60)
    print("YOLOv8 Braille-dot transfer learning")
    print(f"  base model : {model_name}  (COCO pretrained)")
    print(f"  data       : {data_yaml}")
    print(f"  epochs     : {epochs}  batch={batch}  imgsz={imgsz}  device={device}")
    print(f"  aug        : hsv/rot/scale/perspective/mosaic/mixup/erasing")
    print("=" * 60)

    from ultralytics import YOLO

    model = YOLO(model_name)  # downloads pretrained weights on first run

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=run_name,
        exist_ok=True,
        resume=args.resume,
        optimizer=cfg.get("optimizer", "AdamW"),
        lr0=cfg.get("lr0", 0.001),
        lrf=cfg.get("lrf", 0.01),
        weight_decay=cfg.get("weight_decay", 0.0005),
        warmup_epochs=cfg.get("warmup_epochs", 3.0),
        patience=cfg.get("patience", 15),
        workers=cfg.get("workers", 2),
        seed=cfg.get("seed", 42),
        # --- augmentation ---
        hsv_h=cfg.get("hsv_h", 0.015),
        hsv_s=cfg.get("hsv_s", 0.5),
        hsv_v=cfg.get("hsv_v", 0.4),
        degrees=cfg.get("degrees", 10.0),
        translate=cfg.get("translate", 0.1),
        scale=cfg.get("scale", 0.3),
        shear=cfg.get("shear", 2.0),
        perspective=cfg.get("perspective", 0.0005),
        flipud=cfg.get("flipud", 0.0),
        fliplr=cfg.get("fliplr", 0.0),
        mosaic=cfg.get("mosaic", 0.8),
        mixup=cfg.get("mixup", 0.1),
        copy_paste=cfg.get("copy_paste", 0.0),
        erasing=cfg.get("erasing", 0.2),
        close_mosaic=cfg.get("close_mosaic", 10),
        max_det=cfg.get("max_det", 3000),
        plots=True,
        save=True,
    )

    best = project / run_name / "weights" / "best.pt"
    print("\nTraining finished.")
    print(f"  best weights -> {best}")
    print("Next: py -3.11 -m yolo_dot_detect.infer --image test-img.jpeg")
    return results


if __name__ == "__main__":
    main()
