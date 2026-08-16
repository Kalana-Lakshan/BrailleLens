"""Stage 4a step 2 - transfer-learn a single-class Braille cell detector.

Starts from COCO-pretrained YOLO26 and fine-tunes on the dataset built by
prepare_cell_dataset.py.

This machine has CPU-only PyTorch, so a real run belongs on Colab or Kaggle -
see BrailleLens_CellDetector_Colab.ipynb. The script still runs locally with
--device cpu --epochs 1 as a smoke test, which is worth doing before uploading
anything: it catches dataset path and label format mistakes in a minute instead
of after a GPU queue.

Usage (from repo root):
    py -3.11 -m cell_detect.train_detector --smoke-test
    py -3.11 -m cell_detect.train_detector --epochs 80 --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs" / "cells.yaml"


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_data_yaml(raw, here: Path) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    for base in (here, Path.cwd()):
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return here / candidate


def main():
    parser = argparse.ArgumentParser(description="Train the Braille cell detector")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=None, help="Override data.yaml")
    parser.add_argument("--model", type=str, default=None, help="e.g. yolo26s.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu | 0 | 0,1")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="1 epoch, imgsz 640, batch 2, CPU - checks the dataset, not the model")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    if args.smoke_test:
        args.epochs = args.epochs or 1
        args.imgsz = args.imgsz or 640
        args.batch = args.batch or 2
        args.device = args.device or "cpu"
        args.name = args.name or "smoke_test"

    data_yaml = _resolve_data_yaml(args.data or cfg["data"], HERE)
    if not data_yaml.exists():
        raise SystemExit(
            f"Dataset config not found: {data_yaml}\n"
            "Build it first: py -3.11 -m cell_detect.prepare_cell_dataset"
        )

    model_name = args.model or cfg.get("model", "yolo26n.pt")
    epochs = args.epochs if args.epochs is not None else cfg.get("epochs", 80)
    batch = args.batch if args.batch is not None else cfg.get("batch", 4)
    imgsz = args.imgsz if args.imgsz is not None else cfg.get("imgsz", 1280)
    device = args.device if args.device is not None else cfg.get("device", 0)
    run_name = args.name or cfg.get("name", "braille_cell_yolo26")
    project = HERE / cfg.get("project", "runs/detect")

    print("=" * 66)
    print("Braille CELL detector - transfer learning (Stage 4a)")
    print(f"  base model : {model_name}")
    print(f"  data       : {data_yaml}")
    print(f"  epochs     : {epochs}  batch={batch}  imgsz={imgsz}  device={device}")
    print(f"  max_det    : {cfg.get('max_det', 800)}")
    print("=" * 66)

    import torch
    from ultralytics import YOLO

    if str(device) != "cpu" and not torch.cuda.is_available():
        raise SystemExit(
            f"--device {device} requested but torch reports no CUDA "
            f"(torch {torch.__version__}).\n"
            "Train on Colab/Kaggle, or pass --device cpu for a smoke test."
        )

    model = YOLO(model_name)
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
        optimizer=cfg.get("optimizer", "auto"),
        lr0=cfg.get("lr0", 0.01),
        lrf=cfg.get("lrf", 0.01),
        weight_decay=cfg.get("weight_decay", 0.0005),
        warmup_epochs=cfg.get("warmup_epochs", 3.0),
        patience=cfg.get("patience", 15),
        workers=cfg.get("workers", 2),
        seed=cfg.get("seed", 42),
        hsv_h=cfg.get("hsv_h", 0.015),
        hsv_s=cfg.get("hsv_s", 0.5),
        hsv_v=cfg.get("hsv_v", 0.4),
        degrees=cfg.get("degrees", 3.0),
        translate=cfg.get("translate", 0.1),
        scale=cfg.get("scale", 0.3),
        shear=cfg.get("shear", 1.0),
        perspective=cfg.get("perspective", 0.0005),
        flipud=cfg.get("flipud", 0.0),
        fliplr=cfg.get("fliplr", 0.0),
        mosaic=cfg.get("mosaic", 0.5),
        mixup=cfg.get("mixup", 0.0),
        copy_paste=cfg.get("copy_paste", 0.0),
        erasing=cfg.get("erasing", 0.2),
        close_mosaic=cfg.get("close_mosaic", 10),
        max_det=cfg.get("max_det", 800),
        save_period=cfg.get("save_period", 5),
        plots=True,
        save=True,
    )

    best = project / run_name / "weights" / "best.pt"
    print("\nTraining finished.")
    print(f"  best weights -> {best}")
    print("Copy that file to cell_detect/weights/braille_cell_best.pt, then:")
    print("  py -3.11 -m cell_detect.evaluate_detector")
    return results


if __name__ == "__main__":
    main()
