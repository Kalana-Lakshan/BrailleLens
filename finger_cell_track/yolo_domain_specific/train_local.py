"""Fine-tune fingertip YOLO26 on Braille domain dataset (local PC).

Mirrors BrailleLens_Fingertip_Domain_Colab.ipynb — works on CPU or CUDA.

Usage (from repo root)::

    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/yolo_domain_specific/train_local.py

    # Quick smoke test (~few minutes on CPU):
    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/yolo_domain_specific/train_local.py --epochs 5

    # Resume after interrupt:
    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/yolo_domain_specific/train_local.py --resume
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

_HERE = Path(__file__).resolve().parent
_FCT = _HERE.parent
_DEFAULT_DATA = _HERE / "datasets" / "braille_fingertip_yolo" / "data.yaml"
_DEFAULT_WEIGHTS = _FCT / "weights" / "yolo26n_fingertip_best.pt"
_RUNS = _HERE / "runs" / "fingertip_domain"
_RUN_NAME = "yolo26n_braille_finetune"
_EXPORT_WEIGHTS = _FCT / "weights" / "yolo26n_fingertip_braille_best.pt"
_METRICS_OUT = _HERE / "metrics_summary.json"

_DEFAULT_HP = {
    "lr0": 0.001,
    "weight_decay": 0.0005,
    "mosaic": 0.5,
    "mixup": 0.05,
    "fliplr": 0.5,
    "dropout": 0.0,
}


def _resolve_device(requested: str) -> str | int:
    if requested == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    return requested


def _eval_split(model: YOLO, data_yaml: Path, imgsz: int, device, split: str) -> dict:
    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        device=device,
        conf=0.25,
        split=split,
        plots=(split == "val"),
    )
    p = float(metrics.box.mp)
    r = float(metrics.box.mr)
    f1 = 2 * p * r / (p + r + 1e-9)
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Local Braille fingertip YOLO fine-tune")
    p.add_argument("--data", type=Path, default=_DEFAULT_DATA)
    p.add_argument("--weights", type=Path, default=_DEFAULT_WEIGHTS)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=None, help="Default: 8 GPU, 4 CPU")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--device", default="auto", help="auto | cpu | 0")
    p.add_argument("--resume", action="store_true", help="Resume from last.pt")
    p.add_argument("--eval-only", action="store_true", help="Skip train; eval best.pt")
    args = p.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Dataset not found: {args.data}\n"
            "Run: finger_cell_track/yolo_domain_specific/build_dataset.py"
        )

    device = _resolve_device(args.device)
    batch = args.batch if args.batch is not None else (8 if device != "cpu" else 4)

    runs_dir = _RUNS
    weights_dir = runs_dir / _RUN_NAME / "weights"
    last_pt = weights_dir / "last.pt"
    best_pt = weights_dir / "best.pt"

    print(f"device     : {device}")
    if device == "cpu":
        print("NOTE: CPU training is slow. Expect 1–3+ hours for 80 epochs on 48 images.")
        print("      Use --epochs 5 for a quick smoke test, or wait for Colab GPU for full run.")
    print(f"data       : {args.data}")
    print(f"batch      : {batch}")
    print(f"epochs     : {args.epochs}")

    # Fix data.yaml path to absolute (Ultralytics)
    data_root = args.data.parent
    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    cfg["path"] = str(data_root.resolve())
    args.data.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    hp = dict(_DEFAULT_HP)

    if args.eval_only:
        if not best_pt.exists():
            raise SystemExit(f"No checkpoint: {best_pt}")
        model = YOLO(str(best_pt))
    else:
        resume = args.resume or last_pt.exists()
        if resume and last_pt.exists():
            print(f"Resuming from {last_pt}")
            model = YOLO(str(last_pt))
        else:
            if not args.weights.exists():
                raise SystemExit(f"Base weights not found: {args.weights}")
            print(f"Fine-tuning from {args.weights}")
            model = YOLO(str(args.weights))

        model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=batch,
            device=device,
            project=str(runs_dir),
            name=_RUN_NAME,
            exist_ok=True,
            resume=resume and last_pt.exists(),
            save=True,
            save_period=1,
            patience=args.patience,
            workers=0 if sys.platform == "win32" else 4,
            seed=42,
            plots=True,
            lr0=hp["lr0"],
            weight_decay=hp["weight_decay"],
            dropout=hp.get("dropout", 0.0),
            hsv_h=0.015,
            hsv_s=0.50,
            hsv_v=0.40,
            degrees=5.0,
            translate=0.10,
            scale=0.30,
            shear=1.0,
            perspective=0.0005,
            flipud=0.0,
            fliplr=hp["fliplr"],
            mosaic=hp["mosaic"],
            mixup=hp["mixup"],
            close_mosaic=10,
        )

    if not best_pt.exists():
        raise SystemExit(f"Training finished but best.pt missing: {best_pt}")

    model = YOLO(str(best_pt))
    summary = {
        "base_weights": str(args.weights.name),
        "best_weights": str(best_pt),
        "device": str(device),
        "epochs": args.epochs,
        "val": _eval_split(model, args.data, args.imgsz, device, "val"),
        "test": _eval_split(model, args.data, args.imgsz, device, "test"),
        "hyperparameters": hp,
    }

    _METRICS_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== Metrics ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {_METRICS_OUT}")

    _EXPORT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, _EXPORT_WEIGHTS)
    print(f"Exported weights: {_EXPORT_WEIGHTS}")


if __name__ == "__main__":
    main()
