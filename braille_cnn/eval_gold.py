"""Stage 6 — cell-CNN accuracy on held-out Gold pages.

Gold is unlabelled right now. This script exits cleanly with instructions
until LabelMe JSONs exist and have been integrated.

    py -3.11 -m braille_cnn.eval_gold
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_pipeline.contracts import read_manifest, repo_root

from .cnn import SimpleBrailleCNN

NUM_CLASSES = 64
ROOT = repo_root()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the cell CNN on Gold")
    parser.add_argument("--checkpoint", default="braille_cnn/checkpoints/braille_cnn_mixed.pt")
    parser.add_argument("--manifest", default="data_pipeline/manifests/manifest_clean.csv")
    parser.add_argument("--crops-dir", default="data_pipeline/crops")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    if not manifest_path.exists():
        raise SystemExit(
            f"No manifest at {manifest_path}.\n"
            "Build DSBI+Angelina first: py -3.11 -m data_pipeline.integrate"
        )
    frame = read_manifest(manifest_path)
    gold = frame[frame["source"] == "gold"]
    if gold.empty:
        print(
            "Gold has 0 labelled cells — expected until Stage 1b.\n"
            "Label High quality pages with dot strings (see Gold Dataset/"
            "ANNOTATION_GUIDELINES.md), then:\n"
            "  py -3.11 -m data_pipeline.transfer_gold_labels\n"
            "  py -3.11 -m data_pipeline.integrate --sources dbsi angelina gold "
            "--split-mode rebalance\n"
            "  py -3.11 -m data_pipeline.clean\n"
            "  py -3.11 -m data_pipeline.reduce\n"
            "  py -3.11 -m braille_cnn.eval_gold"
        )
        return

    from data_pipeline.crop_dataset import CropDataset

    npz = Path(args.crops_dir) / f"crops_{args.split}.npz"
    if not npz.exists():
        raise SystemExit(f"Missing {npz}. Run: py -3.11 -m data_pipeline.reduce")

    ds = CropDataset(npz, augment=False, sources=["gold"])
    if len(ds) == 0:
        print(f"No Gold crops in {npz} split={args.split}.")
        return

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleBrailleCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Gold {args.split}: {correct / max(total, 1):.4f}  ({correct}/{total})")
    print(ds.describe())
    from .eval_report import write_eval_report

    write_eval_report(
        Path("reports/eval") / f"gold_{args.split}.md",
        f"Gold {args.split} cell-CNN accuracy",
        [
            f"Checkpoint: `{ckpt}`",
            f"Accuracy: **{correct / max(total, 1):.4f}** ({correct}/{total})",
            ds.describe(),
        ],
    )


if __name__ == "__main__":
    main()
