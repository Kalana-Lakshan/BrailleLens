"""Stage 4c — short fine-tune on Gold train pages.

Gold is the only Sinhala, in-domain data, so this runs LAST and SMALL.
Keeps DBSI + Angelina in the batches at ~3:1 against Gold so a few thousand
crops cannot overwrite the mixed model.

Does nothing useful until Gold LabelMe JSONs exist and have been reduced.

    py -3.11 -m braille_cnn.finetune_gold
    py -3.11 -m braille_cnn.finetune_gold --init-checkpoint braille_cnn/checkpoints/braille_cnn_mixed.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from data_pipeline.contracts import read_manifest, repo_root
from data_pipeline.crop_dataset import CropDataset

from .cnn import SimpleBrailleCNN
from .train_classifier import _evaluate

NUM_CLASSES = 64
ROOT = repo_root()
DEFAULT_CROPS = ROOT / "data_pipeline" / "crops"
DEFAULT_INIT = ROOT / "braille_cnn" / "checkpoints" / "braille_cnn_mixed.pt"
DEFAULT_OUT = ROOT / "braille_cnn" / "checkpoints" / "braille_cnn_gold.pt"


def _has_gold() -> bool:
    manifest = ROOT / "data_pipeline" / "manifests" / "manifest_clean.csv"
    if not manifest.exists():
        return False
    frame = read_manifest(manifest)
    return bool((frame["source"] == "gold").any())


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4c — Gold fine-tune")
    parser.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--out-checkpoint", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--gold-ratio", type=float, default=0.25,
                        help="Fraction of each batch that should be Gold (rest is mixed)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if not _has_gold():
        print(
            "No Gold cells in the clean manifest — expected until Stage 1b.\n"
            "Label High quality pages, then:\n"
            "  py -3.11 -m data_pipeline.transfer_gold_labels\n"
            "  py -3.11 -m data_pipeline.integrate --sources dbsi angelina gold "
            "--split-mode rebalance\n"
            "  py -3.11 -m data_pipeline.clean\n"
            "  py -3.11 -m data_pipeline.reduce\n"
            "  py -3.11 -m braille_cnn.finetune_gold"
        )
        return

    train_npz = args.crops_dir / "crops_train.npz"
    val_npz = args.crops_dir / "crops_val.npz"
    gold_train = CropDataset(train_npz, augment=True, sources=["gold"])
    rest_train = CropDataset(train_npz, augment=True, sources=["dbsi", "angelina"])
    gold_val = CropDataset(val_npz, augment=False, sources=["gold"]) if val_npz.exists() else None
    if len(gold_train) == 0:
        raise SystemExit("Gold is in the manifest but crops_train.npz has no Gold rows. Re-run reduce.")

    print(f"Gold train: {gold_train.describe()}")
    print(f"Other train: {rest_train.describe()}")

    mixed = ConcatDataset([gold_train, rest_train])
    n_gold, n_rest = len(gold_train), len(rest_train)
    g = args.gold_ratio
    weights = [g / max(n_gold, 1)] * n_gold + [(1.0 - g) / max(n_rest, 1)] * n_rest
    sampler = WeightedRandomSampler(weights, num_samples=len(mixed), replacement=True)
    train_loader = DataLoader(mixed, batch_size=args.batch_size, sampler=sampler)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SimpleBrailleCNN(num_classes=NUM_CLASSES).to(device)
    if not args.init_checkpoint.exists():
        raise SystemExit(
            f"Need a mixed-domain checkpoint first: {args.init_checkpoint}\n"
            "Run: py -3.11 -m braille_cnn.train_classifier"
        )
    model.load_state_dict(torch.load(args.init_checkpoint, map_location=device, weights_only=True))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=gold_train.class_weights().to(device))

    args.out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best = 0.0
    val_loader = DataLoader(gold_val, batch_size=256, shuffle=False) if gold_val and len(gold_val) else None
    if val_loader:
        best = _evaluate(model, val_loader, device)
        print(f"Gold val before fine-tune: {best:.4f}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = n = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += float(loss) * labels.size(0)
            n += labels.size(0)
        acc = _evaluate(model, val_loader, device) if val_loader else 0.0
        print(f"epoch {epoch:02d}  loss={running / max(n, 1):.4f}  gold_val={acc:.4f}")
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), args.out_checkpoint)
            print(f"  saved {args.out_checkpoint}")

    print(f"best Gold val: {best:.4f}")
    print("Evaluate held-out Gold test: py -3.11 -m braille_cnn.eval_gold --split test")


if __name__ == "__main__":
    main()
