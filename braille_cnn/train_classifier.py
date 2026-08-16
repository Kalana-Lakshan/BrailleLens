"""Stage 4b — train SimpleBrailleCNN on the Stage 2c crop archives.

Reads data_pipeline/crops/crops_{train,val}.npz so DSBI and Angelina are
already mixed. Class weights and an optional domain-balanced sampler stop
the frequent letters / scanner domain from dominating.

This PC is CPU-only. A real run belongs on Colab; --smoke-test checks the
data path locally.

    py -3.11 -m braille_cnn.train_classifier --smoke-test
    py -3.11 -m braille_cnn.train_classifier --epochs 20 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data_pipeline.crop_dataset import CropDataset
from data_pipeline.contracts import repo_root

from .cnn import SimpleBrailleCNN

NUM_CLASSES = 64
ROOT = repo_root()
DEFAULT_CROPS = ROOT / "data_pipeline" / "crops"
DEFAULT_OUT = ROOT / "braille_cnn" / "checkpoints" / "braille_cnn_mixed.pt"


def _evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4b — train the 64-class cell CNN")
    parser.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS)
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--out-checkpoint", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--balance-domains", action="store_true",
                        help="Draw DSBI and Angelina at equal rate in each epoch")
    parser.add_argument("--sources", nargs="+", default=None,
                        help="Restrict to these sources (default: all in the npz)")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Subset the train split (for CPU smoke tests)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="1 epoch, 256 samples — checks the crop archives")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 1
        args.batch_size = 8
        args.max_train_samples = args.max_train_samples or 256
        if args.out_checkpoint == DEFAULT_OUT:
            args.out_checkpoint = ROOT / "braille_cnn" / "checkpoints" / "braille_cnn_smoke.pt"

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_npz = args.crops_dir / "crops_train.npz"
    val_npz = args.crops_dir / "crops_val.npz"
    if not val_npz.exists():
        val_npz = args.crops_dir / "crops_test.npz"

    train_ds = CropDataset(
        train_npz, augment=not args.no_augment, sources=args.sources
    )
    val_ds = CropDataset(val_npz, augment=False, sources=args.sources)
    if args.max_train_samples and args.max_train_samples < len(train_ds):
        n = args.max_train_samples
        idx = np.random.default_rng(0).choice(len(train_ds), size=n, replace=False)
        train_ds.crops = train_ds.crops[idx]
        train_ds.codes = train_ds.codes[idx]
        train_ds.sources = train_ds.sources[idx]
        train_ds.page_groups = train_ds.page_groups[idx]
    print(f"device: {device}")
    print(f"train: {train_ds.describe()}")
    print(f"val  : {val_ds.describe()}")

    sampler = train_ds.domain_balanced_sampler() if args.balance_domains else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
    )
    val_for_eval = val_ds
    if args.smoke_test and len(val_ds) > 512:
        from torch.utils.data import Subset

        val_for_eval = Subset(val_ds, list(range(512)))
    val_loader = DataLoader(val_for_eval, batch_size=256, shuffle=False, num_workers=0)

    model = SimpleBrailleCNN(num_classes=NUM_CLASSES).to(device)
    if args.init_checkpoint:
        model.load_state_dict(
            torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        )
        print(f"loaded {args.init_checkpoint}")

    weights = None if args.no_class_weights else train_ds.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    args.out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best = _evaluate(model, val_loader, device)
    print(f"start val acc: {best:.4f}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * labels.size(0)
            n += labels.size(0)
        acc = _evaluate(model, val_loader, device)
        print(f"epoch {epoch:02d}  loss={running / max(n, 1):.4f}  val_acc={acc:.4f}")
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), args.out_checkpoint)
            print(f"  saved {args.out_checkpoint}")

    print(f"best val acc: {best:.4f}")
    print("Next: py -3.11 -m braille_cnn.eval_angelina")


if __name__ == "__main__":
    main()
