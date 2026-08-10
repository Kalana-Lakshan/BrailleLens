"""Trains DotPatchCNN (dot_classifier.py) on DBSI ground truth, as a learned
replacement for dot_detect.py's brightness-threshold verification step.

Train/val pages are hardcoded and deliberately spread across multiple DBSI
books (not just multiple pages of one book) so the classifier doesn't
overfit to one book's paper/print style; val pages are different pages than
train, for an honest generalization check.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from .dot_classifier import DotPatchCNN
from .dot_patch_dataset import build_dataset

TRAIN_PAGES = [
    "data DBSI/Fundamentals of Massage/FM+11+recto",
    "data DBSI/Fundamentals of Massage/FM+12+recto",
    "data DBSI/Fundamentals of Massage/FM+15+recto",
    "data DBSI/Massage/M+10+recto",
    "data DBSI/Massage/M+11+recto",
    "data DBSI/Math/math+10+recto",
    "data DBSI/Math/math+11+recto",
    "data DBSI/Shaver Yang Fengting/SYF+3+recto",
    "data DBSI/The Second Volume of Ninth Grade Chinese Book 1/SVNGCB1+10+recto",
]
VAL_PAGES = [
    "data DBSI/Fundamentals of Massage/FM+16+recto",
    "data DBSI/Massage/M+12+recto",
    "data DBSI/Math/math+12+recto",
]


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    tp = fp = fn = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return correct / total, precision, recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--out-dir", type=str, default="braille_cnn/checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("building train set...")
    train_patches, train_labels = build_dataset(TRAIN_PAGES, ".", patch_size=args.patch_size)
    print("building val set...")
    val_patches, val_labels = build_dataset(VAL_PAGES, ".", patch_size=args.patch_size)
    print(f"train: {len(train_labels)}  val: {len(val_labels)}")

    def to_dataset(patches, labels):
        x = torch.from_numpy(patches).float().unsqueeze(1) / 255.0
        y = torch.from_numpy(labels)
        return TensorDataset(x, y)

    train_loader = DataLoader(to_dataset(train_patches, train_labels), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(to_dataset(val_patches, val_labels), batch_size=256, shuffle=False)

    model = DotPatchCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = running_correct = running_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (logits.argmax(1) == labels).sum().item()
            running_total += images.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        val_acc, val_prec, val_rec = evaluate(model, val_loader, device)
        val_f1 = 2 * val_prec * val_rec / max(val_prec + val_rec, 1e-9)
        print(f"epoch {epoch:3d}  train_loss {train_loss:.4f}  train_acc {train_acc:.4f}  "
              f"val_acc {val_acc:.4f}  val_precision {val_prec:.4f}  val_recall {val_rec:.4f}  val_f1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), out_dir / "dot_classifier_best.pt")
            print(f"  saved new best checkpoint (val_f1={val_f1:.4f})")

    print(f"\nbest val F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
