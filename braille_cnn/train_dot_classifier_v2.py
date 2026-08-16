"""Retrains DotPatchCNN mixing DBSI + Angelina dot-patch data, so the
detector's verification stage isn't calibrated to DBSI's scanner alone (see
train_dot_classifier.py's original DBSI-only version, and chat history:
the resulting checkpoint gave 0% end-to-end accuracy on a real Angelina
photo despite the character CNN handling the same domain at 99.8% -- a good
classifier can't rescue a bad crop, and the crop was bad because nothing
detection-side had ever seen a real photo).

Train/eval pages are the same DBSI split as the original script, plus
Angelina's own predefined books/train.txt and books/val.txt.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from . import angelina_patch_dataset
from .dot_classifier import DotPatchCNN
from .dot_patch_dataset import build_dataset as build_dbsi_dataset

DBSI_TRAIN_PAGES = [
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
DBSI_VAL_PAGES = [
    "data DBSI/Fundamentals of Massage/FM+16+recto",
    "data DBSI/Massage/M+12+recto",
    "data DBSI/Math/math+12+recto",
]


def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = correct = total = 0
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
    return correct / max(total, 1), precision, recall


def to_dataset(patches, labels, patch_size):
    x = torch.from_numpy(patches).float().unsqueeze(1) / 255.0
    y = torch.from_numpy(labels)
    return TensorDataset(x, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angelina-root", type=str, default="data Angelina/books")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--out-dir", type=str, default="braille_cnn/checkpoints")
    parser.add_argument("--out-name", type=str, default="dot_classifier_mixed.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("building DBSI train/val...")
    dbsi_train_p, dbsi_train_l = build_dbsi_dataset(DBSI_TRAIN_PAGES, ".", patch_size=args.patch_size)
    dbsi_val_p, dbsi_val_l = build_dbsi_dataset(DBSI_VAL_PAGES, ".", patch_size=args.patch_size)
    print(f"  DBSI train: {len(dbsi_train_l)}  val: {len(dbsi_val_l)}")

    print("building Angelina train/val...")
    ang_train_p, ang_train_l = angelina_patch_dataset.build_dataset(
        args.angelina_root, "train", patch_size=args.patch_size
    )
    ang_val_p, ang_val_l = angelina_patch_dataset.build_dataset(
        args.angelina_root, "val", patch_size=args.patch_size
    )
    print(f"  Angelina train: {len(ang_train_l)}  val: {len(ang_val_l)}")

    train_patches = np.concatenate([dbsi_train_p, ang_train_p])
    train_labels = np.concatenate([dbsi_train_l, ang_train_l])
    print(f"combined train: {len(train_labels)}")

    train_loader = DataLoader(to_dataset(train_patches, train_labels, args.patch_size),
                               batch_size=args.batch_size, shuffle=True)
    dbsi_val_loader = DataLoader(to_dataset(dbsi_val_p, dbsi_val_l, args.patch_size), batch_size=256)
    ang_val_loader = DataLoader(to_dataset(ang_val_p, ang_val_l, args.patch_size), batch_size=256)

    model = DotPatchCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name

    best_combined_f1 = -1
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
        dbsi_acc, dbsi_p, dbsi_r = evaluate(model, dbsi_val_loader, device)
        ang_acc, ang_p, ang_r = evaluate(model, ang_val_loader, device)
        dbsi_f1 = 2 * dbsi_p * dbsi_r / max(dbsi_p + dbsi_r, 1e-9)
        ang_f1 = 2 * ang_p * ang_r / max(ang_p + ang_r, 1e-9)
        combined_f1 = (dbsi_f1 + ang_f1) / 2
        print(f"epoch {epoch:3d}  loss {train_loss:.4f}  train_acc {train_acc:.4f}  "
              f"DBSI(P/R/F1) {dbsi_p:.3f}/{dbsi_r:.3f}/{dbsi_f1:.3f}  "
              f"Angelina(P/R/F1) {ang_p:.3f}/{ang_r:.3f}/{ang_f1:.3f}")

        if combined_f1 > best_combined_f1:
            best_combined_f1 = combined_f1
            torch.save(model.state_dict(), out_path)
            print(f"  saved new best checkpoint ({out_path}), combined_f1={combined_f1:.4f}")

    print(f"\nbest combined (DBSI+Angelina averaged) F1: {best_combined_f1:.4f}")


if __name__ == "__main__":
    main()
