"""Fine-tunes the character CNN on real handheld-phone-photo Braille cells
(Angelina dataset), mixed with synthetic + DBSI data in every training batch.

Mixing matters: finetune_dbsi.py trained on DBSI-only and catastrophically
forgot the synthetic domain as a result (see RESULTS.md / README's Known
Issues) -- 98.44% on DBSI, but collapsed to ~14% on synthetic images
afterward. This script mixes all three domains (synthetic + DBSI + Angelina)
into one combined training set from the start, and evaluates on all three
domains' own held-out splits every epoch, specifically to catch that failure
mode immediately if it starts happening again rather than discovering it
after the fact.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, Subset

from .angelina_dataset import AngelinaDataset
from .cnn import SimpleBrailleCNN
from .dataset import SyntheticBrailleDataset
from .dbsi_dataset import DBSIDataset
from .labels import code_to_label

NUM_CLASSES = 64


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbsi-root", type=str, default="data DBSI")
    parser.add_argument("--angelina-root", type=str, default="data Angelina/books")
    parser.add_argument("--init-checkpoint", type=str,
                         default="braille_cnn/checkpoints/braille_cnn_dbsi_finetuned.pt")
    parser.add_argument("--out-checkpoint", type=str,
                         default="braille_cnn/checkpoints/braille_cnn_angelina_finetuned.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--synthetic-samples-per-class", type=int, default=100,
                         help="synthetic samples per class to mix into each training epoch")
    parser.add_argument("--out-dir", type=str, default="braille_cnn/checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("loading Angelina train/val...")
    angelina_train = AngelinaDataset(args.angelina_root, split="train", img_size=args.img_size)
    angelina_val = AngelinaDataset(args.angelina_root, split="val", img_size=args.img_size)
    print(f"  Angelina train: {len(angelina_train)} cells, val: {len(angelina_val)} cells")

    print("loading DBSI train/test...")
    dbsi_train = DBSIDataset(args.dbsi_root, split="train", img_size=args.img_size)
    dbsi_test = DBSIDataset(args.dbsi_root, split="test", img_size=args.img_size)
    print(f"  DBSI train: {len(dbsi_train)} cells, test: {len(dbsi_test)} cells")

    synthetic_train = SyntheticBrailleDataset(
        samples_per_class=args.synthetic_samples_per_class, img_size=args.img_size, train=True
    )
    synthetic_test = SyntheticBrailleDataset(
        samples_per_class=30, img_size=args.img_size, train=False
    )
    print(f"  synthetic train: {len(synthetic_train)} cells (fresh each epoch), test: {len(synthetic_test)} cells")

    train_ds = ConcatDataset([synthetic_train, dbsi_train, angelina_train])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    rng = np.random.default_rng(0)
    dbsi_eval_idx = rng.choice(len(dbsi_test), size=min(6000, len(dbsi_test)), replace=False)
    dbsi_eval_loader = DataLoader(Subset(dbsi_test, dbsi_eval_idx.tolist()), batch_size=256, shuffle=False)
    angelina_eval_loader = DataLoader(angelina_val, batch_size=256, shuffle=False)
    synthetic_eval_loader = DataLoader(synthetic_test, batch_size=256, shuffle=False)

    model = SimpleBrailleCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.init_checkpoint, map_location=device, weights_only=True))
    print(f"loaded init checkpoint: {args.init_checkpoint}")

    start_dbsi = evaluate(model, dbsi_eval_loader, device)
    start_angelina = evaluate(model, angelina_eval_loader, device)
    start_synthetic = evaluate(model, synthetic_eval_loader, device)
    print(f"starting accuracy -- DBSI: {start_dbsi:.4f}  Angelina: {start_angelina:.4f}  synthetic: {start_synthetic:.4f}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Track the best checkpoint by Angelina accuracy specifically (that's the
    # domain we're trying to improve), but print all three every epoch so a
    # forgetting regression on DBSI/synthetic is immediately visible.
    best_angelina = start_angelina
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
        dbsi_acc = evaluate(model, dbsi_eval_loader, device)
        angelina_acc = evaluate(model, angelina_eval_loader, device)
        synthetic_acc = evaluate(model, synthetic_eval_loader, device)
        print(f"epoch {epoch:3d}  train_loss {train_loss:.4f}  train_acc {train_acc:.4f}  "
              f"DBSI {dbsi_acc:.4f}  Angelina {angelina_acc:.4f}  synthetic {synthetic_acc:.4f}")

        if angelina_acc > best_angelina:
            best_angelina = angelina_acc
            torch.save(model.state_dict(), args.out_checkpoint)
            print(f"  saved new best checkpoint ({args.out_checkpoint})")

    if best_angelina == start_angelina:
        print("\ntraining never beat the starting Angelina baseline; reloading init checkpoint")
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device, weights_only=True))
    else:
        model.load_state_dict(torch.load(args.out_checkpoint, map_location=device, weights_only=True))

    print("\nfinal full evaluation with best checkpoint...")
    full_dbsi = evaluate(model, DataLoader(dbsi_test, batch_size=256), device)
    full_angelina = evaluate(model, DataLoader(angelina_val, batch_size=256), device)
    full_synthetic = evaluate(model, DataLoader(synthetic_test, batch_size=256), device)
    print(f"final -- DBSI test: {full_dbsi:.4f}  Angelina val: {full_angelina:.4f}  synthetic test: {full_synthetic:.4f}")


if __name__ == "__main__":
    main()
