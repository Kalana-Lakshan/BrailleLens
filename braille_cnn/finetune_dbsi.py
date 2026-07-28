import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from .cnn import SimpleBrailleCNN
from .dbsi_dataset import DBSIDataset
from .labels import code_to_label

NUM_CLASSES = 64


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion[t, p] += 1
    return correct / total, confusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbsi-root", type=str, default="data DBSI")
    parser.add_argument("--init-checkpoint", type=str, default="braille_cnn/checkpoints/braille_cnn_best.pt")
    parser.add_argument("--out-checkpoint", type=str, default="braille_cnn/checkpoints/braille_cnn_dbsi_finetuned.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--eval-subset", type=int, default=8000)
    parser.add_argument("--out-dir", type=str, default="braille_cnn/checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = DBSIDataset(args.dbsi_root, split="train", img_size=args.img_size)
    test_ds = DBSIDataset(args.dbsi_root, split="test", img_size=args.img_size)
    print(f"DBSI train: {len(train_ds)} cells, DBSI test: {len(test_ds)} cells")

    rng = np.random.default_rng(0)
    subset_idx = rng.choice(len(test_ds), size=min(args.eval_subset, len(test_ds)), replace=False)
    quick_eval_ds = Subset(test_ds, subset_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    quick_eval_loader = DataLoader(quick_eval_ds, batch_size=256, shuffle=False)

    model = SimpleBrailleCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))

    start_acc, _ = evaluate(model, quick_eval_loader, device)
    print(f"pre-finetune accuracy on eval subset: {start_acc:.4f}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = start_acc
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
        eval_acc, _ = evaluate(model, quick_eval_loader, device)
        print(f"epoch {epoch:3d}  train_loss {train_loss:.4f}  train_acc {train_acc:.4f}  eval_subset_acc {eval_acc:.4f}")

        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(model.state_dict(), args.out_checkpoint)
            print(f"  saved new best checkpoint ({args.out_checkpoint})")

    if best_acc == start_acc:
        print("\nfine-tuning never beat the pre-finetune baseline; keeping the original checkpoint for final eval")
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
    else:
        model.load_state_dict(torch.load(args.out_checkpoint, map_location=device))

    print("\nrunning full DBSI test-set evaluation with best checkpoint...")
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    full_acc, confusion = evaluate(model, test_loader, device)
    print(f"final full DBSI test accuracy: {full_acc:.4f}")

    matrix_path = out_dir / "dbsi_test_confusion_finetuned.npy"
    np.save(matrix_path, confusion)
    print(f"saved {matrix_path}")

    errors = [
        (code_to_label(t), code_to_label(p), confusion[t, p])
        for t in range(NUM_CLASSES) for p in range(NUM_CLASSES)
        if t != p and confusion[t, p] > 0
    ]
    errors.sort(key=lambda x: -x[2])
    print("top confused pairs (true -> predicted : count):")
    for true_label, pred_label, count in errors[:15]:
        print(f"  {true_label} -> {pred_label} : {count}")


if __name__ == "__main__":
    main()
