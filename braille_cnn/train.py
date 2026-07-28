import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .cnn import SimpleBrailleCNN
from .dataset import SyntheticBrailleDataset
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


def print_top_confusions(confusion, top_k=10):
    pairs = [
        (code_to_label(t), code_to_label(p), confusion[t, p])
        for t in range(NUM_CLASSES) for p in range(NUM_CLASSES)
        if t != p and confusion[t, p] > 0
    ]
    pairs.sort(key=lambda x: -x[2])
    print("top confused pairs (true -> predicted : count):")
    for true_label, pred_label, count in pairs[:top_k]:
        print(f"  {true_label} -> {pred_label} : {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-samples-per-class", type=int, default=300)
    parser.add_argument("--test-samples-per-class", type=int, default=30)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", type=str, default="braille_cnn/checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = SyntheticBrailleDataset(samples_per_class=args.train_samples_per_class, img_size=args.img_size, train=True)
    test_ds = SyntheticBrailleDataset(samples_per_class=args.test_samples_per_class, img_size=args.img_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = SimpleBrailleCNN(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    best_confusion = None
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
        test_acc, confusion = evaluate(model, test_loader, device)
        print(f"epoch {epoch:3d}  train_loss {train_loss:.4f}  train_acc {train_acc:.4f}  test_acc {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_confusion = confusion
            torch.save(model.state_dict(), out_dir / "braille_cnn_best.pt")

    print(f"\nbest test accuracy: {best_acc:.4f}")
    np.save(out_dir / "confusion_matrix.npy", best_confusion)
    print_top_confusions(best_confusion)


if __name__ == "__main__":
    main()
