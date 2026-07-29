import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cnn import SimpleBrailleCNN
from .dbsi_dataset import DBSIDataset
from .labels import code_to_label

NUM_CLASSES = 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbsi-root", type=str, default="data DBSI")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, default="braille_cnn/checkpoints/braille_cnn_best.pt")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out-dir", type=str, default="braille_cnn/checkpoints")
    parser.add_argument("--sides", type=str, default="recto,verso", help="comma-separated: recto,verso or just one")
    parser.add_argument("--tag", type=str, default="", help="suffix for saved confusion matrix filename")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sides = tuple(s.strip() for s in args.sides.split(","))
    dataset = DBSIDataset(args.dbsi_root, split=args.split, img_size=args.img_size, sides=sides)
    print(f"DBSI {args.split} split ({','.join(sides)}): {len(dataset)} labeled cells")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SimpleBrailleCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion[t, p] += 1

    acc = correct / total
    print(f"accuracy on DBSI {args.split} [{','.join(sides)}]: {acc:.4f} ({correct}/{total})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    matrix_path = out_dir / f"dbsi_{args.split}{tag}_confusion.npy"
    np.save(matrix_path, confusion)
    print(f"saved {matrix_path}")

    per_class_total = confusion.sum(axis=1)
    per_class_correct = np.diag(confusion)
    print("\nper-class recall (worst 15):")
    recalls = [
        (code_to_label(c), per_class_correct[c] / per_class_total[c], per_class_total[c])
        for c in range(NUM_CLASSES) if per_class_total[c] > 0
    ]
    recalls.sort(key=lambda x: x[1])
    for label, recall, support in recalls[:15]:
        print(f"  {label}: {recall:.3f} (n={support})")

    errors = [
        (code_to_label(t), code_to_label(p), confusion[t, p])
        for t in range(NUM_CLASSES) for p in range(NUM_CLASSES)
        if t != p and confusion[t, p] > 0
    ]
    errors.sort(key=lambda x: -x[2])
    print("\ntop confused pairs (true -> predicted : count):")
    for true_label, pred_label, count in errors[:15]:
        print(f"  {true_label} -> {pred_label} : {count}")


if __name__ == "__main__":
    main()
