import argparse

import numpy as np
import torch
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, nargs="+", default=[
        "braille_cnn/checkpoints/braille_cnn_best.pt",
        "braille_cnn/checkpoints/braille_cnn_dbsi_finetuned.pt",
    ])
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--img-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cpu")

    no_persp_ds = SyntheticBrailleDataset(samples_per_class=args.samples_per_class, img_size=args.img_size, train=False, max_perspective=0.0)
    with_persp_ds = SyntheticBrailleDataset(samples_per_class=args.samples_per_class, img_size=args.img_size, train=False, max_perspective=0.20)

    no_persp_loader = DataLoader(no_persp_ds, batch_size=256, shuffle=False)
    with_persp_loader = DataLoader(with_persp_ds, batch_size=256, shuffle=False)

    for ckpt_path in args.checkpoints:
        model = SimpleBrailleCNN(num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        acc_no, _ = evaluate(model, no_persp_loader, device)
        acc_with, confusion_with = evaluate(model, with_persp_loader, device)

        print(f"\n{ckpt_path}")
        print(f"  no perspective (max_perspective=0.0):   {acc_no:.4f}")
        print(f"  with perspective (max_perspective=0.20): {acc_with:.4f}")

        errors = [
            (code_to_label(t), code_to_label(p), confusion_with[t, p])
            for t in range(NUM_CLASSES) for p in range(NUM_CLASSES)
            if t != p and confusion_with[t, p] > 0
        ]
        errors.sort(key=lambda x: -x[2])
        if errors:
            print("  top confused pairs under perspective (true -> predicted : count):")
            for true_label, pred_label, count in errors[:10]:
                print(f"    {true_label} -> {pred_label} : {count}")


if __name__ == "__main__":
    main()
