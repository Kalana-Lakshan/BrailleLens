import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

from .labels import code_to_label

NUM_CLASSES = 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=str, default="braille_cnn/checkpoints/confusion_matrix.npy")
    parser.add_argument("--csv-out", type=str, default="braille_cnn/checkpoints/confusion_matrix.csv")
    parser.add_argument("--png-out", type=str, default="braille_cnn/checkpoints/confusion_matrix.png")
    args = parser.parse_args()

    confusion = np.load(args.matrix)
    labels = [code_to_label(c) for c in range(NUM_CLASSES)]

    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + labels)
        for i, row in enumerate(confusion):
            writer.writerow([labels[i]] + row.tolist())
    print(f"saved {args.csv_out}")

    per_class_total = confusion.sum(axis=1, keepdims=True)
    per_class_total[per_class_total == 0] = 1
    normalized = confusion / per_class_total

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(normalized, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.colorbar(im, ax=ax, label="fraction of true class")
    fig.tight_layout()
    fig.savefig(args.png_out, dpi=150)
    print(f"saved {args.png_out}")

    total = confusion.sum()
    correct = np.trace(confusion)
    print(f"overall accuracy: {correct / total:.4f} ({correct}/{total})")

    errors = [
        (labels[t], labels[p], confusion[t, p])
        for t in range(NUM_CLASSES) for p in range(NUM_CLASSES)
        if t != p and confusion[t, p] > 0
    ]
    errors.sort(key=lambda x: -x[2])
    if errors:
        print("top confused pairs (true -> predicted : count):")
        for true_label, pred_label, count in errors[:15]:
            print(f"  {true_label} -> {pred_label} : {count}")
    else:
        print("no misclassifications in this confusion matrix")


if __name__ == "__main__":
    main()
