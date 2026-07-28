import argparse

import matplotlib.pyplot as plt
import torch

from .cnn import SimpleBrailleCNN
from .dbsi_dataset import DBSIDataset
from .labels import code_to_label

NUM_CLASSES = 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbsi-root", type=str, default="data DBSI")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, default="braille_cnn/checkpoints/braille_cnn_dbsi_finetuned.pt")
    parser.add_argument("--sides", type=str, default="recto,verso")
    parser.add_argument("--true-class", type=int, required=True)
    parser.add_argument("--pred-class", type=int, default=None, help="only show cases misclassified as this class")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--out", type=str, default="braille_cnn/checkpoints/error_inspection.png")
    args = parser.parse_args()

    device = torch.device("cpu")
    sides = tuple(s.strip() for s in args.sides.split(","))
    dataset = DBSIDataset(args.dbsi_root, split=args.split, img_size=args.img_size, sides=sides)

    model = SimpleBrailleCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    matches = []
    with torch.no_grad():
        batch_size = 512
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            images = dataset.crops[start:end].float() / 255.0
            labels = dataset.labels[start:end]
            preds = model(images).argmax(dim=1)
            for i in range(end - start):
                t, p = labels[i].item(), preds[i].item()
                if t != args.true_class:
                    continue
                if args.pred_class is not None and p != args.pred_class:
                    continue
                matches.append((start + i, p))
                if len(matches) >= args.n:
                    break
            if len(matches) >= args.n:
                break

    print(f"found {len(matches)} examples of true={args.true_class} ({code_to_label(args.true_class)})"
          + (f" predicted as {args.pred_class} ({code_to_label(args.pred_class)})" if args.pred_class is not None else " (any prediction)"))

    if not matches:
        return

    cols = 4
    rows = (len(matches) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flat if len(matches) > 1 else [axes]
    for ax, (idx, pred) in zip(axes, matches):
        img = dataset.crops[idx].squeeze().numpy()
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"true={code_to_label(args.true_class)} pred={code_to_label(pred)}", fontsize=9)
        ax.axis("off")
    for ax in list(axes)[len(matches):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
