"""Stage 6 — cell-CNN accuracy on Angelina (handheld photos).

Uses the Stage 2c crop archive when present, otherwise falls back to
AngelinaDataset so this script works before reduce.py has been run.

    py -3.11 -m braille_cnn.eval_angelina
    py -3.11 -m braille_cnn.eval_angelina --checkpoint braille_cnn/checkpoints/braille_cnn_mixed.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cnn import SimpleBrailleCNN
from .labels import code_to_label

NUM_CLASSES = 64


def _load_dataset(args):
    npz = Path(args.crops_dir) / f"crops_{args.split}.npz"
    if npz.exists():
        from data_pipeline.crop_dataset import CropDataset

        ds = CropDataset(npz, augment=False, sources=["angelina"])
        print(f"Angelina from {npz}: {ds.describe()}")
        return ds

    from .angelina_dataset import AngelinaDataset

    split = "val" if args.split == "test" else args.split
    ds = AngelinaDataset(args.angelina_root, split=split, img_size=args.img_size)
    print(f"AngelinaDataset {split}: {len(ds)} cells  ({args.angelina_root})")
    return ds


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the cell CNN on Angelina")
    parser.add_argument("--checkpoint", default="braille_cnn/checkpoints/braille_cnn_mixed.pt")
    parser.add_argument("--crops-dir", default="data_pipeline/crops")
    parser.add_argument("--angelina-root", default="data Angelina/books")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out-dir", default="braille_cnn/checkpoints")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        alt = Path("braille_cnn/checkpoints/braille_cnn_dbsi_finetuned.pt")
        if alt.exists():
            print(f"{ckpt} missing; using {alt}")
            ckpt = alt
        else:
            raise SystemExit(f"Checkpoint not found: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = _load_dataset(args)
    if len(dataset) == 0:
        raise SystemExit("No Angelina cells found. Run data_pipeline.integrate / reduce first.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = SimpleBrailleCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
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

    acc = correct / max(total, 1)
    print(f"Angelina {args.split} accuracy: {acc:.4f} ({correct}/{total})")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"angelina_{args.split}_confusion.npy", confusion)

    support = confusion.sum(axis=1)
    recalls = [
        (code_to_label(c), confusion[c, c] / support[c], int(support[c]))
        for c in range(NUM_CLASSES)
        if support[c] > 0
    ]
    recalls.sort(key=lambda x: x[1])
    print("\nworst 10 class recalls:")
    for label, rec, n in recalls[:10]:
        print(f"  {label}: {rec:.3f} (n={n})")


if __name__ == "__main__":
    main()
