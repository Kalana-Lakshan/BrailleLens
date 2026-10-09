"""Rebuild training / validation curves from the robust fingertip checkpoint (no retraining).

Ultralytics stores the full per-epoch results table in the checkpoint under "train_results".
Writes results.csv plus figures to fingertip_robust/results/.

    python finger_cell_track/fingertip_robust/plot_training_curves.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CKPT = HERE.parent / "weights" / "yolo26n_fingertip_robust_best.pt"
OUT = HERE / "results"


def smooth(y: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Gaussian smoothing with edge padding (same idea as Ultralytics' plot_results)."""
    r = int(4 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(y, r, mode="edge"), k, mode="valid")


def plot_grid(res: dict, epochs: np.ndarray, keys: list[str], ncols: int, path: Path) -> None:
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        y = np.asarray(res[key], dtype=float)
        ax.plot(epochs, y, marker=".", markersize=8, linewidth=2, label="results")
        ax.plot(epochs, smooth(y), ":", linewidth=2, label="smooth")
        ax.set_title(key.replace("(B)", ""), fontsize=12)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    axes.flat[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)


def main() -> None:
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    res = ck["train_results"]
    epochs = np.asarray(res["epoch"], dtype=int)
    OUT.mkdir(parents=True, exist_ok=True)

    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(res.keys())
        w.writerows(zip(*res.values()))
    print("saved", OUT / "results.csv")

    train = ["train/box_loss", "train/cls_loss", "train/l1_loss"]
    val = ["val/box_loss", "val/cls_loss", "val/l1_loss"]
    metrics = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]

    plot_grid(res, epochs, train, 3, OUT / "train_losses.png")
    plot_grid(res, epochs, val, 3, OUT / "val_losses.png")
    plot_grid(res, epochs, metrics, 4, OUT / "val_metrics.png")
    plot_grid(res, epochs, train + metrics[:2] + val + metrics[2:], 5, OUT / "results.png")

    best = int(np.argmax(0.1 * np.asarray(res["metrics/mAP50(B)"]) + 0.9 * np.asarray(res["metrics/mAP50-95(B)"])))
    print(f"best epoch {epochs[best]}: P={res['metrics/precision(B)'][best]:.3f} "
          f"R={res['metrics/recall(B)'][best]:.3f} mAP50={res['metrics/mAP50(B)'][best]:.3f} "
          f"mAP50-95={res['metrics/mAP50-95(B)'][best]:.3f}")


if __name__ == "__main__":
    main()
