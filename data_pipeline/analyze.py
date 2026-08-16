"""Stage 3 - Analysis (EDA): plots and a summary that drive preprocessing choices.

Writes to reports/eda/. This is not decoration; each figure answers a question
that changes a later decision:

  class_distribution   Is the label distribution skewed enough to need class
                       weighting or capping? (Stage 2c --cap-per-class)
  cell_geometry        How far apart are the domains in cell size and pitch?
                       This is the domain gap, quantified.
  dot_fill             Do the datasets put the same share of dots inside a
                       crop? Validates transform.SOURCE_MARGINS.
  brightness           Does normalize_crop actually collapse the per-source
                       brightness differences it is supposed to?
  page_layout          Cells and lines per page, for the detector's max_det.
  crop_samples         A visual check that the box geometry is right at all.
                       Look at this one first - if the crops are not centred
                       on cells, every number downstream is meaningless.

Usage (from repo root):
    py -3.11 -m data_pipeline.analyze
    py -3.11 -m data_pipeline.analyze --samples-per-source 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .contracts import SPLITS, read_manifest, repo_root  # noqa: E402
from .transform import (  # noqa: E402
    IMG_SIZE_DEFAULT,
    dot_fill_fraction,
    extract_crop,
    margin_for,
    to_model_input,
)

ROOT = repo_root()
DEFAULT_IN = ROOT / "data_pipeline" / "manifests" / "manifest_clean.csv"
DEFAULT_OUT = ROOT / "reports" / "eda"

FIGSIZE_WIDE = (11, 4)
FIGSIZE_GRID = (11, 7)


def _save(fig, out_dir: Path, name: str) -> str:
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path.name}")
    return name


def sample_crops(frame, samples_per_source: int, seed: int = 42) -> dict[str, dict]:
    """Extract a random sample of real crops per source, for pixel-level stats.

    Sampling is page-stratified: whole pages are picked first, then cells from
    them, so we open few images instead of one image per cell.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}

    for source, source_rows in frame.groupby("source"):
        pages = source_rows["image_path"].unique()
        rng.shuffle(pages)
        crops: list[np.ndarray] = []
        raw_means: list[float] = []

        for image_path in pages:
            if len(crops) >= samples_per_source:
                break
            image = cv2.imread(str(ROOT / str(image_path)), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            page_rows = source_rows[source_rows["image_path"] == image_path]
            take = min(len(page_rows), max(samples_per_source // 8, 8))
            for row in page_rows.sample(n=take, random_state=seed).itertuples(index=False):
                crop = extract_crop(
                    image, (row.x0, row.y0, row.x1, row.y1),
                    margin=margin_for(source), img_size=IMG_SIZE_DEFAULT,
                )
                if crop is None:
                    continue
                crops.append(crop)
                raw_means.append(float(crop.mean()))
                if len(crops) >= samples_per_source:
                    break

        if crops:
            out[str(source)] = {"crops": np.stack(crops), "raw_means": np.asarray(raw_means)}
            print(f"  sampled {len(crops):,} crops from {source}")
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def plot_class_distribution(frame, out_dir: Path) -> str:
    sources = sorted(frame["source"].unique())
    fig, axes = plt.subplots(len(sources), 1, figsize=(11, 2.6 * len(sources)), squeeze=False)
    for ax, source in zip(axes[:, 0], sources):
        counts = np.bincount(frame.loc[frame["source"] == source, "code"], minlength=64)
        ax.bar(range(64), counts, width=0.85)
        present = int((counts > 0).sum())
        top = int(counts.argmax())
        share = counts.max() / max(counts.sum(), 1)
        ax.set_title(
            f"{source}: {counts.sum():,} cells, {present}/64 classes present, "
            f"most frequent code {top} at {share:.1%}"
        )
        ax.set_xlabel("dot code (0-63)")
        ax.set_ylabel("cells")
        ax.set_yscale("log")
    return _save(fig, out_dir, "class_distribution.png")


def plot_cell_geometry(frame, out_dir: Path) -> str:
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
    widths = frame["x1"] - frame["x0"]
    heights = frame["y1"] - frame["y0"]

    for source in sorted(frame["source"].unique()):
        mask = frame["source"] == source
        axes[0].hist(widths[mask], bins=60, alpha=0.55, label=source)
        axes[1].hist(frame.loc[mask, "dot_pitch_px"], bins=60, alpha=0.55, label=source)
        axes[2].hist((widths / heights)[mask], bins=60, alpha=0.55, label=source)

    axes[0].set_title("cell box width (px)")
    axes[1].set_title("dot pitch (px) = box height / 2")
    axes[2].set_title("box aspect ratio (w/h)")
    for ax in axes:
        ax.legend()
        ax.set_ylabel("cells")
    return _save(fig, out_dir, "cell_geometry.png")


def plot_page_layout(frame, out_dir: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    per_page = frame.groupby(["source", "image_path"]).size().reset_index(name="cells")
    for source in sorted(per_page["source"].unique()):
        values = per_page.loc[per_page["source"] == source, "cells"]
        axes[0].hist(values, bins=30, alpha=0.55, label=f"{source} (max {values.max():,})")
    axes[0].set_title("cells per page")
    axes[0].set_xlabel("cells")
    axes[0].legend()

    page_sizes = frame.groupby("image_path")[["img_w", "img_h"]].first()
    axes[1].scatter(page_sizes["img_w"], page_sizes["img_h"], s=8, alpha=0.5)
    axes[1].set_title("page resolution")
    axes[1].set_xlabel("width (px)")
    axes[1].set_ylabel("height (px)")
    return _save(fig, out_dir, "page_layout.png")


def plot_dot_fill(samples: dict, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    for source, payload in sorted(samples.items()):
        fill = np.array([dot_fill_fraction(c) for c in payload["crops"]])
        ax.hist(fill, bins=40, alpha=0.55, label=f"{source} (median {np.median(fill):.3f})")
    ax.set_title("dot fill fraction per crop - closer curves mean margins are matched")
    ax.set_xlabel("fraction of crop occupied by dot structure")
    ax.set_ylabel("crops")
    ax.legend()
    return _save(fig, out_dir, "dot_fill.png")


def plot_brightness(samples: dict, out_dir: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharey=True)
    for source, payload in sorted(samples.items()):
        axes[0].hist(payload["raw_means"], bins=50, alpha=0.55, label=source)
        after = np.array([to_model_input(c).mean() for c in payload["crops"]])
        axes[1].hist(after, bins=50, alpha=0.55, label=source)
    axes[0].set_title("raw crop mean (0-255) - before normalize_crop")
    axes[1].set_title("crop mean (0-1) - after normalize_crop")
    for ax in axes:
        ax.legend()
        ax.set_ylabel("crops")
    return _save(fig, out_dir, "brightness.png")


def plot_crop_samples(samples: dict, out_dir: Path) -> str:
    """Visual sanity check. Wrong boxes are obvious here and nowhere else."""
    cols = 10
    sources = sorted(samples)
    rows_per_source = 2
    fig, axes = plt.subplots(
        len(sources) * rows_per_source, cols,
        figsize=FIGSIZE_GRID, squeeze=False,
    )
    for s_idx, source in enumerate(sources):
        crops = samples[source]["crops"]
        picks = np.linspace(0, len(crops) - 1, cols * rows_per_source).astype(int)
        for k, pick in enumerate(picks):
            ax = axes[s_idx * rows_per_source + k // cols][k % cols]
            ax.imshow(crops[pick], cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(source, fontsize=9)
    fig.suptitle("Sample cell crops - each should be one Braille cell, centred")
    return _save(fig, out_dir, "crop_samples.png")


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------


def write_summary(frame, samples: dict, figures: list[str], out_dir: Path) -> Path:
    lines = ["# Stage 3 - exploratory data analysis", ""]

    lines += ["## Dataset totals", "",
              "| Source | Cells | Images | Page groups | Classes present |",
              "|---|---:|---:|---:|---:|"]
    for source, group in frame.groupby("source"):
        present = int((np.bincount(group["code"], minlength=64) > 0).sum())
        lines.append(
            f"| {source} | {len(group):,} | {group['image_path'].nunique():,} | "
            f"{group['page_group'].nunique():,} | {present}/64 |"
        )

    lines += ["", "## Split integrity", "",
              "| Source | " + " | ".join(SPLITS) + " |",
              "|---|" + "---:|" * len(SPLITS)]
    for source, group in frame.groupby("source"):
        counts = group["split"].value_counts()
        lines.append(
            f"| {source} | " + " | ".join(f"{counts.get(s, 0):,}" for s in SPLITS) + " |"
        )
    spans = frame.groupby("page_group")["split"].nunique()
    leaked = int((spans > 1).sum())
    lines += ["", f"Page groups spanning more than one split: **{leaked}** "
                  f"({'OK' if leaked == 0 else 'LEAKAGE - fix before training'})."]

    lines += ["", "## Class imbalance", ""]
    for source, group in frame.groupby("source"):
        counts = np.bincount(group["code"], minlength=64)
        nonzero = counts[counts > 0]
        lines.append(
            f"- **{source}**: most frequent code {int(counts.argmax())} holds "
            f"{counts.max() / counts.sum():.1%} of cells; "
            f"ratio between most and least frequent present class is "
            f"{nonzero.max() / nonzero.min():,.0f}x; "
            f"{int((counts == 0).sum())} codes never appear."
        )
    lines += ["", "This is what justifies class weighting in training and the "
                  "`--cap-per-class` option in Stage 2c."]

    lines += ["", "## Geometry and domain gap", "",
              "| Source | Median box w x h (px) | Median dot pitch (px) | "
              "Median page (px) | Median dot fill |", "|---|---|---:|---|---:|"]
    for source, group in frame.groupby("source"):
        width = (group["x1"] - group["x0"]).median()
        height = (group["y1"] - group["y0"]).median()
        pages = group.groupby("image_path")[["img_w", "img_h"]].first().median()
        fill = samples.get(str(source))
        fill_text = (
            f"{np.median([dot_fill_fraction(c) for c in fill['crops']]):.3f}"
            if fill else "n/a"
        )
        lines.append(
            f"| {source} | {width:.0f} x {height:.0f} | {group['dot_pitch_px'].median():.1f} | "
            f"{pages['img_w']:.0f} x {pages['img_h']:.0f} | {fill_text} |"
        )

    lines += ["", "## Detector sizing", ""]
    per_page = frame.groupby("image_path").size()
    lines.append(
        f"Cells per page: median {per_page.median():.0f}, "
        f"95th percentile {per_page.quantile(0.95):.0f}, max {per_page.max():,}. "
        f"Set the cell detector's `max_det` above the max, not above the median."
    )

    lines += ["", "## Figures", ""]
    for name in figures:
        lines.append(f"### {name}\n\n![{name}]({name})\n")

    path = out_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 - EDA over the cell manifest")
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--samples-per-source", type=int, default=400,
                        help="Real crops sampled per source for pixel-level figures")
    args = parser.parse_args()

    frame = read_manifest(args.in_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Read {len(frame):,} cells from {args.in_path}")

    print("Sampling real crops ...")
    samples = sample_crops(frame, args.samples_per_source)

    print("Writing figures ...")
    figures = [
        plot_class_distribution(frame, args.out_dir),
        plot_cell_geometry(frame, args.out_dir),
        plot_page_layout(frame, args.out_dir),
    ]
    if samples:
        figures += [
            plot_crop_samples(samples, args.out_dir),
            plot_dot_fill(samples, args.out_dir),
            plot_brightness(samples, args.out_dir),
        ]

    summary = write_summary(frame, samples, figures, args.out_dir)
    print(f"\nWrote {summary}")
    print("Look at crop_samples.png first - it is the check that the cell boxes are right.")


if __name__ == "__main__":
    main()
