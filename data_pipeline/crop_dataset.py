"""Torch Dataset over the Stage 2c crop archives.

Loads one crops_<split>.npz into memory as uint8 and converts to float only per
sample, so a 73k-crop training split costs about 300 MB rather than 1.2 GB.

Augmentation is applied here rather than baked into the npz, so one extraction
run can serve many training runs with different jitter settings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .transform import Augmenter, to_model_input

NUM_CLASSES = 64


class CropDataset(Dataset):
    """Cell crops from one split of the Stage 2c archives.

    augment=True should be used for training splits only. Each worker gets its
    own RNG seeded from the sample index so augmentation is reproducible and
    does not repeat identically across workers.
    """

    def __init__(
        self,
        npz_path: str | Path,
        augment: bool = False,
        augmenter: Augmenter | None = None,
        sources: list[str] | None = None,
        seed: int = 0,
    ) -> None:
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Crop archive not found: {npz_path}\n"
                "Build it first: py -3.11 -m data_pipeline.reduce"
            )
        with np.load(npz_path, allow_pickle=False) as data:
            crops = data["crops"]
            codes = data["codes"]
            source_names = data["sources"]
            page_groups = data["page_groups"]

        if sources:
            keep = np.isin(source_names, list(sources))
            crops, codes = crops[keep], codes[keep]
            source_names, page_groups = source_names[keep], page_groups[keep]

        self.crops = crops
        self.codes = codes.astype(np.int64)
        self.sources = source_names
        self.page_groups = page_groups
        self.augment = augment
        self.augmenter = augmenter or Augmenter()
        self.seed = seed

    def __len__(self) -> int:
        return len(self.codes)

    def __getitem__(self, idx: int):
        crop = self.crops[idx]
        if self.augment:
            rng = np.random.default_rng((self.seed + idx) % (2**32))
            crop = self.augmenter(crop, rng)
        arr = to_model_input(crop)
        return torch.from_numpy(arr).unsqueeze(0), int(self.codes[idx])

    # ---------------------------------------------------------------- helpers

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.codes, minlength=NUM_CLASSES)

    def source_counts(self) -> dict[str, int]:
        names, counts = np.unique(self.sources, return_counts=True)
        return {str(n): int(c) for n, c in zip(names, counts)}

    def class_weights(self, floor: int = 1) -> torch.Tensor:
        """Inverse-frequency weights for CrossEntropyLoss.

        Normalised to mean 1 so the effective learning rate does not change
        when weighting is switched on. Absent classes get weight 0 rather than
        infinity.
        """
        counts = self.class_counts().astype(np.float64)
        weights = np.zeros_like(counts)
        present = counts >= floor
        weights[present] = counts[present].sum() / (present.sum() * counts[present])
        if weights[present].mean() > 0:
            weights[present] /= weights[present].mean()
        return torch.tensor(weights, dtype=torch.float32)

    def domain_balanced_sampler(self, target_ratio: dict[str, float] | None = None):
        """Sampler that draws sources at fixed proportions.

        DSBI has far more cells than Angelina, so uniform sampling makes the
        model mostly a flatbed-scanner model. This lets a batch be composed to
        a chosen mix instead - for example equal parts of each domain - which is
        what stopped the catastrophic forgetting seen in earlier runs.
        """
        counts = self.source_counts()
        if target_ratio is None:
            target_ratio = {name: 1.0 / len(counts) for name in counts}

        total = sum(target_ratio.get(name, 0.0) for name in counts)
        if total <= 0:
            raise ValueError(f"target_ratio {target_ratio} matches no source in {list(counts)}")

        per_sample = {
            name: (target_ratio.get(name, 0.0) / total) / max(counts[name], 1)
            for name in counts
        }
        weights = np.array([per_sample[str(s)] for s in self.sources], dtype=np.float64)
        return WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=len(self),
            replacement=True,
        )

    def describe(self) -> str:
        counts = self.class_counts()
        present = int((counts > 0).sum())
        return (
            f"{len(self):,} crops  {present}/64 classes  "
            f"sources={self.source_counts()}  "
            f"pages={len(np.unique(self.page_groups)):,}"
        )
