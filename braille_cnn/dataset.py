import numpy as np
import torch
from torch.utils.data import Dataset

from .normalize import normalize_crop
from .render import render_braille_cell


class SyntheticBrailleDataset(Dataset):
    """Infinite-variety synthetic Braille dataset.

    Train split: each __getitem__ renders a fresh random sample (unseeded),
    so every epoch sees new pose/lighting/noise variation for the same class.
    Test split: seeded per-index so the evaluation set is fixed and reproducible.
    """

    def __init__(self, codes=range(64), samples_per_class=200, img_size=64, train=True, seed=42, max_perspective=0.20):
        self.codes = list(codes)
        self.samples_per_class = samples_per_class
        self.img_size = img_size
        self.train = train
        self.seed = seed
        self.max_perspective = max_perspective

    def __len__(self):
        return len(self.codes) * self.samples_per_class

    def __getitem__(self, idx):
        class_idx = idx % len(self.codes)
        code = self.codes[class_idx]
        rng = np.random.default_rng() if self.train else np.random.default_rng(self.seed + idx)
        img = render_braille_cell(code, img_size=self.img_size, augment=True, rng=rng, max_perspective=self.max_perspective)
        arr = normalize_crop(img)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor, torch.tensor(class_idx, dtype=torch.int64)
