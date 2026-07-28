"""Loads real photographed/scanned Braille cells from the DSBI dataset
(data DBSI/), using its ground-truth per-dot pixel annotations, so a model
trained on synthetic data can be evaluated against (or later fine-tuned on)
real images.

Annotation format (see "data DBSI/DBSI-README.md"): for each de-skewed
+recto.jpg / +verso.jpg page image, the matching +recto.txt / +verso.txt has:
  line 1: skew angle (informational; image is already de-skewed)
  line 2: x-pixel position of each dot-column, 2 per Braille-cell-column
  line 3: y-pixel position of each dot-row, 3 per Braille-cell-row
  remaining lines: "row col d1 d2 d3 d4 d5 d6" (1-indexed cell grid position
  + its 6 dots, in the same dot1..dot6 order used throughout this package)
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _parse_split_file(split_path):
    entries = []
    with open(split_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            book, filename = line.replace("/", "\\").split("\\")
            base = filename[:-4] if filename.lower().endswith(".jpg") else filename
            entries.append((book, base))
    return entries


def _parse_annotation(txt_path):
    with open(txt_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    if len(lines) <= 3:
        return None  # page has no dots on this side
    verticals = [int(x) for x in lines[1].split()]
    horizontals = [int(x) for x in lines[2].split()]
    cells = []
    for line in lines[3:]:
        if not line.strip():
            continue
        parts = line.split()
        row, col = int(parts[0]), int(parts[1])
        dots = tuple(int(x) for x in parts[2:8])
        cells.append((row, col, dots))
    return verticals, horizontals, cells


def _dots_to_code(dots) -> int:
    return sum((1 << i) for i, d in enumerate(dots) if d == 1)


def _cell_box(verticals, horizontals, row, col, margin_scale=0.8):
    x0 = verticals[(col - 1) * 2]
    x1 = verticals[(col - 1) * 2 + 1]
    y0 = horizontals[(row - 1) * 3]
    y1 = horizontals[(row - 1) * 3 + 1]
    y2 = horizontals[(row - 1) * 3 + 2]
    margin_x = max(x1 - x0, 1) * margin_scale
    margin_y = max(y1 - y0, 1) * margin_scale
    return (x0 - margin_x, y0 - margin_y, x1 + margin_x, y2 + margin_y)


class DBSIDataset(Dataset):
    """Eagerly decodes and crops every cell once at construction time (each
    page image is opened exactly once), instead of re-opening full-resolution
    page JPEGs per __getitem__ call. With shuffle=True during training,
    per-sample lazy loading defeats any single-image cache and turns training
    into a full-page JPEG decode on almost every sample -- this avoids that.
    """

    def __init__(self, root, split="test", img_size=64, sides=("recto", "verso"), margin_scale=0.8):
        self.root = Path(root)
        self.img_size = img_size
        entries = _parse_split_file(self.root / f"{split}.txt")

        pending = []  # (image_path, box, code), grouped by image for sequential decode
        for book, base in entries:
            book_dir = self.root / book
            for side in sides:
                txt_path = book_dir / f"{base}+{side}.txt"
                img_path = book_dir / f"{base}+{side}.jpg"
                if not txt_path.exists() or not img_path.exists():
                    continue
                parsed = _parse_annotation(txt_path)
                if parsed is None:
                    continue
                verticals, horizontals, cells = parsed
                for row, col, dots in cells:
                    box = _cell_box(verticals, horizontals, row, col, margin_scale)
                    pending.append((str(img_path), box, _dots_to_code(dots)))

        pending.sort(key=lambda item: item[0])

        # stored as uint8 (not float32) to keep tens of thousands of crops in
        # memory cheaply; converted to float only per-batch in __getitem__
        crops = np.empty((len(pending), img_size, img_size), dtype=np.uint8)
        labels = np.empty(len(pending), dtype=np.int64)
        current_path, current_img = None, None
        for i, (path, box, code) in enumerate(pending):
            if path != current_path:
                current_img = Image.open(path).convert("L")
                current_path = path
            crop = current_img.crop(tuple(int(round(v)) for v in box))
            crop = crop.resize((img_size, img_size), Image.Resampling.BICUBIC)
            crops[i] = np.asarray(crop, dtype=np.uint8)
            labels[i] = code

        self.crops = torch.from_numpy(crops).unsqueeze(1)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.crops[idx].float() / 255.0, self.labels[idx]
