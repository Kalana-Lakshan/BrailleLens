"""Loads real handheld-phone-photo Braille cells from the Angelina dataset
(AngelinaDataset-master/books/), for fine-tuning on the domain gap this
project has been chasing all session: DBSI's checkpoints/calibrations are
tuned for a flatbed scanner, and confirmed NOT to transfer to real photos.

Format (see AngelinaDataset-master/README.md): each `<image>.labeled.jpg`
has a matching `<image>.labeled.csv` with one row per Braille cell:
    left;top;right;bottom;label
left/top/right/bottom are normalized [0,1) box coordinates (multiply by the
image's own width/height for pixels). label is an int 0-63 using the exact
same bit convention as this project's own dots_to_code (dot i -> bit i-1) --
verified directly against AngelinaDataset-master/src/label_tools.py
(`v = [1, 2, 4, 8, 16, 32]`), so no conversion is needed. Boxes are
consistently ~full-cell-sized regardless of how many dots are active (not
tightly cropped to just the active dots), so they can be used directly like
DBSI's cell boxes.

label=63 (all 6 dots) is reserved by this dataset's annotation convention
for "XX"/markout (illegible/crossed-out text), not necessarily a real
6-dot cell -- excluded here as a safety measure, even though it's rare.

Train/val split: uses the dataset's own predefined books/train.txt and
books/val.txt (not a random split), so results are comparable to anything
else evaluated against this dataset's standard split.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MARKOUT_CODE = 63


def _read_csv_annotation(csv_path):
    rects = []
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            left, top, right, bottom, label = line.split(";")
            code = int(label)
            if code == MARKOUT_CODE:
                continue
            rects.append((float(left), float(top), float(right), float(bottom), code))
    return rects


def _list_split(books_root, split):
    """books/train.txt or books/val.txt: one relative image path per line."""
    list_path = Path(books_root) / f"{split}.txt"
    paths = []
    with open(list_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace("\\", "/")
            if line:
                paths.append(Path(books_root) / line)
    return paths


class AngelinaDataset(Dataset):
    """Eagerly decodes and crops every cell once at construction time, same
    reasoning as DBSIDataset: lazy per-sample loading with shuffle=True
    would re-decode full-resolution photos almost every sample.
    """

    def __init__(self, books_root, split="train", img_size=64, margin_scale=0.15, max_images=None):
        self.img_size = img_size
        image_paths = _list_split(books_root, split)
        if max_images is not None:
            image_paths = image_paths[:max_images]

        crops, labels = [], []
        for img_path in image_paths:
            csv_path = img_path.with_suffix("").with_suffix(".csv") if img_path.suffix == ".jpg" else img_path
            # <name>.labeled.jpg -> <name>.labeled.csv
            csv_path = img_path.parent / (img_path.stem + ".csv")
            if not img_path.exists() or not csv_path.exists():
                continue
            rects = _read_csv_annotation(csv_path)
            if not rects:
                continue
            image = Image.open(img_path).convert("L")
            w, h = image.size
            for left, top, right, bottom, code in rects:
                x0, y0, x1, y1 = left * w, top * h, right * w, bottom * h
                bw, bh = x1 - x0, y1 - y0
                mx, my = bw * margin_scale, bh * margin_scale
                box = (
                    max(x0 - mx, 0), max(y0 - my, 0),
                    min(x1 + mx, w), min(y1 + my, h),
                )
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                crop = image.crop(tuple(int(round(v)) for v in box)).resize(
                    (img_size, img_size), Image.Resampling.BICUBIC
                )
                crops.append(np.asarray(crop, dtype=np.uint8))
                labels.append(code)

        self.crops = torch.from_numpy(np.stack(crops)).unsqueeze(1) if crops else torch.empty(0, 1, img_size, img_size)
        self.labels = torch.tensor(labels, dtype=torch.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        from .normalize import normalize_crop
        arr = normalize_crop(self.crops[idx].squeeze(0).numpy())
        return torch.from_numpy(arr).unsqueeze(0), self.labels[idx]
