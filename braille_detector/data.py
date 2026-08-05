"""Page-level data loading for the detector: DBSI + Angelina, each page
providing a full image plus a list of (box, code) ground-truth cells --
what a single-stage object detector needs (unlike braille_cnn/'s per-cell
classifiers, which only ever see one pre-cropped cell at a time).

Kept self-contained (own annotation parsers, not importing braille_cnn/)
per this folder's separation from the existing pipeline.
"""

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .boxes import mirror_code

# Per-dataset pixel-scale normalization: DBSI (~200dpi flatbed scans) and
# Angelina (handheld phone photos) have very different absolute dot pitch
# in pixels (confirmed this session: DBSI dx=~21px, Angelina dx=~13.5px on
# a representative page/photo). The detector uses one fixed anchor size
# for the whole training set, so both sources are rescaled toward a common
# target dot pitch (16px, matching the model's 16x16 feature-map stride --
# see model.py) before cropping. This is a coarse, fixed-factor
# approximation (not a per-image auto-measurement, to avoid depending on
# the very grid-fitting machinery this detector is meant to replace) --
# good enough for a first working version, not a precise calibration.
DBSI_TARGET_DX = 21.2
ANGELINA_TARGET_DX = 13.5
DETECTOR_TARGET_DX = 16.0
DBSI_SCALE = DETECTOR_TARGET_DX / DBSI_TARGET_DX
ANGELINA_SCALE = DETECTOR_TARGET_DX / ANGELINA_TARGET_DX

DBSI_SUBJECT_DIRS = [
    "Fundamentals of Massage",
    "Massage",
    "Math",
    "Shaver Yang Fengting",
    "The Second Volume of Ninth Grade Chinese Book 1",
    "The Second Volume of Ninth Grade Chinese Book 2",
]


def dots_to_code(dots):
    return sum((1 << i) for i, d in enumerate(dots) if d == 1)


def parse_dbsi_txt(txt_path):
    """Returns list of (x0, y0, x1, y1, code) tight cell boxes. Some DBSI
    pages (blank facing/title pages between chapters) have an empty
    annotation file -- no grid, no cells -- returned as an empty list."""
    with open(txt_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) < 3:
        return []
    v = [float(x) for x in lines[1].split()]
    h = [float(x) for x in lines[2].split()]
    boxes = []
    for line in lines[3:]:
        parts = line.split()
        row, col = int(parts[0]), int(parts[1])
        dots = [int(x) for x in parts[2:8]]
        code = dots_to_code(dots)
        if code == 0:
            continue
        x0, x1 = v[(col - 1) * 2], v[(col - 1) * 2 + 1]
        y0, y1, y2 = h[(row - 1) * 3], h[(row - 1) * 3 + 1], h[(row - 1) * 3 + 2]
        boxes.append((x0, y0, x1, y2, code))
    return boxes


def parse_angelina_csv(csv_path, img_w, img_h, markout_code=63):
    boxes = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if not row:
                continue
            left, top, right, bottom, label = row
            code = int(label)
            if code == 0 or code == markout_code:
                continue
            boxes.append((float(left) * img_w, float(top) * img_h,
                          float(right) * img_w, float(bottom) * img_h, code))
    return boxes


def list_dbsi_pages(root="data DBSI"):
    root = Path(root)
    pages = []
    for subject in DBSI_SUBJECT_DIRS:
        subject_dir = root / subject
        if not subject_dir.exists():
            continue
        for txt_path in sorted(subject_dir.glob("*+recto.txt")) + sorted(subject_dir.glob("*+verso.txt")):
            img_path = txt_path.with_suffix(".jpg")
            if img_path.exists():
                pages.append((img_path, txt_path, "dbsi"))
    return pages


def list_angelina_pages(books_root="AngelinaDataset-master/books", split="train"):
    books_root = Path(books_root)
    pages = []
    with open(books_root / f"{split}.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace("\\", "/")
            if not line:
                continue
            img_path = books_root / line
            csv_path = img_path.with_suffix("").with_suffix(".csv") if img_path.suffix != ".csv" else img_path
            # img_path looks like .../IMG_x.labeled.jpg -> csv is .../IMG_x.labeled.csv
            csv_path = img_path.parent / (img_path.stem + ".csv")
            if img_path.exists() and csv_path.exists():
                pages.append((img_path, csv_path, "angelina"))
    return pages


def _rotate_boxes(boxes, angle_deg, cx, cy, ncx, ncy):
    """Rotates each box's 4 corners the same way PIL's Image.rotate(angle_deg)
    rotates the image content, re-centers on the rotated (possibly expanded)
    canvas, and returns the axis-aligned bounding box of the 4 rotated
    corners. The true rotated shape of a box is a parallelogram, not a box
    -- axis-aligned is a standard, fine approximation at the small (+-5
    degree) angles used here.

    Sign convention verified empirically (not derived by hand, to avoid a
    silent y-down/CCW sign bug corrupting every box label in training): a
    single marked pixel at a known offset was rotated with PIL and its
    actual output position compared against candidate formulas -- this one
    (theta negated relative to the textbook CCW-in-math-coords formula)
    matched PIL's actual behavior.
    """
    theta = np.radians(-angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    corners_x = np.stack([x0, x1, x0, x1], axis=1)
    corners_y = np.stack([y0, y0, y1, y1], axis=1)
    dx, dy = corners_x - cx, corners_y - cy
    rx = dx * cos_t - dy * sin_t
    ry = dx * sin_t + dy * cos_t
    new_x, new_y = ncx + rx, ncy + ry
    out = boxes.copy()
    out[:, 0], out[:, 1] = new_x.min(axis=1), new_y.min(axis=1)
    out[:, 2], out[:, 3] = new_x.max(axis=1), new_y.max(axis=1)
    return out


class BraillePageDataset(Dataset):
    """One sample = one random crop_size x crop_size window from a random
    page, rescaled to a common dot-pitch target, with the ground-truth
    boxes inside that window (clipped at the edges), scale-jitter (+-30%
    zoom, +-10% independent vertical stretch), light rotation (+-5 degrees,
    70% of samples), and a horizontal-flip augmentation (mirroring box
    x-coordinates AND each affected code's dot pattern via mirror_code) --
    matching the paper's augmentation set (see train.py's docstring history
    for why this was added after the first training run: without it, the
    model only ever saw two fixed calibrated scales and was not robust to
    a third, differently-scaled photo).

    """

    def __init__(self, pages, crop_size=416, epoch_length=2000, train=True):
        self.pages = pages
        self.crop_size = crop_size
        self.epoch_length = epoch_length
        self.train = train
        self._cache = {}

    def __len__(self):
        return self.epoch_length if self.train else len(self.pages)

    def _load_page(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        img_path, ann_path, source = self.pages[idx]
        image = Image.open(img_path).convert("L")
        w, h = image.size
        if source == "dbsi":
            boxes = parse_dbsi_txt(ann_path)
            scale = DBSI_SCALE
        else:
            boxes = parse_angelina_csv(ann_path, w, h)
            scale = ANGELINA_SCALE
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        boxes = np.array([(x0 * scale, y0 * scale, x1 * scale, y1 * scale, c)
                           for x0, y0, x1, y1, c in boxes], dtype=np.float64) if boxes else np.empty((0, 5))
        result = (image, boxes)
        if len(self._cache) < 64:  # small LRU-less cache, bounded to avoid unbounded memory growth
            self._cache[idx] = result
        return result

    def __getitem__(self, idx):
        if self.train:
            page_idx = np.random.randint(len(self.pages))
        else:
            page_idx = idx
        image, boxes = self._load_page(page_idx)
        w, h = image.size
        cs = self.crop_size

        if not self.train:
            if w <= cs or h <= cs:
                pad_w, pad_h = max(0, cs - w + 1), max(0, cs - h + 1)
                padded = Image.new("L", (w + pad_w, h + pad_h), color=255)
                padded.paste(image, (0, 0))
                image = padded
                w, h = image.size
            crop = image.crop((0, 0, cs, cs))
            crop_boxes = boxes.copy()
            if len(crop_boxes):
                crop_boxes[:, 0] = np.clip(crop_boxes[:, 0], 0, cs)
                crop_boxes[:, 1] = np.clip(crop_boxes[:, 1], 0, cs)
                crop_boxes[:, 2] = np.clip(crop_boxes[:, 2], 0, cs)
                crop_boxes[:, 3] = np.clip(crop_boxes[:, 3], 0, cs)
                cx = (crop_boxes[:, 0] + crop_boxes[:, 2]) / 2
                cy = (crop_boxes[:, 1] + crop_boxes[:, 3]) / 2
                keep = (cx > 0) & (cx < cs) & (cy > 0) & (cy < cs) & \
                       (crop_boxes[:, 2] - crop_boxes[:, 0] > 2) & (crop_boxes[:, 3] - crop_boxes[:, 1] > 2)
                crop_boxes = crop_boxes[keep]
        else:
            # Scale-jitter (paper: random resize +-30%) is implemented as a
            # "zoom": a window smaller/larger than crop_size is grabbed and
            # resized TO crop_size, rather than resizing the whole (possibly
            # huge, e.g. DBSI's ~1300x1750-scaled) page every sample, which
            # would be far more expensive per step for the same effect.
            jitter = np.random.uniform(0.7, 1.3)
            vstretch = np.random.uniform(0.9, 1.1)  # paper: +-10% independent vertical stretch
            win_w = cs / jitter
            win_h = cs / (jitter * vstretch)
            # Rotation (paper: +-5 degrees) needs extra margin so the
            # rotated crop's corners (which go blank/undefined outside the
            # original content) can be trimmed away before the final resize
            # -- otherwise blank wedges would show up as fake "content".
            angle = np.random.uniform(-5, 5) if np.random.rand() < 0.7 else 0.0
            margin = 1.15 if angle != 0 else 1.0
            raw_w, raw_h = win_w * margin, win_h * margin

            pad_w, pad_h = max(0, int(np.ceil(raw_w - w)) + 1, 0), max(0, int(np.ceil(raw_h - h)) + 1, 0)
            if pad_w > 0 or pad_h > 0:
                padded = Image.new("L", (w + pad_w, h + pad_h), color=255)
                padded.paste(image, (0, 0))
                page_boxes = boxes
                image_for_crop = padded
                w2, h2 = padded.size
            else:
                image_for_crop, page_boxes, w2, h2 = image, boxes, w, h

            x0 = np.random.uniform(0, max(0.0, w2 - raw_w))
            y0 = np.random.uniform(0, max(0.0, h2 - raw_h))
            raw_crop = image_for_crop.crop((x0, y0, x0 + raw_w, y0 + raw_h))

            raw_boxes = page_boxes.copy()
            if len(raw_boxes):
                raw_boxes[:, [0, 2]] -= x0
                raw_boxes[:, [1, 3]] -= y0

            if angle != 0:
                rot_crop = raw_crop.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True,
                                            fillcolor=255)
                rw, rh = rot_crop.size
                if len(raw_boxes):
                    raw_boxes = _rotate_boxes(raw_boxes, angle, raw_w / 2, raw_h / 2, rw / 2, rh / 2)
            else:
                rot_crop = raw_crop
                rw, rh = raw_w, raw_h

            # Center-crop the rotated (now margin-free-needed) region down
            # to the pre-resize window size, trimming rotation's blank
            # corners.
            cx0, cy0 = (rw - win_w) / 2, (rh - win_h) / 2
            trimmed = rot_crop.crop((cx0, cy0, cx0 + win_w, cy0 + win_h))
            if len(raw_boxes):
                raw_boxes[:, [0, 2]] -= cx0
                raw_boxes[:, [1, 3]] -= cy0

            crop = trimmed.resize((cs, cs), Image.Resampling.BICUBIC)
            sx, sy = cs / win_w, cs / win_h
            crop_boxes = raw_boxes
            if len(crop_boxes):
                crop_boxes[:, [0, 2]] *= sx
                crop_boxes[:, [1, 3]] *= sy
                crop_boxes[:, 0] = np.clip(crop_boxes[:, 0], 0, cs)
                crop_boxes[:, 1] = np.clip(crop_boxes[:, 1], 0, cs)
                crop_boxes[:, 2] = np.clip(crop_boxes[:, 2], 0, cs)
                crop_boxes[:, 3] = np.clip(crop_boxes[:, 3], 0, cs)
                cxb = (crop_boxes[:, 0] + crop_boxes[:, 2]) / 2
                cyb = (crop_boxes[:, 1] + crop_boxes[:, 3]) / 2
                keep = (cxb > 0) & (cxb < cs) & (cyb > 0) & (cyb < cs) & \
                       (crop_boxes[:, 2] - crop_boxes[:, 0] > 2) & (crop_boxes[:, 3] - crop_boxes[:, 1] > 2)
                crop_boxes = crop_boxes[keep]

        if self.train and np.random.rand() < 0.5:
            crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if len(crop_boxes):
                new_x0 = cs - crop_boxes[:, 2]
                new_x1 = cs - crop_boxes[:, 0]
                crop_boxes[:, 0], crop_boxes[:, 2] = new_x0, new_x1
                crop_boxes[:, 4] = [mirror_code(int(c)) for c in crop_boxes[:, 4]]

        arr = np.asarray(crop, dtype=np.float32)
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)
        img_tensor = torch.from_numpy(arr).unsqueeze(0).float()
        boxes_tensor = torch.from_numpy(crop_boxes[:, :4]).float() if len(crop_boxes) else torch.zeros((0, 4))
        labels_tensor = torch.from_numpy(crop_boxes[:, 4]).long() if len(crop_boxes) else torch.zeros((0,), dtype=torch.long)
        return img_tensor, boxes_tensor, labels_tensor


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    boxes = [b[1] for b in batch]
    labels = [b[2] for b in batch]
    return images, boxes, labels
