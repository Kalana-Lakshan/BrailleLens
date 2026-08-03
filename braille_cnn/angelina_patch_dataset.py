"""Builds a labeled dot/not-dot patch dataset from the Angelina dataset, for
extending DotPatchCNN's training beyond DBSI -- mirrors dot_patch_dataset.py,
adapted for the fact that Angelina's ground truth is per-CELL boxes (one box
+ 0-63 code per Braille cell), not per-DOT positions like DBSI's.

Deriving individual dot positions: Angelina's cell boxes are consistently
full-cell-sized regardless of how many dots are active (confirmed: median
~23x36px whether the cell has 1 or 5 active dots), so the 6 canonical dot
slots can be read off the box geometry (left/right edge = the two dot
columns, top/mid/bottom = the three dot rows) combined with which bits are
set in the code. Visual spot-check confirmed this geometric estimate lands
close to but not exactly on the real dot -- same situation DBSI-anchored
positives had, and the same fix applies: use the geometric estimate only as
an anchor to find the nearest actual detected peak from our own detector,
and use THAT (not the raw geometric guess) as the training patch center, so
training matches what the classifier actually sees at inference. Falls back
to the raw geometric position if no candidate peak is nearby.

label=63 (all 6 dots) is this dataset's markout/illegible convention (see
angelina_dataset.py), excluded here too.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .dot_detect import detect_dot_centers

MARKOUT_CODE = 63
NEGATIVE_EXCLUSION_RADIUS = 15.0  # smaller than DBSI's 25px: Angelina cells are physically smaller
ANCHOR_SEARCH_RADIUS = 15.0
JITTER_PX = 3.0


def _read_csv_boxes(csv_path, img_w, img_h):
    boxes = []
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            left, top, right, bottom, label = line.split(";")
            code = int(label)
            if code == MARKOUT_CODE:
                continue
            boxes.append((
                float(left) * img_w, float(top) * img_h,
                float(right) * img_w, float(bottom) * img_h, code,
            ))
    return boxes


def _active_dot_positions(box, code):
    x0, y0, x1, y1 = box
    xs = [x0, x0, x0, x1, x1, x1]
    ys = [y0, (y0 + y1) / 2, y1, y0, (y0 + y1) / 2, y1]
    return [(xs[i], ys[i]) for i in range(6) if code & (1 << i)]


def _crop_patch(image, x, y, patch_size, jitter=0.0, rng=None):
    if jitter > 0 and rng is not None:
        x = x + rng.uniform(-jitter, jitter)
        y = y + rng.uniform(-jitter, jitter)
    half = patch_size // 2
    box = (int(round(x - half)), int(round(y - half)), int(round(x - half)) + patch_size, int(round(y - half)) + patch_size)
    if box[0] < 0 or box[1] < 0 or box[2] > image.width or box[3] > image.height:
        return None
    return np.asarray(image.crop(box), dtype=np.uint8)


def _list_split(books_root, split):
    list_path = Path(books_root) / f"{split}.txt"
    paths = []
    with open(list_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace("\\", "/")
            if line:
                paths.append(Path(books_root) / line)
    return paths


def extract_patches(image_path, csv_path, patch_size=32, neg_random_per_pos=0.5,
                     hard_neg_z_threshold=2.0, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    image = Image.open(image_path).convert("L")
    w, h = image.size
    cell_boxes = _read_csv_boxes(csv_path, w, h)
    if not cell_boxes:
        return np.empty((0, patch_size, patch_size), dtype=np.uint8), np.empty((0,), dtype=np.int64)

    true_dot_positions = []
    for box_code in cell_boxes:
        box, code = box_code[:4], box_code[4]
        true_dot_positions.extend(_active_dot_positions(box, code))
    true_dot_positions = np.array(true_dot_positions)

    patches, labels = [], []
    gray = np.asarray(image, dtype=np.float32)
    candidates = detect_dot_centers(gray, z_threshold=hard_neg_z_threshold)

    if len(candidates) > 0:
        cand_tree = cKDTree(candidates)
        cand_dist, cand_idx = cand_tree.query(true_dot_positions, k=1)
    else:
        cand_dist = np.full(len(true_dot_positions), np.inf)
        cand_idx = None

    for i, (x, y) in enumerate(true_dot_positions):
        if cand_idx is not None and cand_dist[i] <= ANCHOR_SEARCH_RADIUS:
            ax, ay = candidates[cand_idx[i]]
        else:
            ax, ay = x, y
        p = _crop_patch(image, ax, ay, patch_size, jitter=JITTER_PX, rng=rng)
        if p is not None:
            patches.append(p)
            labels.append(1)

    if len(candidates) > 0:
        tree = cKDTree(true_dot_positions)
        dist, _ = tree.query(candidates, k=1)
        hard_neg_pts = candidates[dist > NEGATIVE_EXCLUSION_RADIUS]
        for x, y in hard_neg_pts:
            p = _crop_patch(image, x, y, patch_size, jitter=JITTER_PX, rng=rng)
            if p is not None:
                patches.append(p)
                labels.append(0)

    n_random = int(len(true_dot_positions) * neg_random_per_pos)
    tree = cKDTree(true_dot_positions)
    made = attempts = 0
    while made < n_random and attempts < n_random * 20:
        attempts += 1
        x = rng.uniform(patch_size, w - patch_size)
        y = rng.uniform(patch_size, h - patch_size)
        d, _ = tree.query([x, y], k=1)
        if d < NEGATIVE_EXCLUSION_RADIUS:
            continue
        p = _crop_patch(image, x, y, patch_size)
        if p is not None:
            patches.append(p)
            labels.append(0)
            made += 1

    return np.stack(patches), np.array(labels, dtype=np.int64)


def build_dataset(books_root, split, patch_size=32, seed=0, max_images=None):
    rng = np.random.default_rng(seed)
    image_paths = _list_split(books_root, split)
    if max_images is not None:
        image_paths = image_paths[:max_images]
    all_patches, all_labels = [], []
    for img_path in image_paths:
        csv_path = img_path.parent / (img_path.stem + ".csv")
        if not img_path.exists() or not csv_path.exists():
            continue
        p, l = extract_patches(img_path, csv_path, patch_size=patch_size, rng=rng)
        if len(l) == 0:
            continue
        all_patches.append(p)
        all_labels.append(l)
    if not all_patches:
        return np.empty((0, patch_size, patch_size), dtype=np.uint8), np.empty((0,), dtype=np.int64)
    patches, labels = np.concatenate(all_patches), np.concatenate(all_labels)
    print(f"  Angelina {split}: {len(labels)} patches ({(labels==1).sum()} pos, {(labels==0).sum()} neg) "
          f"from {len(image_paths)} images")
    return patches, labels
