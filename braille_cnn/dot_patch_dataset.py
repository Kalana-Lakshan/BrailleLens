"""Builds a labeled dot/not-dot patch dataset from DBSI ground truth, for
training a learned dot classifier (dot_classifier.py) -- see the DSBI paper's
own pipeline (Haar+Adaboost dot detector, F1 0.948-0.970), which this mirrors:
a trained classifier instead of a hand-tuned brightness threshold, since the
threshold sweep in RESULTS.md/chat history proved no single threshold gets
both good precision and good recall.

Positive patches: centered on the DETECTOR'S OWN candidate peak nearest each
real dot, not the exact ground-truth position -- matters a lot: at inference
this classifier only ever sees patches centered on detect_dot_centers'
(imperfectly localized) peaks, never the true pixel-exact position. Training
on perfectly-centered positives taught the first version of this classifier
"dot exactly centered", not "dot near the center" -- confirmed empirically,
it was rejecting ~8% of real dots that the detector found fine but weren't
dead-center in their crop (see chat history). Random jitter augmentation is
added on top for further robustness beyond just the one nearest-candidate
offset.

Negative patches: a mix of
  - "hard" negatives: candidates from our own dot_detect.py that do NOT match
    a real dot -- i.e. the classifier is trained specifically on the actual
    failure modes of the existing detector (border artifacts, misc noise).
  - verso bleed-through negatives: when a paired verso.txt/.jpg exists for
    the page, the verso page's own real dots are mirror-mapped into the
    recto image's coordinate frame and used as explicit negatives -- i.e.
    "here is exactly what real bleed-through looks like, on this exact
    photo" -- rather than hoping hard-negative mining happens to catch a
    few incidentally (confirmed empirically it mostly doesn't: bleed-through
    was only ~5% of one test page's false positives, too rare to be learned
    well from incidental examples alone).
  - random background patches, for general "not a dot" coverage.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .dbsi_dataset import _parse_annotation
from .dot_detect import detect_dot_centers

MATCH_TOL = 6.0
# Wider exclusion radius specifically for picking NEGATIVE training patches.
# DBSI's ground truth is confirmed incomplete (see RESULTS.md / chat history --
# both FM+1's sparse row coverage and FM+11's fine-grained gaps produced
# "false positives" that were actually just unlabeled real dots). Using the
# tight MATCH_TOL here would silently poison training with mislabeled real
# dots marked as negatives. 25px is a bit more than one full cell pitch, so
# it also excludes near-miss/jittery detections right next to a real dot.
NEGATIVE_EXCLUSION_RADIUS = 25.0
JITTER_PX = 3.0


def _true_dot_positions(txt_path):
    parsed = _parse_annotation(txt_path)
    if parsed is None:
        return np.empty((0, 2))
    v, h, cells = parsed
    pts = []
    for row, col, dots in cells:
        x0, x1 = v[(col - 1) * 2], v[(col - 1) * 2 + 1]
        y0, y1, y2 = h[(row - 1) * 3], h[(row - 1) * 3 + 1], h[(row - 1) * 3 + 2]
        xs = [x0, x0, x0, x1, x1, x1]
        ys = [y0, y1, y2, y0, y1, y2]
        for i, d in enumerate(dots):
            if d == 1:
                pts.append((xs[i], ys[i]))
    return np.array(pts) if pts else np.empty((0, 2))


def _crop_patch(image, x, y, patch_size, jitter=0.0, rng=None):
    if jitter > 0 and rng is not None:
        x = x + rng.uniform(-jitter, jitter)
        y = y + rng.uniform(-jitter, jitter)
    half = patch_size // 2
    box = (int(round(x - half)), int(round(y - half)), int(round(x - half)) + patch_size, int(round(y - half)) + patch_size)
    if box[0] < 0 or box[1] < 0 or box[2] > image.width or box[3] > image.height:
        return None
    return np.asarray(image.crop(box), dtype=np.uint8)


def _verso_bleedthrough_positions(txt_path, recto_width):
    """Mirror-maps a paired verso page's real dots into the recto image's
    coordinate frame (x' = width - x), if a verso.txt exists alongside the
    given recto.txt path. Returns empty array if no verso pair exists.
    """
    verso_txt = str(txt_path).replace("+recto.txt", "+verso.txt")
    if not Path(verso_txt).exists() or verso_txt == str(txt_path):
        return np.empty((0, 2))
    verso_pts = _true_dot_positions(verso_txt)
    if len(verso_pts) == 0:
        return verso_pts
    mapped = verso_pts.copy()
    mapped[:, 0] = recto_width - mapped[:, 0]
    return mapped


def extract_patches(image_path, txt_path, patch_size=32, neg_random_per_pos=0.5,
                     hard_neg_z_threshold=2.0, rng=None):
    """Returns (patches uint8 array [N,patch_size,patch_size], labels [N] 1/0)."""
    rng = rng if rng is not None else np.random.default_rng()
    image = Image.open(image_path).convert("L")
    true_pts = _true_dot_positions(txt_path)
    if len(true_pts) == 0:
        return np.empty((0, patch_size, patch_size), dtype=np.uint8), np.empty((0,), dtype=np.int64)

    patches, labels = [], []

    gray = np.asarray(image, dtype=np.float32)
    candidates = detect_dot_centers(gray, z_threshold=hard_neg_z_threshold)

    # positives: anchor each on the nearest DETECTED candidate (matches what
    # the classifier actually sees at inference), falling back to the exact
    # ground-truth position only if the detector missed that dot entirely.
    if len(candidates) > 0:
        cand_tree = cKDTree(candidates)
        cand_dist, cand_idx = cand_tree.query(true_pts, k=1)
    else:
        cand_dist = np.full(len(true_pts), np.inf)
        cand_idx = None
    for i, (x, y) in enumerate(true_pts):
        if cand_idx is not None and cand_dist[i] <= 8.0:
            ax, ay = candidates[cand_idx[i]]
        else:
            ax, ay = x, y
        p = _crop_patch(image, ax, ay, patch_size, jitter=JITTER_PX, rng=rng)
        if p is not None:
            patches.append(p)
            labels.append(1)

    # hard negatives: our own detector's false positives on this exact image
    if len(candidates) > 0:
        tree = cKDTree(true_pts)
        dist, _ = tree.query(candidates, k=1)
        hard_neg_pts = candidates[dist > NEGATIVE_EXCLUSION_RADIUS]
        for x, y in hard_neg_pts:
            p = _crop_patch(image, x, y, patch_size, jitter=JITTER_PX, rng=rng)
            if p is not None:
                patches.append(p)
                labels.append(0)

    # verso bleed-through negatives: real dots from the paired back page,
    # mirrored into this image's coordinate frame
    verso_pts = _verso_bleedthrough_positions(txt_path, image.width)
    if len(verso_pts) > 0:
        tree = cKDTree(true_pts)
        dist, _ = tree.query(verso_pts, k=1)
        verso_neg_pts = verso_pts[dist > NEGATIVE_EXCLUSION_RADIUS]
        for x, y in verso_neg_pts:
            p = _crop_patch(image, x, y, patch_size, jitter=JITTER_PX, rng=rng)
            if p is not None:
                patches.append(p)
                labels.append(0)

    # random background negatives, away from any true dot
    n_random = int(len(true_pts) * neg_random_per_pos)
    tree = cKDTree(true_pts)
    made = 0
    attempts = 0
    while made < n_random and attempts < n_random * 20:
        attempts += 1
        x = rng.uniform(patch_size, image.width - patch_size)
        y = rng.uniform(patch_size, image.height - patch_size)
        d, _ = tree.query([x, y], k=1)
        if d < NEGATIVE_EXCLUSION_RADIUS:
            continue
        p = _crop_patch(image, x, y, patch_size)
        if p is not None:
            patches.append(p)
            labels.append(0)
            made += 1

    return np.stack(patches), np.array(labels, dtype=np.int64)


def build_dataset(page_stems, root, patch_size=32, seed=0):
    """page_stems: list of e.g. 'data DBSI/Fundamentals of Massage/FM+11+recto'."""
    rng = np.random.default_rng(seed)
    all_patches, all_labels = [], []
    for stem in page_stems:
        img_path = f"{stem}.jpg"
        txt_path = f"{stem}.txt"
        p, l = extract_patches(img_path, txt_path, patch_size=patch_size, rng=rng)
        all_patches.append(p)
        all_labels.append(l)
        print(f"  {stem}: {len(l)} patches ({(l==1).sum()} pos, {(l==0).sum()} neg)")
    return np.concatenate(all_patches), np.concatenate(all_labels)
