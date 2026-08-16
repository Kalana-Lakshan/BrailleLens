# braille_detector — single-stage object-detection approach

A from-scratch, self-contained implementation of the approach in Ovodov,
*"Optical Braille Recognition Using Object Detection CNN"* (the paper behind
the Angelina dataset, see `docs/CNN paper.pdf`): one CNN finds and classifies
whole Braille cells directly in a single forward pass, instead of `braille_cnn/`'s
multi-stage pipeline (dot detection → grid-fitting → clustering → per-cell
crop → classify).

Kept entirely separate from `braille_cnn/` on purpose — no shared imports, no
edits to any existing file. This folder can be deleted without touching the
rest of the project.

## Why this exists

This session spent a lot of effort fixing bugs in `braille_cnn/dot_detect.py`'s
grid-fitting (scale mismatches, a column-swap ambiguity, divider-line
contamination) — all bugs that exist *because* that pipeline depends on
fitting an explicit periodic grid model to bridge dot detection and cell
classification. The paper's architecture has no such model: each 16×16-pixel
region of the feature map independently proposes "is a character here, and
if so what/where," so none of those grid-fitting failure modes can occur.
The tradeoff is a heavier model and a very different training setup.

## Files

| File | What it does |
|---|---|
| `boxes.py` | `NUM_CLASSES` (63, codes 1-63), `mirror_code` (label under horizontal flip), `box_iou`, `nms` (IOU=0.02, matching the paper — Braille characters never overlap). |
| `data.py` | `parse_dbsi_txt`/`parse_angelina_csv` — fresh, independent annotation parsers (not imported from `braille_cnn/`). `list_dbsi_pages`/`list_angelina_pages` enumerate real page images + their box/label ground truth. `BraillePageDataset` — random `crop_size`×`crop_size` window per sample + horizontal-flip augmentation (with `mirror_code`), from a page pre-rescaled to a common ~16px dot pitch (see below). |
| `model.py` | `BrailleDetector` — small stride-16 CNN backbone (4 stride-2 blocks) + a box-regression head + a class head, one fixed-size anchor per 16×16 feature-map cell (paper's RetinaNet simplification: one scale, one anchor, since every character is nearly the same size). |
| `loss.py` | FocalLoss (classification) + smooth-L1 (box regression), matching `L = L_loc + λ·L_cls`. Each ground-truth box is assigned to the single feature-map cell containing its center — no IOU-based anchor matching needed since there's only one anchor per cell. |
| `train.py` | Trains on combined DBSI + Angelina page data. |
| `infer.py` | Runs a trained checkpoint on a full page in one (optionally tiled) forward pass, NMS, and evaluates against ground truth with the same matched-center methodology used for `braille_cnn/`'s numbers this session, so results are directly comparable. |

## Key design choices / scoped-down vs. the paper

- **Per-dataset pixel-scale normalization**: DBSI (~200dpi scans) and Angelina
  (handheld photos) have very different absolute dot pitch in pixels
  (confirmed this session: DBSI dx≈21px, Angelina dx≈13.5px). Both are
  rescaled by a fixed, precomputed factor toward a common ~16px target pitch
  before cropping (`DBSI_SCALE`/`ANGELINA_SCALE` in `data.py`), so one fixed
  anchor size works for the whole training set. This is a coarse, hardcoded
  approximation, not a per-image auto-measurement — deliberately avoiding any
  dependency on `braille_cnn/`'s grid-fitting machinery, which is exactly what
  this approach is meant to not need.
- **Augmentation**: only random crop + horizontal flip (with label mirroring).
  The paper also does continuous random resize/stretch/rotation; not
  implemented here yet — a natural next step if this baseline is worth
  building on.
- **Anchor size**: one fixed `(ANCHOR_W, ANCHOR_H)` in `model.py`, a rough
  compromise between DBSI's and Angelina's differing box *conventions* (DBSI's
  ground-truth box is tight to just the active dots; Angelina's spans the
  full nominal cell regardless of which dots are active) — not
  precision-tuned.
- **DBSI blank pages** (empty `.txt` annotation files, e.g. chapter title
  pages) are included as zero-box (all-background) training samples rather
  than filtered out.
- **Not included**: the paper's 44 negative (non-Braille) training images,
  and its λ_cls annealing schedule (1→1000 over 500 epochs) — `train.py`
  just uses one fixed `--lambda-cls`.

## Usage

```
python -m braille_detector.train --steps 3000 --batch-size 4
python -m braille_detector.infer --checkpoint braille_detector/checkpoints/detector.pt \
    --image "data DBSI/Fundamentals of Massage/FM+17+recto.jpg" \
    --annotation "data DBSI/Fundamentals of Massage/FM+17+recto.txt" --source dbsi
```

## Status

First working proof-of-concept, not yet a tuned/converged model — see chat
history for the actual accuracy numbers from the first training run and how
they compare to `braille_cnn/`'s grid-fitting pipeline (97.4% DBSI / 43.2%
Angelina, end-to-end, on the same two held-out test pages).
