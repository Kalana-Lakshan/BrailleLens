# End-to-end inference pipeline

How a photo or scan turns into transcribed text, stage by stage, with the exact files and
functions involved. Covers the `--auto` path (`infer_page.py`'s `run_auto_transcribe`), which is
what both static-image inference and `camera_capture/` (live video) call — this is the pipeline
that has to work with **zero prior knowledge of the page layout** (no ground-truth coordinates),
since that's what a live glasses deployment actually needs.

Entry point: `braille_cnn/infer_page.py`, `run_auto` (prints + saves debug overlay) wraps
`run_auto_transcribe` (the actual pipeline, returns structured data).

---

## Stage 1 — Load image

The input (a file path via `--image`, or a camera frame from `camera_capture/camera.py`) is
converted to a grayscale array. No other preprocessing at this stage.

## Stage 2 — Find candidate dots

**File: `dot_detect.py`, function `detect_dot_centers`**

- Blurs the image lightly (`smooth_sigma`) to kill paper-grain noise, and separately with a wide
  blur (`background_sigma`) to estimate the slow lighting gradient across the page.
- Subtracts the two (`diff`), then computes a **local z-score** of that signal (how many local
  standard deviations above the local neighborhood's own mean each pixel is) — not a single
  global brightness cutoff, so it adapts to photos/scans with different lighting instead of
  needing to be retuned per photo.
- Local maxima of that z-score above `--dot-z-threshold` are candidate dots.
- Strips anything within `border_margin` px of the image edge (the physical page/frame boundary
  reads as a false "highlight," confirmed to be a real, distinct source of false positives).
- Applies `--dot-peak-y-offset` (a fixed pixel correction) if set — corrects a confirmed,
  consistent detection bias specific to DBSI's scanner (peaks land ~2-6px above the true dot
  center, a real asymmetric-lighting signature in the underlying signal). Off (0.0) by default
  since unverified on other capture setups (e.g. a phone camera) — re-measure before assuming it
  transfers.
- Output: a big, generous list of `(x, y)` candidates — tuned to favor *recall* (catch every real
  dot), accepting that real dots will be mixed in with a fair amount of noise at this stage.

## Stage 3 — Verify which candidates are real dots (optional but recommended)

**File: `dot_classifier.py`** (`DotPatchCNN`, a tiny CNN) **+ `infer_page.py`'s `verify_dots`**

- If `--dot-classifier-checkpoint` is given, each candidate gets a 32×32 patch cropped around it
  and passed through this small trained classifier: "is there really a dot here?"
- This is what pushes candidate precision from ~44-53% up to ~99%, cleaning out the noise Stage 2
  deliberately let through (no single brightness threshold gets both good precision *and* recall
  — confirmed empirically by sweeping the threshold on held-out pages).
- Trained by `train_dot_classifier.py`, using labeled data built by `dot_patch_dataset.py` from
  DBSI ground truth: positives anchored on the *detector's own* candidate peak (not the exact
  ground-truth pixel — matters, since that's what the classifier actually sees at inference) with
  jitter augmentation; negatives from the detector's real false positives plus explicit
  mirrored-verso bleed-through examples.
- Checkpoint: `checkpoints/dot_classifier_best.pt`. Trained/validated on DBSI only — not yet
  confirmed to transfer to real handheld photos.

## Stage 4 — Fit the page's cell grid

**File: `dot_detect.py`, function `fit_cell_grid`**

- From the (now clean) point set, estimates the physical layout: intra-cell dot pitch (`dx`,
  `dy`), cell-to-cell pitch (`Px`, `Py`), and each line's horizontal starting position
  (`line_phase_x` — fit **per line**, since indentation/content varies line to line, unlike `Py`
  which is fit once for the whole page since lines stack vertically in a regular way).
- **Column-swap ambiguity fix (`_resolve_column_ambiguity`)**: a cell's two dot columns are `dx`
  apart, so fitting each line's phase independently can lock onto "the right column is offset 0"
  instead of "the left column is offset 0" — indistinguishable from one line's x-positions alone.
  Confirmed on a real DBSI page: this was silently costing ~67 accuracy points end-to-end (30.1% →
  97.4% once fixed). Fixed by chain-resolving each line's phase against its already-resolved
  *neighbor* in line order (not one single global reference — a global-reference version fixed
  DBSI identically but wrecked Angelina, since a handheld photo's phase can legitimately drift
  smoothly line-to-line from perspective skew, and forcing every line to one fixed reference fights
  that real drift). See `README.md` Key finding 16.
- Refined with a least-squares polish pass (`_refine_x_fit`/`_refine_y_fit`, iterated a few times)
  for precision beyond the coarse tolerance-voting search.
- Returns `None` if there aren't enough points or the page doesn't show clean periodicity (e.g.
  real skew) — everything downstream falls back to the pre-grid approach in that case.
- Only trust this on a *clean* point set — the same fit attempted on raw, unverified (noisy)
  detections failed outright; it needs Stage 3's cleanup first.

## Stage 5 — Group points into cells

Two paths, depending on whether Stage 4 succeeded:

- **5a (preferred): `cluster_by_grid`** — assigns each point directly to its nearest fitted grid
  slot. Cells can't accidentally merge just because they're geometrically close, since assignment
  is by grid slot, not distance. Fixes a real bug in 5b: distance-based clustering could silently
  fuse two different real cells into one "valid" (not merged-flagged) cluster whenever their
  combined point count stayed under the 6-dot limit (e.g. a 1-dot cell next to a 2-dot cell) —
  confirmed to cost ~150 of 618 true cells their own cluster on a held-out page.
- **5b (fallback): `cluster_into_cells`** — connected-components clustering by a distance
  threshold (`--link-distance`, or auto-estimated via `estimate_link_distance`). Used when no
  reliable grid could be fit (e.g. real skewed photos, in general).

## Stage 6 — Crop each cell

**File: `infer_page.py`, `_grid_crop_box` / `_cluster_crop_box`**

- For grid-covered cells: crop box uses the grid's own `dx`/`dy` and reproduces whichever crop
  *shape* the loaded character-CNN checkpoint was actually trained on — selected via
  `--crop-shape`. `dbsi` (default) reproduces `DBSIDataset._cell_box`'s exact (asymmetric — width
  and height use different margin multipliers) formula: width `dx*(1+2*margin)`, height
  `2*dy*(1+margin)`. `angelina` reproduces `AngelinaDataset`'s box convention instead: width
  `2*dx*(1+2*margin)`, height `3*dy*(1+2*margin)` (symmetric margin) — confirmed empirically that
  its cell boxes are ~2×dx wide by ~3×dy tall, not DBSI's dx-wide/2×dy-tall shape. Using the wrong
  shape for the loaded checkpoint's domain silently costs ~33 accuracy points end-to-end even with
  an otherwise-correct grid fit (see `README.md` Key finding 12-13) — always match `--crop-shape`
  to `--checkpoint`'s training domain. A cluster's own point centroid is also a bad crop center for
  asymmetric dot patterns (e.g. only the right-column dots active pulls the centroid off the true
  cell center); `grid_cell_center` derives the true center from the page-wide grid model instead.
- For non-grid-covered cells: falls back to a page-wide average cell size (`_estimate_cell_size`)
  and the cluster's own centroid.
- Each crop is resized to 64×64.

## Stage 7 — Classify each cell

**File: `cnn.py`** (`SimpleBrailleCNN`, checkpoint `braille_cnn_dbsi_finetuned.pt`) **+
`normalize.py`'s `normalize_crop`**

- Each crop's brightness/contrast is normalized (mean/std standardized) before being batched
  through the CNN — makes the model robust to absolute brightness differences between crop
  sources (DBSI's tan paper vs. a phone photo of white paper, for example).
- Output: a 0–63 dot-pattern code + softmax confidence per cell.
- This model's own accuracy is *not* the bottleneck when the crop is right: a controlled test on
  identical cells got 47.8% with this pipeline's crop vs. 96.2% with an exact ground-truth crop —
  same model both times.

## Stage 8 — Group into lines, assemble text

**File: `infer_page.py`**

- `_group_into_lines_by_grid` (grid available) or `_group_into_lines` (fallback) groups cells into
  reading-order lines. The grid-based version uses the grid's own line index directly — exact,
  instead of a separate y-gap heuristic that can disagree with it (confirmed: the heuristic
  reported 40 "lines" on a page the grid correctly resolved to 26).
- `_assemble_transcription` → `_decode_line_with_confidence` walks each line left to right,
  inserting word-spaces on x-gaps, applying `--conf-threshold` (low-confidence cells become `_`).
- **File: `labels.py`** — `code_to_label`/`decode_sequence` map each 0–63 code to an English
  letter or Sinhala character, including the two-cell indicator+modifier logic for Sinhala vowel
  signs.

## Output

A structured dict (`num_dots`, `lines`, `sentence`, `boxes`, `preds`, `confidences`, …) — printed
to terminal by `run_auto`/`camera_capture`, optionally rendered as a debug overlay PNG
(`--debug-out`) via `PIL.ImageDraw` (green/red boxes = per-cluster dot extent, merged flag;
blue boxes = actual crop sent to the CNN, with its predicted label).

---

## Two important asymmetries to remember

- **Stages 3 and 4's `peak_y_offset`, and the `dot_classifier_best.pt` weights, are all
  calibrated/trained on DBSI scans specifically** — not yet confirmed to help (or not hurt) real
  phone photos. Re-measure/retrain before assuming they transfer to a different capture setup.
- **Stage 7's character CNN has its own, separate, still-unresolved domain gap on real photos** —
  fixing detection (everything in Stages 2–6) doesn't touch that. The two problems are
  independent; today's work only closed the detection/clustering one.

## No ground truth required at inference

Unlike `DBSIDataset` (used for training/`eval_dbsi.py`), which reads exact ground-truth pixel
coordinates from DBSI's `.txt` annotation files and crops directly at them, **this pipeline never
uses those files**. Stages 2–6 solve "where are the cells" from raw pixels alone, with zero prior
knowledge of the page layout — the actual problem a live deployment (no pre-measured page) needs
solved. The `.txt` files are only ever used *after the fact*, in validation scripts, to check how
well an independently-detected result matches reality.
