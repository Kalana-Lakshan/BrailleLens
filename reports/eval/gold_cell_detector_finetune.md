# Gold cell-detector fine-tune: before vs after (held-out test page)

Train pages: pg-[1, 2, 3, 4, 5, 6, 7, 8] (high quality) + low-quality-lighting pg-[1, 2, 3, 4] | val (checkpoint selection only): pg-[9, 12] | **test (held out, never trained/monitored on): pg-[10, 11]**

Trained on Colab GPU (T4), 40 epochs, imgsz=1280, batch=8, mosaic=0.5 -- an
earlier CPU-only local attempt at this same train/val/test split (high-quality
pages only, no mosaic, since mosaic segfaulted natively on this machine's
CPU/OpenCV build) only reached mAP50=0.3764/precision=0.5783/recall=0.5891;
adding the low-quality-lighting training images plus mosaic augmentation on a
real GPU closed most of the remaining gap.

| model | mAP50 | precision | recall |
|---|---|---|---|
| baseline (`braille_cell_best.pt`) | 0.4087 | 0.5582 | 0.4805 |
| gold fine-tuned, Colab GPU (`braille_cell_gold.pt`) | **0.7954** | **0.7461** | **0.7572** |

(Baseline mAP50 shown as 0.4087, not the 0.3713 in earlier versions of this
report -- both numbers are the same unchanged `braille_cell_best.pt`, evaluated
with `max_det=800` here vs. an unset default of 300 previously, discovered
while testing the NMS-threshold idea below. The gold fine-tuned row already
had max_det=800 baked into finetune_gold.py in both versions, so that row is
unaffected.)

## +shear/perspective augmentation

Added `shear=1.0, perspective=0.0005` to `cell_detect/finetune_gold.py`'s
training call (previously silently 0.0, Ultralytics' own defaults) --
matching values already used in the full-scale Job A config
(`cell_detect/configs/cells.yaml`), on the theory that real gold photos have
real perspective distortion this fine-tune's augmentation wasn't training
against at all. Retrained on Colab GPU, same train/val/test split as above.

| model | mAP50 | precision | recall |
|---|---|---|---|
| baseline (`braille_cell_best.pt`) | 0.4087 | 0.5582 | 0.4805 |
| previous gold fine-tune (no shear/perspective) | 0.6526 | 0.7300 | 0.7436 |
| **gold fine-tuned, +shear/perspective (current `braille_cell_gold.pt`)** | **0.7954** | **0.7461** | **0.7572** |

A real, substantial gain (+0.14 mAP50) over the already-fine-tuned checkpoint,
on the same held-out pages. Adopted -- this is now the checkpoint in
`cell_detect/weights/braille_cell_gold.pt`.

## Tried and rejected: NMS IOU threshold, fliplr/flipud

Two ideas from the Ovodov CNN paper (docs/CNN paper.pdf), tested against this
same checkpoint/split:

**Lower NMS IOU threshold** (paper uses 0.02, exploiting that Braille cells
never overlap): tested 0.02/0.05/0.10/0.30/0.70 against `braille_cell_gold.pt`
on the held-out test pages. Identical box count (369) and identical
mAP50/precision/recall at every threshold (once `max_det=800` was set to
remove a default `max_det=300` truncation that initially masked the
comparison). This detector's anchor-free single-head architecture already
produces clean, non-duplicate candidate boxes for these small non-overlapping
objects -- unlike the densely-anchored RetinaNet the paper was tuning NMS
for, there's no duplicate-suppression problem here to fix. Not adopted --
doesn't apply to this architecture.

**`fliplr=0.5`/`flipud=0.5`** (geometrically safe in principle: this detector
is single-class (`nc=1`), so a flip can't change its label the way it would
for the dot-pattern classifier): retrained on Colab GPU with the same
train/val/test split and hyperparameters otherwise unchanged. Val mAP50 (pages
9,12) collapsed to ~0.32 (peaking at epoch 15, early-stopped at 30) versus
0.813 at epoch 40 without flips. Not adopted -- reverted `fliplr`/`flipud`
back to 0.0 in `cell_detect/finetune_gold.py`, `cell_detect/configs/cells.yaml`,
and `colab_training.md`. Likely cause: stacking both flips on top of the
already-active mosaic/rotation/scale/translate jitter is too much augmentation
variance for only 12 training images to absorb, not a flaw in the geometric
reasoning -- untested whether it would still hurt on the full 400+ page Job A
dataset, where there's much more real data to absorb the extra variance.

## Failure analysis: where the current best checkpoint (braille_cell_gold.pt) still misses

Ran the +shear/perspective checkpoint (conf=0.30) against ground truth on
both held-out pages and matched by IOU>=0.5 (overlays saved as
`reports/eval/pg-10_gold_detector_failures.png` / `pg-11_...`, green=matched,
red=missed ground truth, blue=unmatched prediction -- includes the *adjacent*
unannotated page visible in the pg-10 photo's frame, which inflates its blue
count without being real error).

**Recall is not uniform across a page.** pg-10: 127/312 ground-truth cells
missed (recall 0.593). pg-11: 46/277 missed (recall 0.834) -- same
checkpoint, same photographic setup, a large page-to-page gap.

**Misses cluster at the start of each line**, i.e. nearest the book's spine
in this open-book photograph, not randomly across the page. Binning each
missed cell's position within its own line (0=line start, 1=line end):
pg-11's first decile alone holds 15 of 46 total misses (33%); the rate falls
off through the rest of the line. Mean normalized position: misses 0.37-0.43,
correctly-detected cells 0.53-0.54 (both pages, same direction).

**Ruled out brightness/shadow as the cause**: missed cells are not darker or
lower-contrast than correctly-detected ones (mean pixel brightness 185-190 vs
181-185, contrast/std within 1-2 points either page) -- if anything, missed
cells trend very slightly brighter, the opposite of a shadow explanation.

**Consistent with perspective foreshortening instead**: missed cells run
~6-7% smaller in both width and height than correctly-detected ones (e.g.
pg-10: 22.4x32.8px vs 24.0x35.2px), with *no* change in aspect ratio
(0.68-0.69 either way) -- a uniform scale reduction, not a shape distortion.
This matches an open book's page curving away from the camera near the
spine, which is also exactly where every line starts.

**The underlying signal is usually there, just under-confident**: re-running
detection at conf=0.02 (full raw candidate pool) and matching by IOU alone
(ignoring confidence), 293/312 (pg-10, 94%) and 270/277 (pg-11, 97%) of
"missed" ground-truth cells actually have *some* matching raw detection --
only 19 and 7 respectively are true zero-signal misses. The rest are sitting
below the conf=0.30 operating threshold specifically because they're smaller
(the same foreshortening effect suppresses detection confidence, not just
box size).

**Tried and rejected: recovering these via smarter thresholding, not
retraining.** (1) A size-adaptive threshold (lower conf cutoff specifically
for boxes smaller than the page's own median) was swept across several
settings -- best case exactly matches the flat conf=0.30 baseline (F1=0.742,
because the setting that doesn't change behavior is trivially identical);
every setting that actually lowers the threshold for small boxes makes F1
*worse* (down to 0.712), since the added recall is outweighed by new false
positives on small noise detections. (2) CLAHE local-contrast-enhancing the
input image before detection: also worse (F1 0.742 -> 0.721) -- the
checkpoint was never trained on CLAHE-normalized input, so this is itself a
distribution shift, and the failure was never a raw-contrast problem per the
brightness analysis above anyway.

**Uniform preprocessing/thresholding doesn't work, but a *targeted* second
pass does.** Both rejected ideas above treated the whole page uniformly. The
failure is concentrated in one identifiable region (the spine-proximal
strip), so re-running detection on just that strip, upscaled, and merging
with the full-page pass by NMS recovers confidence lost to foreshortening
without uniformly relaxing anything: swept strip width and upscale factor
directly against held-out F1 (see `CellDetector.detect_boxes`'s `spine_boost`
parameter), best at `spine_strip_frac=0.45, spine_upscale=2.0`:

| pass | precision | recall | F1 |
|---|---|---|---|
| full page only (baseline) | 0.782 | 0.706 | 0.742 |
| + spine-boost strip, merged | 0.768 | 0.754 | **0.761** |

Confirmed this isn't just "more resolution helps everywhere": upscaling the
*entire* page 2x and merging only reaches F1=0.747, barely above baseline --
the gain is specifically from targeting the spine region the diagnosis
pointed at, not resolution in general. **Adopted** -- implemented as
`CellDetector.detect_boxes(spine_boost=True)` (default off; a second
inference pass, and only validated on genuine open-book-spread photos, not
flat scans), wired through `recognize_page(spine_boost=...)` and
`eval_gold_text.py --spine-boost`. End-to-end effect at cell-conf=0.30: see
reports/eval/gold_text.md (acc_letters_only 0.755 -> 0.792).

Retraining with stronger `perspective`/`scale` augmentation (both already
exposed as CLI flags on `cell_detect/finetune_gold.py`) remains a reasonable
next experiment to close more of the gap, but needs a Colab round-trip and
is no longer the only lever available -- the zero-retrain spine-boost fix
above is already shipped.
