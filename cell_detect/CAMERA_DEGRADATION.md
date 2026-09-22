# Camera-degradation augmentation (AiSee glasses domain gap)

The AiSee glasses camera produces noticeably softer/lower-quality images
than the phone cameras used to shoot every Gold page. This is a synthetic
augmentation that closes that domain gap for training, calibrated against
a real AiSee photo rather than guessed. It lives in
[`gold_finetune_2.py`](gold_finetune_2.py) -- a fork of the original,
still-unmodified `finetune_gold.py`, which stays as the proven fine-tune
for phone-camera-quality input.

## 1. Measuring the real gap

Using a real photo taken with the AiSee glasses camera, sharpness was
measured on the braille-page region via Laplacian variance
(`cv2.Laplacian(gray, cv2.CV_64F).var()` -- a standard focus/blur metric,
higher = crisper edges):

- AiSee glasses photo: **~33**
- Phone-camera gold photos (e.g. pg-1): **~300-350**

That's roughly a 9-10x sharpness drop. The same AiSee photo's background
objects (non-braille) were comparably sharp to the phone shots, pointing
to a near-field focus issue specific to close-up text, not a generic
low-res sensor.

## 2. The transform

Downscale the image, then upscale it back to original size:

```python
def _degrade_image(src_path: Path, dst_path: Path, downscale: float = DEGRADE_DOWNSCALE) -> None:
    img = cv2.imread(str(src_path))
    h, w = img.shape[:2]
    small = cv2.resize(img, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(dst_path), back)
```

Shrinking destroys high-frequency detail proportionally to the downscale
factor; re-expanding via interpolation produces a softer image. This is a
physically-plausible way to simulate a lower-resolution/out-of-focus
capture, rather than inventing arbitrary blur parameters.

## 3. Calibrating the downscale factor

Swept downscale factors against pg-1's own braille-region sharpness,
looking for the one whose degraded sharpness matched the real AiSee
photo's ~33.

**Result: `DEGRADE_DOWNSCALE = 0.39`** (shrink to 39% size, then back up)
-> measured sharpness **33.1**, a near-exact match.

## 4. Cross-checking with a second metric

To avoid curve-fitting a single number, pixel contrast (std deviation) was
also compared:

| | contrast (std) |
|---|---|
| Real AiSee photo | ~9.8 |
| Undegraded original | ~14.3 |
| 0.39-downscaled version | ~11.3 |

The degraded version's contrast landed much closer to the real target
than the original, moving in the same direction as the sharpness match --
two independent measurements agreeing gave confidence the calibration
wasn't a fluke of the one metric being tuned.

## 5. No extra blur needed

Gaussian blur (sigma=1.0) layered on top of the downscale overshot the
target sharpness by 10-30x for any downscale factor tried. The
`INTER_CUBIC` upscale step alone already supplies more smoothing than the
real gap needs, so the final transform is downscale -> upscale only.

## 6. Applying it to the training set

The "degraded" variant reuses the high-quality image's annotations
verbatim -- the transform only softens pixels, it doesn't move any cell.

- **Train** (pages 1-8): pages 1-4 get a `-degraded` copy generated *in
  place of* their real low-quality-lighting photo, keeping the total
  train count at exactly 12 images (the count established as not
  destabilizing this fine-tune -- see
  `reports/eval/gold_cell_detector_finetune.md`'s "Settled: train stays at
  12 images" section). Pages 5-8 are high-quality only, unchanged.
- **Val/test** (pages 9, 12 / 10, 11): each page gets high-quality,
  low-quality-lighting, *and* degraded copies all added alongside each
  other, so performance can be measured across all three quality
  conditions.

## Results (`braille_cell_gold_degraded.pt` vs `braille_cell_gold.pt`)

Evaluated on held-out pages, mAP50 / precision / recall, both checkpoints
against the same current (zero-label-cleaned) annotations:

**Test only (pg-10, 11 -- n=2 images per variant):**

| variant | `braille_cell_gold.pt` | `braille_cell_gold_degraded.pt` |
|---|---|---|
| high-quality | 0.672 / 0.720 / 0.786 | 0.832 / 0.816 / 0.875 |
| low-quality-lighting | 0.737 / 0.765 / 0.839 | 0.896 / 0.845 / 0.927 |
| camera-degraded | 0.674 / 0.717 / 0.782 | 0.800 / 0.775 / 0.845 |

**Val + test (pg-9, 10, 11, 12 -- n=4 images per variant):**

| variant | `braille_cell_gold.pt` | `braille_cell_gold_degraded.pt` |
|---|---|---|
| high-quality | 0.736 / 0.765 / 0.813 | 0.881 / 0.864 / 0.890 |
| low-quality-lighting | 0.783 / 0.752 / 0.863 | 0.907 / 0.862 / 0.921 |
| camera-degraded | 0.732 / 0.744 / 0.806 | 0.854 / 0.850 / 0.836 |

The degraded-trained checkpoint beats the current default on every
metric, on every variant -- not just the degraded ones -- and the margin
held steady (not noise) when the sample doubled from 2 to 4 pages.

**Caveat**: this only proves generalization across our *synthetic*
degradation. It does not confirm the synthetic transform is representative
of real AiSee camera output beyond the single calibration photo -- that
would need at least one more real, annotated AiSee photo to verify
directly.

**Status**: not yet adopted. `braille_cell_gold.pt` remains the active
default; `braille_cell_gold_degraded.pt` is a validated-but-pending
candidate (see `reports/eval/gold_cell_detector_finetune.md` for the
full experiment log).
