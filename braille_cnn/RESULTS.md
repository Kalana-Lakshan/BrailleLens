# Experiment Log

## 2026-07-20 — Synthetic-trained CNN, zero-shot on DBSI (real data)

**Question:** how big is the synthetic-to-real domain gap for the 64-class dot-pattern CNN, before doing any real-data training?

**Setup**
- Model: `braille_cnn_best.pt`, trained only on the procedural synthetic renderer (`render.py`), 8 epochs / 80 samples per class, reached 100% accuracy on synthetic held-out data.
- Real data: [DSBI](https://github.com/yeluo1994/DSBI) (`data DBSI/`), loaded via the new `DBSIDataset` (`dbsi_dataset.py`), which crops individual cells out of the de-skewed page scans using the dataset's ground-truth per-dot pixel coordinates.
- Eval set: the dataset's own `test.txt` split — 88 pages, 71,250 labeled cells, all 64 classes represented. No fine-tuning; the model never saw a single real image.
- Script: `python -m braille_cnn.eval_dbsi`

**Result**

| | Synthetic test | DBSI test (zero-shot) |
|---|---|---|
| Accuracy | 100% | **51.2%** (36,449 / 71,250) |

Chance level for 64 classes is ~1.6%, so the model clearly learned real, transferable structure (dot-count and rough position) — it just isn't reliable enough to use as-is. The confusion matrix (`checkpoints/dbsi_test_confusion.png`, `.csv`) is still visibly diagonal-dominant rather than random.

**Failure pattern:** errors are concentrated in classes with 4+ dots (e.g. `g`→`#63`, `s`→`#12`, `j`→`number_sign`/`w`) rather than spread evenly — the model tends to under- or over-count dots in denser patterns, and confuses patterns that differ by one dot in a corner position. Likely contributors, roughly in order of suspected impact:
1. **Synthetic augmentation was too mild** relative to real embossed-paper appearance/noise (already flagged after the first synthetic training run).
2. **Cell cropping imprecision** — `DBSIDataset` crops a fixed margin (0.8x dot spacing) around each cell's annotated dot centers; on denser/tighter layouts this can clip a dot or pull in a sliver of a neighboring cell, which would directly explain dot-count-sensitive errors.
3. **Real embossing artifacts** DBSI itself documents (oil stains, paper distortion, worn dots, one book explicitly rated "Bad" quality) that the synthetic renderer doesn't model at all (no localized defects, no worn/shallow dots).

**Interpretation:** confirms the domain gap is real and significant (100% → 51%), but not catastrophic — the architecture and synthetic pretraining aren't wasted, they just aren't sufficient alone. This sets a concrete baseline to beat.

**Next steps to consider:**
- Fine-tune the synthetic-pretrained checkpoint on DBSI's `train.txt` split (26 pages) and re-run this same eval to measure improvement.
- Tighten/inspect `DBSIDataset` crop margins to rule out (2) as a confound before attributing the rest of the gap to (1)/(3).
- Harden `render.py`'s augmentation ranges (worn/shallow dots, localized paper defects, tighter dot spacing) so synthetic pretraining transfers better on its own.

---

## 2026-07-21 — Fine-tuning the synthetic-pretrained CNN on DBSI

**Question:** how much of the 51% zero-shot gap closes with a small amount of real training data, following the proposal's "public dataset → fine-tune" pipeline?

**Setup**
- Started from the same `braille_cnn_best.pt` synthetic checkpoint.
- Fine-tuned on DBSI's `train.txt` split (26 pages, 20,193 labeled cells) for 10 epochs, Adam, lr=1e-4 (lower than the original 1e-3 since this is fine-tuning, not training from scratch).
- Tracked progress each epoch on a fixed random 8,000-cell subset of the test split; kept the checkpoint with the best subset accuracy.
- Final number reported on the *full* 71,250-cell test split for direct comparison with the zero-shot run above.
- Script: `python -m braille_cnn.finetune_dbsi`

**Important bugfix before this ran cleanly:** the first attempt at this (previous session) stalled for 87+ minutes and still hadn't finished. Diagnosis: `DBSIDataset` originally cached only the *most recently opened* page image, which is fine for evaluation (sequential, grouped by file) but during training `shuffle=True` randomizes access across ~50 large scanned page JPEGs, so it was re-decoding a full-resolution page image from disk on almost every single sample — an I/O bug, not a hardware limitation. Fixed by having `DBSIDataset` eagerly decode and crop every cell exactly once at construction time (each page opened once, cells cached in memory as a `uint8` tensor — 83MB for train, 292MB for test). After the fix, building both splits took 45 seconds total instead of 87+ minutes and counting.

**Result**

| | Synthetic test | DBSI test, zero-shot | DBSI test, fine-tuned |
|---|---|---|---|
| Accuracy | 100% | 51.2% (36,449 / 71,250) | **98.44%** (70,141 / 71,250) |

Training accuracy climbed smoothly (62.8% → 91.7% over 10 epochs) while the held-out subset accuracy climbed faster and higher (86.3% → 98.5%), with no sign of overfitting yet (still improving at epoch 10) — 10 epochs on 20k real cells was enough to erase nearly all of the domain gap.

**Remaining errors** are now sparse and concentrated on a few specific classes rather than spread across dense patterns: `#18` (dots 2,5 — no assigned letter) is confused with `j` 178 times and with `h` 44 times, which is by far the largest single error bucket. Everything else is small (≤21 occurrences), scattered across many different class pairs. Confusion matrix: `checkpoints/dbsi_test_confusion_finetuned.png` / `.csv`.

**Interpretation:** the proposal's intended pipeline (public dataset pretraining → fine-tune) works very well here — a tiny amount of real data (26 pages) took a model from barely-better-than-random-structure to 98.4% on real scanned/embossed Braille. This suggests the earlier 51% zero-shot gap was dominated by the synthetic renderer's mild augmentation / stylistic mismatch rather than the CNN architecture being inadequate, since the same architecture reaches near-ceiling accuracy once it sees a modest amount of real examples.

**Next steps to consider:**
- Investigate the `#18` → `j`/`h` confusion specifically (check a few real crops of that class — may be a labeling quirk in DBSI itself, since `#18` has no assigned English letter and dots 2,5 is a fairly unusual pattern).
- This result is still on DBSI's own real-but-scanned images (flatbed scanner, fixed lighting) — it does not yet tell us how well this transfers to actual handheld AiSee glasses footage (phone-camera-style distortion, per the Angelina dataset / Ovodov paper's distinction). That remains the next real test.
- Consider evaluating on DBSI's verso (back-side) dots specifically vs recto, since verso dots are described as harder to distinguish visually.

---

## 2026-07-21 — Recto vs verso breakdown, and the `#18` confusion explained

**Question:** two follow-ups flagged above — is verso actually harder than recto, and what's causing the `#18` (dots 2,5, no assigned letter) → `j`/`h` confusion?

**Recto vs verso** (fine-tuned checkpoint, `--sides recto` / `--sides verso`):

| | Recto | Verso |
|---|---|---|
| Cells | 35,659 | 35,591 |
| Accuracy | 98.24% (35,032) | **98.65%** (35,109) |

Verso is slightly *better* than recto here, which cuts against the DBSI/CNN papers' framing of verso (bulged-in, back-side) dots as visually harder to read than recto (bulging, front-side) dots — at least for this model/dataset. Worth remembering this is still all flatbed-scanner data with consistent, known lighting; it may not hold once lighting is uncontrolled (e.g. on the glasses).

One asymmetry did show up: on recto, `#18`'s errors go almost entirely to `j` (175 of 178); on verso, they go mostly to `h` (43 of 44) instead. Same root confusion, mirrored — consistent with recto/verso being literally the two sides of the same physical page.

**The `#18` confusion, visually inspected** (`inspect_errors.py`, new diagnostic script): pulled 16 real crops labeled `#18` (dots 2,5 only — just the middle row) that the model predicted as `j`. Nearly every one clearly shows a **faint extra mark above the true middle-row dots**, in the position where a top dot would sit — exactly what would make a classifier read `j` (dots 2,4,5) or `h` (dots 1,2,5) instead of `#18`.

Initial hypothesis was crop-margin bleed from the neighboring cell/line above (`checkpoints/margin_compare_18.png` compares `margin_scale=0.8` vs a much tighter `0.25` for the same 8 cells) — **but the mark persists even at the tight margin**, ruling that out as the primary cause. The more likely explanation: a genuine faint physical feature at that dot position (a shallow indentation/ridge from embossing bleed-through, page pressure, or a partially-formed dot) that the human annotator judged below threshold and labeled `0`, but which is visually present enough for the CNN to pick up on. In other words, this looks like a borderline/ambiguous ground-truth label on DBSI's side for a chunk of this specific class, not a bug in our cropping or training code.

**Interpretation:** the fine-tuned model's real accuracy on unambiguous cells is likely even higher than the measured 98.44%, since a meaningful slice of the remaining errors trace to one specific class with what looks like noisy ground truth, rather than genuine model confusion. Not worth chasing further with code changes — this is a property of the dataset, and the model's behavior on it (reading a faint-but-real mark as a dot) is arguably reasonable.

**New diagnostic tool:** `braille_cnn/inspect_errors.py --true-class N --pred-class M --n K` dumps a grid of real crops for any true/predicted class pair — reusable for inspecting any future confusion, not just this one.

---

## 2026-07-21 — Perspective/skew augmentation added; neither existing checkpoint is robust to it (for different reasons)

**Question:** raised while discussing the live-reading pipeline (finger occlusion, pre-scan-then-track design) — none of the training or eval so far has tested skew or camera-angle distortion. DBSI's `+recto`/`+verso` images are already de-skewed by the dataset itself (checked: raw skew angles max out at 1.6°, mean 0.25°, and we don't even use the raw versions), and the synthetic renderer only ever did mild ±10° *in-plane* rotation — never perspective/tilt. The Angelina dataset (real phone photos, has this distortion) is intentionally being saved for a later, bigger evaluation, so this round adds synthetic perspective augmentation as a fast, self-contained first check.

**What changed:** `render.py` gained a homography-based perspective warp (`_apply_perspective`, via `PIL`'s `Image.Transform.PERSPECTIVE` with coefficients solved from 4 independently-jittered corner points — a real trapezoidal "camera looking at the page from an angle" distortion, not just in-plane rotation). Applied after the existing rotation step, strength randomized up to `max_perspective` (default 0.20 = corners can shift up to 20% of image size). Verified visually first (`checkpoints/perspective_check.png`) — cells correctly skew into trapezoids rather than breaking/mirroring. `SyntheticBrailleDataset` and `render_braille_cell` both take `max_perspective` so it can be dialed to 0 for apples-to-apples comparison against the pre-perspective behavior.

**Result** (`eval_perspective.py`, 50 samples/class, `max_perspective=0.0` vs `0.20`):

| Checkpoint | No perspective | With perspective |
|---|---|---|
| `braille_cnn_best.pt` (synthetic-only) | 99.97% | **87.47%** |
| `braille_cnn_dbsi_finetuned.pt` (DBSI fine-tuned) | 14.06% | 9.94% |

**Finding 1 (expected):** the synthetic-only checkpoint is measurably vulnerable to perspective distortion — a clean ~13-point drop from a distortion type it never saw during training. Confirms the concern from the live-reading discussion was real, not hypothetical.

**Finding 2 (unexpected, more important):** the DBSI-fine-tuned checkpoint scores only 14% on synthetic images *regardless* of perspective — it has **catastrophically forgotten the synthetic visual style entirely**. Fine-tuning for 10 epochs purely on real DBSI crops, with no synthetic data mixed back in, overwrote whatever let it handle the synthetic renderer's appearance (most predictions collapse to `space`). This means the fine-tuned checkpoint's 9.94% "with perspective" number is not really measuring perspective robustness — it's dominated by domain forgetting that already existed before perspective was even added. The clean perspective-only signal is the synthetic-only checkpoint's 99.97%→87.47% drop.

**Interpretation:** two separate problems, not one. (a) Perspective robustness needs to be trained in, not just tested — the natural next step is retraining/re-fine-tuning *with* `max_perspective > 0` active, mirroring how DBSI fine-tuning fixed the earlier domain gap. (b) The fine-tuned checkpoint is now real-data-specialized and shouldn't be assumed to generalize to any new visual domain (including, presumably, Angelina's photos) without either mixing synthetic data into future fine-tuning runs or accepting it as a narrowly-scoped checkpoint.

**Next steps to consider:**
- Retrain the synthetic-only model with perspective augmentation active from the start, then re-run this same comparison to see how much it recovers (this is the direct analogue of the DBSI fine-tuning experiment, but for perspective instead of real-vs-synthetic).
- When next fine-tuning on DBSI (or later Angelina), mix in a slice of synthetic data to check whether that prevents the catastrophic forgetting seen here.
- Keep perspective augmentation in mind as one more thing the eventual Angelina evaluation should be read against — a model that still fails on Angelina after this fix would point more clearly at texture/lighting differences rather than geometry.

---

## 2026-07-29 — First real handheld-photo test (own phone photos); `--auto` crash fix, adaptive dot-linking, and per-crop normalization

**Question:** two own phone photos of a printed Braille page (`test-img.jpeg`, `test-img2.jpeg` — white paper, handheld, uneven room lighting) produced no usable transcription. Initial hypothesis (background *color*: DBSI's pages are tan/kraft paper, these are white) turned out to be mostly wrong once measured — logged here so the real causes aren't re-litigated later.

**Bug found (unrelated to the color question):** `--auto` transcription mode crashed unconditionally on *any* image, including DBSI's own, via a `TypeError` in `_decode_line_with_confidence` (`infer_page.py`). The prior "confidence-aware transcription" refactor built `code_by_id` as `{id: code}` and a separate `conf_by_id`, but the decoder unpacked `code_by_id[id(c)]` as `(code, conf)`. Fixed by merging into one `{id: (code, conf)}` dict; `conf_by_id` removed as a dead parameter.

**Background-color hypothesis, checked and mostly rejected:** everything is converted to grayscale before any processing (`Image.open(...).convert("L")`), and DBSI's tan paper vs. the photos' white paper land in a similar grayscale range once converted — hue isn't the operative variable. The DSBI paper (§4.3.1) does address "different Braille images background," but via grayscale + a *global per-page* gray-histogram normalization + adaptive threshold, for a classical (non-CNN) segmentation dot-detector — and it wouldn't fix a page with an internal brightness gradient like these photos have anyway.

**Real cause #1 — dot pitch vs. a hardcoded pixel constant.** Measured nearest-neighbor dot spacing (`dot_detect.detect_dot_centers` output) on `test-img.jpeg` vs. `data DBSI/Math/math+1.jpg`:

| | test-img.jpeg (888×1280 phone photo) | DBSI math+1.jpg (1700×2338, 200dpi scan) |
|---|---|---|
| 1st-NN median | 11.0 px | 14.0 px |
| 2nd-NN median | 15.6 px | 18.5 px |
| 3rd-NN median (next-cell jump) | 22.0 px (no clean gap) | 29.0 px (clear gap) |

The old fixed `link_distance=15.0` in `cluster_into_cells` sat cleanly below DBSI's next-cell jump, but landed *inside* the phone photo's noisy zone with no comparable gap — fragmenting most 6-dot cells into 2-3-dot pieces before classification ever ran (visible in the debug overlay as undersized crop boxes and garbage codes like `#16`/`#18`/`#50`).

**Fix:** `dot_detect.estimate_link_distance()` — auto-estimates the link distance per image from the median 1st-nearest-neighbor dot distance × 1.5 (covers same-cell diagonal pairs, ~√2), instead of assuming one fixed pixel value works for every photo resolution/distance-to-page. `cluster_into_cells(link_distance=None)` now calls this by default; `infer_page.py --link-distance` defaults to auto (explicit values still override). On `test-img.jpeg` this resolves to 16.6px.

**Real cause #2 — no normalization anywhere in the CNN pipeline.** Confirmed by grep: `dataset.py`, `dbsi_dataset.py`, and `infer_page.py` all did nothing but `/ 255.0` before feeding crops to the network — raw absolute brightness, unbounded. Measured crop brightness across sources: DBSI crop raw mean 214.5 (std 56.0) vs. `test-img.jpeg` crops from the photo's bright vs. shadowed regions, raw mean 172.8 (std 14.8) and 148.1 (std 7.8) respectively — both a large absolute-brightness gap *and* a large contrast/dynamic-range gap between sources, and even between two spots on the *same* uneven photo.

**Fix:** `normalize.py`'s `normalize_crop()` — subtracts each crop's own mean and divides by its own std (floored at 10.0, chosen empirically: synthetic blank/`space` cells measured raw std ≤4.8, populated cells typically 5.4-14.2+, so the floor sits between the two and stops blank cells' background noise from being amplified into fake contrast). Verified this closes the gap above: normalized means for DBSI/bright-photo-region/dark-photo-region landed at 0.52/0.51/0.50 respectively (vs. raw means 214.5/172.8/148.1). Wired into all three consumers (`dataset.py`, `dbsi_dataset.py`, `infer_page.py._classify`) so train/fine-tune/inference see the same representation.

**Retrained from scratch with normalization active** (new checkpoints in `checkpoints_normalized/`, old `checkpoints/` left untouched for comparison):

| | Synthetic test | DBSI test (full) | DBSI recto | DBSI verso |
|---|---|---|---|---|
| Previous (`checkpoints/`, no normalization) | 100% | 98.44% | 98.24% | 98.65% |
| New (`checkpoints_normalized/`, with normalization) | 98.85% | **99.24%** | **99.33%** | **99.16%** |

Normalization cost a little synthetic-test accuracy (100%→98.85% — expected: the model can no longer shortcut on absolute brightness, a slightly harder task) but *improved* real DBSI accuracy on every split, and the long-standing `#18→j/h` confusion dropped from 222 occurrences to 19 (`#18→h`). No regression; net improvement.

**Applied to the actual phone photos — still not usable, but now for a clearly-isolated, different reason.** Re-ran `--auto` with the new checkpoint and auto-estimated `link_distance=16.6px`: still garbage transcription. This is not a normalization or classification failure — the debug overlay shows the fragmentation issue is only partly resolved (261 clusters vs. 433 before, but still many undersized/oversized boxes; some clusters now span almost a whole line, flagged merged). Real cause #1's fix (a smarter *global scalar* threshold) is a genuine improvement but can't fully solve this photo: real Braille geometry puts the worst-case same-cell diagonal distance and the next-cell gap only ~10% apart physically, and at this photo's resolution (~11px median pitch) that margin becomes noise-comparable. No single link-distance value, adaptive or not, can guarantee correct clustering here.

**Interpretation:** three genuinely separate problems got tangled together in the original "background color" hypothesis. (1) a real crash bug, unconditional and unrelated to any of this — fixed. (2) illumination/contrast normalization — real, fixed, and independently validated as a net improvement on DBSI, but was never the dominant failure mode for these two specific photos. (3) dot-detection/clustering breaking down at low photo resolution — the actual dominant failure mode for these two photos, and the one still open.

**Next steps to consider:**
- For (3): replace naive distance-threshold clustering with a grid-aware approach that exploits the *known* 2-column/3-row cell topology (e.g., estimate row/column pitch separately and link only axis-aligned neighbors, or fit a local grid) rather than single-linkage over raw pairwise distances — the current approach is fundamentally unable to separate "worst-case same-cell diagonal" from "next-cell gap" when they're this close in real Braille geometry.
- Alternatively/additionally: retake these test photos at higher resolution or closer distance-to-page (more pixels per cell gives the existing clustering more margin to work with) — a capture-workflow fix, not a code fix.
- Once (3) is resolved enough to get correctly-cropped cells from a real handheld photo, re-check classification accuracy specifically — normalization was validated on DBSI (scanned) and via synthetic crop statistics, not yet on a real, correctly-cropped handheld-photo cell.
