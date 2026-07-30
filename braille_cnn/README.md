# braille_cnn

Single-Braille-cell classifier: given one cropped cell image, predict which of the 64 possible
dot patterns (2×3 grid, 6 bits) it is. Letter/language decoding is a separate lookup step
(`labels.py`), not baked into the model. This is the first sub-problem of the larger
Braille-to-Sinhala transcription pipeline.

For the full stage-by-stage walkthrough of what happens to an image at inference time (which
file/function handles each step), see [`PIPELINE.md`](PIPELINE.md).

---

## Architecture Overview

```
Camera / Image
     │
     ▼
┌─────────────────────────────────────────────────┐
│  dot_detect.py  — Dot Detection & Cell Clustering│
│  • GaussianBlur difference → percentile threshold│
│  • Non-max suppression to find dot centres       │
│  • Connected-component clustering → cell bboxes  │
└───────────────────┬─────────────────────────────┘
                    │  per-cell 64×64 grayscale crop
                    ▼
┌─────────────────────────────────────────────────┐
│  cnn.py  — SimpleBrailleCNN                     │
│  Input : (B, 1, 64, 64) float32 [0,1]           │
│  Block 1: Conv2d(1→16) + BN + ReLU + MaxPool    │
│  Block 2: Conv2d(16→32) + BN + ReLU + MaxPool   │
│  Block 3: Conv2d(32→64) + BN + ReLU + MaxPool   │
│  Head  : AdaptiveAvgPool(4) → Linear(1024,128)  │
│           → ReLU → Dropout(0.3) → Linear(128,64)│
│  Output: 64-dim logits → softmax → argmax        │
│  → predicted dot-pattern code (0-63)             │
└───────────────────┬─────────────────────────────┘
                    │  sequence of integer codes
                    ▼
┌─────────────────────────────────────────────────┐
│  labels.py  — Two-Pass Sinhala Decoder          │
│                                                 │
│  Pass 1 — Indicator detection:                  │
│    code 60 (dots 3,4,5,6)  = short-vowel ind.  │
│    code 61 (dots 1,3,4,5,6)= long-vowel ind.   │
│    If cell[i] is an indicator → consume         │
│    cell[i+1] as a modifier → produce vowel sign │
│    and attach it to the preceding consonant     │
│    (e.g. ක + ා-sign → කා)                      │
│                                                 │
│  Pass 2 — Single-cell lookup:                   │
│    CODE_TO_SINHALA[code] → Sinhala character    │
│    All 63 codes mapped, zero duplicates         │
│                                                 │
│  Output: Unicode Sinhala string                 │
└─────────────────────────────────────────────────┘
                    │
                    ▼
          Terminal / TTS output
```

**Since this diagram was drawn**, `dot_detect.py`'s first stage gained an optional learned
verification step (`DotPatchCNN`, via `--dot-classifier-checkpoint`) between dot detection and
cell clustering, and a grid-fitting step (`fit_cell_grid`/`cluster_by_grid`) that replaces
distance-based clustering when it succeeds — see Key findings 7-10 below for why, and the
Quick start section for the full command.

### Why this split architecture?

The CNN predicts **which 6-dot pattern** is embossed — a pure visual task with 64 classes.
The language mapping is a **separate deterministic lookup** (`labels.py`), not baked into the
model weights. This means:
- The same trained CNN can decode English *or* Sinhala by switching `lang="en"` / `lang="si"`.
- Fixing a label error requires editing one Python dict, not retraining.
- The two-cell vowel-sign system (indicator + modifier) is handled entirely in the decoder layer
  with no impact on the CNN at all.

---

## Sinhala Decoding — Technical Details

Standard Braille uses a 6-dot cell (dots numbered 1–6, left column top→bottom, right column
top→bottom). The integer code is `sum(1 << (dot-1) for dot in filled_dots)`, giving 64 unique
patterns (0 = empty/space, 1–63 = filled patterns).

### Single-cell mapping (`CODE_TO_SINHALA`)

All 63 non-zero codes map to a unique Sinhala character:

| Code range | Characters |
|---|---|
| 3, 11, 27, 31, 10, 26 | Independent short vowels: අ ආ ඇ ඈ ඉ ඊ |
| 5, 7, 21, 23, 17, 25, 9, 57, 58 | Independent vowels: උ ඌ ඍ ඏ එ ඒ ඔ ඕ ඖ |
| 19, 51, 18, 50, 16, 48 | ka-varga: ක ඛ ග ඝ ඞ ඟ |
| 33, 35, 34, 38, 29 | ca-varga: ච ඡ ජ ඣ ඤ |
| 20, 52, 36, 40, 53 | ṭa-varga: ට ඨ ඩ ඪ ණ |
| 6, 22, 2, 41 | ta-varga: ත ථ ද න |
| 15, 47, 1, 46, 28 | pa-varga: ප ඵ බ භ ම |
| 39, 13, 37, 45 | Semi-vowels: ය ර ල ව |
| 43, 55, 14, 30 | Sibilants/fricatives: ශ ෂ ස හ |
| 44, 42, 49 | Special: ළ ෆ ඹ |
| 32, 4, 54, 56, 24, 59 | Diacritics: ං ඃ ් ඁ ෘ ෲ |

### Two-cell vowel sign system (`TWO_CELL_VOWEL_SIGNS`)

Combining vowel signs require **two consecutive cells** — an indicator cell followed by a
modifier cell:

```
indicator (code 60) + modifier → combining vowel sign (attaches to preceding consonant)
indicator (code 61) + modifier → standalone independent vowel
```

**13 combining signs supported** (short-vowel indicator, code 60):

| Modifier | Vowel sign | Example |
|---|---|---|
| code 1  | ා | කා |
| code 3  | ැ | කැ |
| code 10 | ි | කි |
| code 26 | ී | කී |
| code 5  | ු | කු |
| code 7  | ූ | කූ |
| code 17 | ෙ | කෙ |
| code 25 | ේ | කේ |
| code 9  | ො | කො |
| code 27 | ෝ | කෝ |
| code 21 | ෘ | කෘ |
| code 58 | ෞ | කෞ |
| code 11 | ෑ | කෑ |

**11 independent long vowels** (long-vowel indicator, code 61):
ආ ඇ ඈ ඊ ඓ ඌ ඒ ඕ ඖ ඍ

The `decode_sequence(codes, lang='si')` function applies both passes in order. Combining marks
(Unicode range U+0DCA–U+0DDF) are appended directly to the previous character's string so the
Unicode renderer produces the correct composed glyph.

---

## Setup

```
pip install -r braille_cnn/requirements.txt
```

## Quick start

```bash
# train from scratch on synthetic data
python -m braille_cnn.train

# fine-tune on real DBSI scans
python -m braille_cnn.finetune_dbsi

# evaluate on DBSI
python -m braille_cnn.eval_dbsi --checkpoint braille_cnn/checkpoints/braille_cnn_dbsi_finetuned.pt

# train the dot verification classifier (DBSI ground truth)
python -m braille_cnn.train_dot_classifier

# infer on a real page photo (auto dot-detection, recommended)
python -m braille_cnn.infer_page --image path/to/photo.jpg --auto --lang si --debug-out debug.png

# full validated pipeline on a DBSI-style scan (see Key findings 7-10)
python -m braille_cnn.infer_page --image path/to/scan.jpg --auto --lang si \
    --dot-z-threshold 2.0 --dot-peak-y-offset 5.5 \
    --dot-classifier-checkpoint braille_cnn/checkpoints/dot_classifier_best.pt --debug-out debug.png

# verify the Sinhala label table (run after any edit to labels.py)
python -m braille_cnn.check_labels
```

---

## Checkpoints (`checkpoints/`)

| File | Trained on | Good at | Bad at |
|---|---|---|---|
| `braille_cnn_best.pt` | Synthetic renders only | Synthetic images (~100%) | Real photos (51.2% zero-shot on DBSI) |
| `braille_cnn_dbsi_finetuned.pt` | Synthetic → fine-tuned on DBSI real scans | DBSI-style flatbed scans (98.44%) | Synthetic images (~14% — catastrophic forgetting) |
| `dot_classifier_best.pt` | DBSI dot positions + own detector's false positives + mirrored verso bleed-through | Verifying candidate dots on DBSI-style scans (~99% precision, ~98.5% recall — see below) | Not yet confirmed on handheld phone photos (DBSI-only training data) |

There is currently **no character-classification checkpoint good at both** synthetic and real
images, and neither is validated on handheld phone photos (see Known Issues).

---

## Files

**Core**

| File | What it does |
|---|---|
| `cnn.py` | `SimpleBrailleCNN` — 3-conv-block CNN, 64-class softmax output. |
| `labels.py` | Full Sinhala/English label tables + `decode_sequence()` two-pass decoder. All 63 codes mapped to unique Sinhala characters. Two-cell vowel-sign system implemented. |
| `check_labels.py` | Sanity-check script: prints all 64 code mappings, checks for duplicates, tests `decode_sequence()`. |
| `render.py` | Procedural synthetic Braille cell image generator (Gaussian-bump shading, rotation, perspective warp, blur/noise augmentation). |
| `dataset.py` | `SyntheticBrailleDataset` — infinite-variety train split, seeded reproducible test split. |
| `dbsi_dataset.py` | `DBSIDataset` — loads real cells from `data DBSI/` using per-dot pixel ground truth. Eagerly caches all crops at construction. |

**Training**

| File | What it does |
|---|---|
| `train.py` | Trains on synthetic data → `braille_cnn_best.pt`. Adam lr=1e-3, CrossEntropy, 15 epochs. |
| `finetune_dbsi.py` | Fine-tunes on DBSI real train split → `braille_cnn_dbsi_finetuned.pt`. Adam lr=1e-4, 10 epochs. |

**Evaluation / diagnostics**

| File | What it does |
|---|---|
| `eval_dbsi.py` | Evaluates a checkpoint on DBSI (`--sides recto\|verso\|recto,verso`). |
| `eval_perspective.py` | Benchmarks checkpoints on synthetic data with/without perspective warp. |
| `inspect_errors.py` | Dumps real crop images for a given (true-class, predicted-class) pair. |
| `show_confusion.py` | Turns a saved `.npy` confusion matrix into a CSV + heatmap PNG. |
| `preview.py` | Renders a sample grid of all 64 synthetic classes. |

**Inference on real pages**

| File | What it does |
|---|---|
| `dot_detect.py` | Finds embossed-dot highlights in a raw photo (local z-score, not a global percentile — see Key findings), then either `cluster_by_grid` (preferred, needs `fit_cell_grid` to succeed) or `cluster_into_cells` (distance-based fallback) to group them into per-cell clusters. `fit_cell_grid`/`grid_cell_center` fit the page's regular cell grid from a clean point set and derive each cell's *true* center/pitch from the page-wide model instead of that cell's own (possibly incomplete/asymmetric) points. |
| `dot_classifier.py` | `DotPatchCNN` — tiny binary CNN (32×32 patch → dot/not-dot), a learned replacement for brightness-threshold-only dot verification (mirrors the DSBI paper's own Haar+Adaboost approach). |
| `dot_patch_dataset.py` | Builds `DotPatchCNN`'s training data from DBSI ground truth: positives anchored on the detector's own candidate peak (not the exact ground-truth pixel — matters, see Key findings) + jitter augmentation; negatives from the detector's real false positives plus explicit mirrored-verso bleed-through examples. |
| `train_dot_classifier.py` | Trains `DotPatchCNN` → `dot_classifier_best.pt`. |
| `infer_page.py` | End-to-end inference on a real page photo. `--auto` mode uses `dot_detect.py` (handles skew, variable line length). Fixed `--rows`/`--cols` grid mode for flat scans only. Always check `--debug-out`. Key flags for the full validated pipeline: `--dot-classifier-checkpoint braille_cnn/checkpoints/dot_classifier_best.pt --dot-peak-y-offset 5.5` (DBSI-calibrated, see Key findings — omit `--dot-peak-y-offset` for other capture domains until re-measured there). |

---

## Key findings (full detail in `RESULTS.md`)

1. Synthetic-only training hits 100% on synthetic test data — not a measure of real-world readiness.
2. Zero-shot synthetic→real (DBSI) collapses to **51.2%** — significant domain gap confirmed.
3. Fine-tuning on DBSI real train split fixes it: **98.44%** on full DBSI test set.
4. Verso is slightly better than recto (98.65% vs 98.24%) — contradicts the papers' framing of verso as harder.
5. Perspective/skew: synthetic-only checkpoint drops 99.97% → **87.47%** under a realistic homography warp. Not yet retrained to fix this.
6. Real handheld phone photo: dot detection and cropping work visually, but neither checkpoint produces coherent letters — raking-light phone photos don't match either training domain. This remains the primary blocker for real deployment (see item 11).
7. **A single global brightness threshold can't detect dots reliably across different photos** — settings tuned on one phone photo (~1000 real dots found) collapsed to ~140 (mostly false) on a differently-lit DBSI flatbed scan. Fixed with a *local* z-score instead of a global percentile (`dot_detect.detect_dot_centers`) — adapts per-region, no per-photo retuning.
8. **No single detection threshold gets both good precision and good recall** — a sweep on held-out DBSI pages showed either ~44% precision/~86% recall or the reverse, never both. Fixed by adding `DotPatchCNN`, a learned dot/not-dot classifier (same strategy as the DSBI paper's Haar+Adaboost stage, modernized): **44-53% → ~99% precision at ~90%+ recall** on held-out pages.
9. **Distance-based clustering silently merges adjacent real cells** whenever their combined dot count stays under the 6-dot limit (e.g. a 1-dot cell next to a 2-dot cell) — cost ~150 of 618 true cells their own cluster on a held-out page, invisibly (no merge-flag triggered). Fixed by `cluster_by_grid`: assign each point to its nearest *fitted grid slot* instead of clustering by geometric distance — matched-cell coverage went from 76% to 99% on that page.
10. **This detector's peak lands a real, consistent ~2-6px ABOVE the true dot center** on DBSI scans (confirmed by averaging the z-field over 1000+ ground-truth dot positions on 5 different books — bright above center, cleanly negative below; a genuine asymmetric-lighting signature, not fixable by smarter peak-finding since the underlying signal itself is off-center). A calibrated `peak_y_offset=5.5` correction, combined with findings 8-9, took **end-to-end cell classification from 24.9% to 73.9-97.4%** across three held-out DBSI pages (see `infer_page.py`'s `--dot-peak-y-offset` flag). Confirmed via a controlled test that the *character CNN itself* is not the bottleneck: same cells, exact-vs-approximate crop only, 47.8% vs 96.2%.
11. None of findings 7-10's specific calibrations (`peak_y_offset`, `DotPatchCNN`'s weights) are confirmed to transfer to real handheld phone photos — they were measured/trained on DBSI's specific scanner. Re-measure before assuming they apply to a different capture setup.

---

## Known issues / gotchas

- **DBSI-finetuned checkpoint has catastrophic forgetting** on synthetic images (14%) — fine-tuning was real-data-only with no synthetic mixed back in.
- **`DBSIDataset` must eagerly cache crops at construction** — lazy loading with `shuffle=True` triggers near-continuous full-JPEG decodes (87+ min). If DBSI loading is slow, check this first.
- `data DBSI/` (~370 MB) is gitignored — download separately and place at repo root.
- **`torch.load` calls in all inference/eval scripts are missing `weights_only=True`** — will raise an error in PyTorch 2.6+. Add `weights_only=True` to every `torch.load` call before upgrading PyTorch.
- Windows/PowerShell: `pdftoppm` (poppler) must be on `PATH` for PDF page rendering.

---

## Not yet built

| Feature | Status |
|---|---|
| **Live camera feed** (Branch 2: `feat/camera-capture`) | Done — see `camera_capture/` |
| **Sinhala terminal output** (Branch 3: `feat/live-sinhala-output`) | Done — see `camera_capture/` |
| **DBSI-domain detection + classification pipeline** | Done — 73.9-97.4% end-to-end on held-out DBSI pages (was ~25%), see Key findings 7-10 |
| **Handheld phone camera fine-tuning** (character CNN *and* `DotPatchCNN`/`peak_y_offset`) | Blocked on collecting labeled real-camera crops — see Key finding 11 |
| **Perspective/skew robustness** | Renderer supports it; model not yet retrained with it enabled |
| **Finger occlusion handling** | Proposed: pre-scan page to build `(row,col)→char` table, then track fingertip per frame |
| **Angelina dataset** | Deferred — needs full-page detection or object-detection architecture |
