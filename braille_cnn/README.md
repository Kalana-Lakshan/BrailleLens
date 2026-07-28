# braille_cnn

Single-Braille-cell classifier: given one cropped cell image, predict which of the 64 possible
dot patterns (2x3 grid, 6 bits) it is. Letter/language decoding is a separate lookup step
(`labels.py`), not baked into the model. This is the first sub-problem of the larger AI-powered
Braille learning app built on NUS AiSee smart glasses (see `Project Proposal - Group 15.pdf`).

## Setup

```
pip install -r braille_cnn/requirements.txt
```

## Quick start

```bash
# train from scratch on synthetic data
python -m braille_cnn.train

# fine-tune the result on real DBSI scans
python -m braille_cnn.finetune_dbsi

# evaluate a checkpoint on DBSI
python -m braille_cnn.eval_dbsi --checkpoint braille_cnn/checkpoints/braille_cnn_dbsi_finetuned.pt

# try a checkpoint on your own photo of a real page (auto-detects cells, no manual grid needed)
python -m braille_cnn.infer_page --image path/to/photo.jpg --auto --debug-out debug.png
```

## Checkpoints (`checkpoints/`)

| File | Trained on | Good at | Bad at |
|---|---|---|---|
| `braille_cnn_best.pt` | synthetic renders only | synthetic images (~100%) | real photos (51.2% zero-shot on DBSI) |
| `braille_cnn_dbsi_finetuned.pt` | synthetic → fine-tuned on DBSI real scans | DBSI-style scans (98.44%) | synthetic images (~14% — catastrophic forgetting, see Known Issues) |

There is currently **no checkpoint good at both**, and neither is validated on handheld phone
photos (see Known Issues).

## Files

**Core building blocks**
| File | What it does |
|---|---|
| `cnn.py` | `SimpleBrailleCNN` — small 3-conv-block CNN, 64-class output. |
| `labels.py` | `CODE_TO_LETTER` (Grade-1 English, all 64), `CODE_TO_SINHALA` (24 of 64, hand-transcribed from a chart image — see Known Issues below). |
| `render.py` | Procedurally generates synthetic Braille cell images (dot shading, rotation, perspective warp, blur/noise). |
| `dataset.py` | `SyntheticBrailleDataset` — infinite-variety train split, seeded/reproducible test split. |
| `dbsi_dataset.py` | `DBSIDataset` — loads real cells from `data DBSI/` using its per-dot pixel ground truth. Eagerly decodes/crops all cells at construction — don't make this lazy again (see Known Issues). |

**Training**
| File | What it does |
|---|---|
| `train.py` | Trains on synthetic data only → `braille_cnn_best.pt`. |
| `finetune_dbsi.py` | Fine-tunes that checkpoint on DBSI's real train split → `braille_cnn_dbsi_finetuned.pt`. |

**Evaluation / diagnostics**
| File | What it does |
|---|---|
| `eval_dbsi.py` | Evaluates a checkpoint on DBSI (`--sides recto\|verso\|recto,verso`). |
| `eval_perspective.py` | Evaluates checkpoints on synthetic data with/without perspective warp. |
| `inspect_errors.py` | Dumps real crop images for a given (true-class, predicted-class) pair. |
| `show_confusion.py` | Turns a saved `.npy` confusion matrix into a CSV + heatmap PNG. |
| `preview.py` | Renders a sample grid of all 64 synthetic classes, for a quick visual sanity check. |

**Inference on a real, unannotated page photo**
| File | What it does |
|---|---|
| `dot_detect.py` | Finds actual embossed-dot highlights in a raw photo and groups them into per-cell clusters from their measured positions (percentile threshold + connected components). No assumption about page layout. |
| `infer_page.py` | Runs a checkpoint on a real page photo. Two modes: `--auto` (uses `dot_detect.py`, handles skew/variable line length, recommended) or fixed `--rows`/`--cols` grid (only valid for a flat, evenly-spaced scan). Always check `--debug-out` before trusting predictions. |

## Key findings (full detail in `RESULTS.md`)

1. Synthetic-only training hits 100% on synthetic test data, but that alone says nothing about
   real-world readiness.
2. Zero-shot synthetic→real (DBSI) collapses to 51.2% — confirms a real domain gap.
3. Fine-tuning on DBSI's real train split fixes it: 98.44% on the full DBSI test set.
4. Verso is *slightly better* than recto on DBSI (98.65% vs 98.24%) — contradicts the papers'
   framing of verso as harder, at least for this scanner dataset.
5. Perspective/skew was never tested until deliberately added to the renderer — the synthetic-only
   checkpoint drops 99.97% → 87.47% under a realistic homography warp. Not yet retrained to fix this.
6. Tried `infer_page.py --auto` on a real handheld phone photo of a physical Braille book: cell
   detection and cropping work well (verified visually), but **neither checkpoint produces coherent
   letters** on it — a raking-light phone photo doesn't visually match either checkpoint's training
   domain (DBSI flatbed scans, or synthetic Gaussian-bump renders). This is the next real blocker for
   practical (AiSee glasses) use — not yet fixed. Likely fix: fine-tune on real labeled photos from
   the actual target camera/lighting setup, mixed with existing data to avoid repeating finding 6
   below.

## Known issues / gotchas

- **The DBSI-finetuned checkpoint has catastrophically forgotten the synthetic domain** (14% accuracy
  on synthetic images) — fine-tuning was real-data-only, with nothing synthetic mixed back in. Don't
  assume it generalizes to a new visual domain without addressing this.
- **`DBSIDataset` must eagerly decode/cache crops at construction, not lazily per-sample** — lazy
  loading with `shuffle=True` re-decodes full-resolution JPEGs almost every sample (87+ min and not
  finishing). If DBSI loading ever seems slow again, check this first.
- `data DBSI/` (~370MB) is gitignored — download it separately and place it at the repo root before
  running anything that touches `DBSIDataset` (`finetune_dbsi.py`, `eval_dbsi.py`).
- `CODE_TO_SINHALA` is a **hand transcription** from a compressed chart image, only 24/64 codes
  filled in — needs a native speaker's proofread before real TTS use.
- Windows/PowerShell: reading PDF pages needs `pdftoppm` (poppler) on `PATH`.

## Not yet built

- **Finger occlusion handling** — during real reading, the student's finger covers the dots the
  camera needs to see. Current thinking: pre-scan the page once (finger clear) to build a full
  `(row, col) -> character` lookup table, then during the lesson only track the fingertip and detect
  local grid alignment per frame to know which `(row, col)` is touched, rather than re-recognizing
  it live.
- **Angelina dataset** (real handheld phone photos, perspective distortion, page curvature) is
  intentionally deferred to a later, bigger evaluation round. It only has full-page bounding-box
  annotations (not pre-cropped cells like DBSI), so using it will likely mean either a cropping step
  or moving toward an object-detection architecture instead of the current per-cell classifier.
