# BrailleLens plan status

Checklist against the data-science-lifecycle plan
(`braille_learning_system_build`). Status is **as of 16 Aug 2026**, on
branch `Kalana/improvement-preprocessing`.

Legend: **done** = in the repo and already run locally where it can be.
**code ready, waiting on GPU / labels** = scripts exist but cannot finish
on this CPU-only machine, or Gold pages are still unlabelled.
**not started** = still to do.

Mobile folders `braille_app_pipeline/` and `braille_lens_flutter/` stay
untouched on purpose.

GPU training for another person: **[`colab_training.md`](colab_training.md)**
(Job A = Stage 4a, Job B = Stage 4b — separate Colab runtimes).

---

## Stage 0 — Unblock the machine

| Item | Status |
|---|---|
| Angelina path defaults (`data Angelina/books`) | **done** |
| Duplicate checkpoint `.pt` files removed from git | **done** |
| Isolated `finger_cell_track/.venv` (MediaPipe + torch + ultralytics) | **done** |
| Root env MediaPipe (TensorFlow protobuf clash) | **left as-is** — use the finger venv for Stage 5 |

---

## Stage 1 — Collection / Gold labels

| Item | Status |
|---|---|
| DSBI under `data DBSI/` (gitignored) | **done** (on disk) |
| Angelina under `data Angelina/books` (gitignored) | **done** (on disk) |
| Gold photos: 12 pages × High/Low lighting | **done** (on disk) |
| Stage 1b: LabelMe JSON on the **12 High** pages (dot-string labels like `1345`) | **not started** |
| Homography-transfer High boxes onto Low twins (`data_pipeline.transfer_gold_labels`) | **code ready**, 0 labels so it is a no-op |

Gold High and Low are the same 12 physical pages. Label High only; split by
`page_group` (pages 1–8 train, 9–10 val, 11–12 test) so twins cannot leak.

---

## Stage 2 — Data pipeline

| Item | Status |
|---|---|
| 2a `data_pipeline/integrate.py` | **done** |
| 2b `data_pipeline/clean.py` + `reports/cleaning_log.md` | **done** |
| 2c `data_pipeline/reduce.py` → `crops_{train,val,test}.npz` | **done** (gitignored on disk) |
| 2d `transform.py` / `CropDataset` (normalize + augment at load) | **done** |
| Rebalanced split (recommended for training) | **done** — train 120,809 / val 21,095 / test 15,655 cells, no page_group leakage |
| Official split mode (for published DSBI comparability) | **code ready** — use `--split-mode official` if needed |

Handoff file: `data_pipeline/manifests/manifest_clean.csv` (gitignored).

---

## Stage 3 — EDA

| Item | Status |
|---|---|
| `data_pipeline/analyze.py` | **done** |
| Plots in `reports/eda/` | **done** |

---

## Stage 4 — Models

Two-stage recogniser: **YOLO finds cell boxes**, then **SimpleBrailleCNN
classifies each 64×64 crop** (64 classes, codes 0–63). Dot detection +
grid fitting is fallback / baseline only.

| Item | Status |
|---|---|
| 4a `prepare_cell_dataset.py` + `detect_cells.py` + `train_detector.py` | **done** (code) |
| 4a YOLO dataset `cell_detect/datasets/braille_cells/` | **done** on disk (332 / 56 / 44 pages) |
| 4a **GPU train ~80 epochs** | **not started** — see Job A in `colab_training.md`. Local 1-epoch CPU smoke: mAP50 0.125 only. Do not use that `best.pt` as `braille_cell_best.pt`. |
| 4b `braille_cnn/train_classifier.py` (mixed domain + class weights) | **done** (code) |
| 4b crop archives on disk | **done** |
| 4b **GPU train ~20 epochs `--balance-domains`** | **not started** — see Job B in `colab_training.md`. Local `--smoke-test` is 256 crops, not a shippable model. |
| 4c `braille_cnn/finetune_gold.py` | **code ready** — exits cleanly until Gold JSONs exist |
| 4d `braille_cnn/recognize.py` → `recognize_page(image, backend="cells"|"dots")` | **done** |

Until 4a / 4b weights exist, live scan falls through to dots / DotNeuralNet.

---

## Stage 5 — Live fingertip app

| Item | Status |
|---|---|
| 5.1 `autoscan.py` `PageWatcher` (hands-free page capture) | **done** |
| 5.2 prescan uses `recognize_page()`; DotNeuralNet only as fallback | **done** |
| 5.2 auto-scan **on by default** (`--scan-backend auto\|cells\|dots\|dnn`) | **done** |
| 5.3 `MediaPipeTip` + `SkinContourTip` fallback (`--tip-backend auto`) | **done** |
| 5.3 `eval_tip.py` on real oCam footage | **code ready, not run yet** |
| 5.4 `live_app.py` dwell → print code + character (`--lang si` default) | **done** |

Run Stage 5 from `finger_cell_track/.venv`, not the broken root MediaPipe.

---

## Stage 6 — Evaluation

| Item | Status |
|---|---|
| Scripts: `eval_angelina`, `eval_gold`, `eval_end_to_end`, `evaluate_detector` | **done** (they write `reports/eval/*.md`) |
| Real numbers after GPU weights + Gold labels | **not started** |

---

## What the other person should do now

1. **Job A** — Colab GPU, Stage 4a cell YOLO. Send back `braille_cell_best.pt`.
2. **Job B** — Colab GPU, Stage 4b CNN. Send back `braille_cnn_mixed.pt`.

Full copy-paste: [`colab_training.md`](colab_training.md).

## What stays with the owner after that

1. Drop the two `.pt` files into the paths listed in `colab_training.md`.
2. Stage 1b Gold labelling, then `transfer_gold_labels` → re-integrate /
   clean / reduce with gold → `finetune_gold` → `eval_gold --split test`.
3. `eval_tip.py` in the finger venv on the oCam recording.
4. Stage 6 metrics once the real weights exist.
