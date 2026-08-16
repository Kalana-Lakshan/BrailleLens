# Stage 4a — Cell detector

Single-class YOLO: **where are the Braille cells on this page?**

This is the first half of the two-stage recogniser. `braille_cnn/` answers
**what is each cell?** (code 0–63).

Do not confuse this folder with `yolo_dot_detect/`, which finds individual
raised *dots* and stays as the fallback / baseline.

```
manifest_clean.csv
        |
        v
  prepare_cell_dataset.py   ->  datasets/braille_cells/
  train_detector.py         ->  runs/detect/.../weights/best.pt
  detect_cells.py           ->  CellDetector API (boxes)
  evaluate_detector.py      ->  mAP50 / precision / recall
```

---

## Why cells, not dots

The old chain was dots → `fit_cell_grid` → cells. Grid fitting is where
handheld accuracy died (DBSI end-to-end ~97%, Angelina ~43%). Detecting
cell boxes skips that step and also gives the live app the boxes it needs
for the finger hit-test.

---

## Files

| File | Role |
|---|---|
| `prepare_cell_dataset.py` | Manifest → YOLO `images/` + `labels/` + `data.yaml` |
| `train_detector.py` | Transfer-learn from `yolo26n.pt` |
| `configs/cells.yaml` | Hyperparams (`fliplr=0`, `flipud=0`, `max_det=800`) |
| `detect_cells.py` | `CellDetector.detect_boxes(image)` |
| `evaluate_detector.py` | Val metrics |
| [`../colab_training.md`](../colab_training.md) | Job A: Colab GPU training (separate from CNN Job B) |

---

## Run (from repo root)

```bash
# 1. Build the YOLO dataset from the clean manifest
py -3.11 -m cell_detect.prepare_cell_dataset

# 2. Smoke-test the dataset locally (1 epoch, CPU) — catches path/label bugs
py -3.11 -m cell_detect.train_detector --smoke-test

# 3. Real training: Colab / Kaggle GPU (see COLAB_SETUP.md)
#    Then copy best.pt to:
#    cell_detect/weights/braille_cell_best.pt

# 4. Evaluate
py -3.11 -m cell_detect.evaluate_detector
```

If `prepare_cell_dataset` warns that **val is empty**, rebuild the manifest
with `--split-mode rebalance` (official DSBI has no val; official Angelina
has no test).

---

## Label format

One line per cell, class `0 = braille_cell`:

```
0  x_center  y_center  width  height    # all in [0, 1]
```

The 64-way dot code is **not** a YOLO class. That is the CNN's job.

---

## Next stage

`py -3.11 -m braille_cnn.recognize --image test-img.jpeg --backend cells`
