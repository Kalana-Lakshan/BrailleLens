# YOLOv8 Braille Dot Detection

Separate module that replaces classical peak-finding (`braille_cnn/dot_detect.py`)
with a **YOLO26** (or YOLOv8) transfer-learned embossed-dot detector, trained on
DSBI with strong data augmentation.

**Prefer Google Colab GPU** for training — see [`COLAB_SETUP.md`](COLAB_SETUP.md)
and [`BrailleLens_YOLO26_Colab.ipynb`](BrailleLens_YOLO26_Colab.ipynb).

Does **not** modify the existing CNN cell classifier — only improves the
*where are the dots?* stage. Detected centers plug into the same
`cluster_into_cells()` used by the live pipeline.

---

## Why this exists

| Stage | Current (`braille_cnn`) | This folder |
|-------|-------------------------|-------------|
| Dot finding | Gaussian blur difference + percentile + NMS | **YOLOv8n** fine-tuned on DSBI |
| Cell grouping | Classical clustering | Unchanged (reuse) |
| Cell → letter | `SimpleBrailleCNN` | Unchanged |

Classical detection fails on uneven phone lighting and worn dots. A learned
detector with photometric / geometric augmentation targets that domain gap.

---

## Folder layout

```
yolo_dot_detect/
├── prepare_dataset.py   # Step 1: DSBI annotations → YOLO boxes
├── visualize_labels.py  # Step 1b: sanity-check label overlays
├── tile_dataset.py      # Step 1c: slice pages into 640px tiles (avoids OOM)
├── pack_for_colab.py    # Zip dataset for Google Drive upload
├── train.py             # Step 2: transfer learning + augmentation
├── evaluate.py          # Step 3: mAP / precision / recall
├── infer.py             # Step 4: run on a photo + optional cell clustering
├── detect_dots.py       # API: YoloDotDetector / detect_dot_centers_yolo
├── configs/default.yaml # Hyperparams + aug settings
├── datasets/            # Generated (gitignored)
└── runs/                # Training outputs (gitignored)
```

---

## Prerequisites

- Python 3.11 (`py -3.11`)
- DSBI clone at `data DBSI/` (same as the main README)
- Dependencies:

```bash
py -3.11 -m pip install -r yolo_dot_detect/requirements.txt
```

GPU optional. Defaults use `device=cpu` and `batch=8`. If you have CUDA, pass
`--device 0` and raise `--batch`.

---

## Step-by-step

### Step 1 — Convert DSBI → YOLO labels

Each *raised* embossed dot becomes one YOLO box (`class 0 = braille_dot`).
Empty grid positions are ignored.

```bash
# from repo root
py -3.11 -m yolo_dot_detect.prepare_dataset --dbsi-root "data DBSI/data"
```

Optional: copy images instead of symlinking:

```bash
py -3.11 -m yolo_dot_detect.prepare_dataset --dbsi-root "data DBSI/data" --copy-images
```

Preview a few pages:

```bash
py -3.11 -m yolo_dot_detect.visualize_labels --n 4
```

Outputs land in `yolo_dot_detect/datasets/braille_dots/` (`images/`, `labels/`, `data.yaml`).

### Step 1c — Tile the pages (strongly recommended)

A full page holds 1,000–5,000 dots. YOLO's label assigner allocates memory
proportional to *boxes × anchors*, so full pages cause
`CUDA OutOfMemoryError in TaskAlignedAssigner`. Tiling also keeps dots at native
pixel size instead of shrinking them during resize.

```bash
py -3.11 -m yolo_dot_detect.tile_dataset
```

Produces `datasets/braille_dots_tiled/` with ~110 dots per 640px tile. Train on
it with `imgsz=640`, and run inference with `--tile 640`.

### Step 2 — Transfer learning + data augmentation

Starts from **COCO-pretrained** `yolov8n.pt` and fine-tunes on the Braille-dot
dataset. Augmentations (see `configs/default.yaml`):

- HSV (hue / saturation / brightness)
- Rotation ±10°, translate, scale, mild perspective
- Mosaic + mixup (helps small objects)
- Random erasing (worn / occluded dots)
- **No** left-right or up-down flips (Braille orientation must stay correct)

```bash
py -3.11 -m yolo_dot_detect.train
```

Useful overrides:

```bash
# shorter smoke run
py -3.11 -m yolo_dot_detect.train --epochs 5 --batch 4

# larger backbone + GPU
py -3.11 -m yolo_dot_detect.train --model yolov8s.pt --device 0 --batch 16
```

Best checkpoint:

`yolo_dot_detect/runs/detect/braille_dot_yolov8/weights/best.pt`

### Step 3 — Evaluate

```bash
py -3.11 -m yolo_dot_detect.evaluate
```

Reports Precision, Recall, mAP50, mAP50-95 on the DSBI test split (used as val).

### Step 4 — Infer on a page photo

Tiled inference is on by default (`--tile 640`), matching tiled training:

```bash
py -3.11 -m yolo_dot_detect.infer --image test-img.jpeg
py -3.11 -m yolo_dot_detect.infer --image test-img.jpeg --cluster --link-distance 15
py -3.11 -m yolo_dot_detect.infer --image test-img.jpeg --compare-classical
py -3.11 -m yolo_dot_detect.infer --image test-img.jpeg --tile 0   # whole-image
```

### Use from code

```python
from yolo_dot_detect import YoloDotDetector
from braille_cnn.dot_detect import cluster_into_cells

det = YoloDotDetector(conf=0.25, device="cpu", tile=640)
centers = det.detect("test-img.jpeg")          # (N, 2) xy
clusters = cluster_into_cells(centers, link_distance=15.0)
```

---

## Transfer learning summary

1. **Pretrained**: Ultralytics YOLOv8 trained on COCO (generic objects).
2. **Fine-tune**: single class `braille_dot` on DSBI raised-dot boxes.
3. **Augment**: photometric + geometric transforms during training to close the
   scanner → phone-camera gap without needing handheld labels yet.
4. **Downstream**: YOLO centers → existing cell clustering → existing CNN.

---

## Tuning tips

| Symptom | Try |
|---------|-----|
| Misses faint dots | Lower `--conf` (e.g. 0.15); raise `hsv_v` / `erasing` and retrain |
| Too many false positives | Raise `--conf`; check label previews for noisy GT |
| Merged cells after clustering | Adjust `--link-distance` (measure median NN distance of YOLO centers) |
| Slow on CPU | Keep `yolov8n.pt`, `imgsz=640`, `batch=4` |
| Phone photos still weak | Collect a few labelled handheld pages and fine-tune further from `best.pt` |
| Dense pages miss dots at inference | Ensure `max_det=3000` (already default in this package) |
| Training very slow on CPU | Use `--device 0` if GPU available; or `--epochs 5` for a smoke run |

**Note:** Training on CPU with 1000+ boxes/page takes ~30–40s per batch. A full 15-epoch run can take a few hours. Prefer a CUDA GPU when possible.

---

## Relation to the rest of BrailleLens

This folder is self-contained. The Flutter app and `camera_capture/` still use
classical detection until you wire `YoloDotDetector` into `infer_page.py`
(optional follow-up).
