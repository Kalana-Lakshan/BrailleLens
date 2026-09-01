# Braille Domain YOLO Fine-Tune

Fine-tune `yolo26n_fingertip_best.pt` on **60 LabelMe-annotated** Braille fingertip photos.

Raw images: `Gold Dataset/Braille_fingertip/`  
Generated YOLO dataset: `datasets/braille_fingertip_yolo/`  
Colab upload zip: `colab_upload/braille_fingertip_yolo.zip`

---

## Workflow

```
LabelMe (60 photos) → build_dataset.py → train_local.py  (or Colab notebook)
```

### Local training (no GPU / Colab unavailable)

```powershell
$py = "finger_cell_track\.venv\Scripts\python.exe"

# Quick smoke test (~few min on CPU):
& $py finger_cell_track/yolo_domain_specific/train_local.py --epochs 5

# Full fine-tune (slow on CPU — 1–3+ hours; leave PC awake):
& $py finger_cell_track/yolo_domain_specific/train_local.py

# Resume after interrupt:
& $py finger_cell_track/yolo_domain_specific/train_local.py --resume
```

Outputs:
- Weights: `finger_cell_track/weights/yolo26n_fingertip_braille_best.pt`
- Metrics: `finger_cell_track/yolo_domain_specific/metrics_summary.json`
- Plots: `finger_cell_track/yolo_domain_specific/runs/fingertip_domain/yolo26n_braille_finetune/`

**Your PC:** Stage 5 venv has `torch 2.13+cpu` — training works but is much slower than Colab T4.

### Colab training (when GPU available)

```
build_dataset.py → pack_for_colab.py → Colab notebook → weights + metrics
```

### Step 1 — Annotate (manual)

Follow [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md).

```powershell
py -3.11 -m labelme "Gold Dataset/Braille_fingertip"
```

### Step 2 — Build YOLO dataset (PC)

```powershell
$py = "finger_cell_track\.venv\Scripts\python.exe"
& $py finger_cell_track/yolo_domain_specific/build_dataset.py
```

Creates `datasets/braille_fingertip_yolo/` with train/val/test split (48/6/6, seed 42).

### Step 3 — Pack for Colab

```powershell
& $py finger_cell_track/yolo_domain_specific/pack_for_colab.py
```

Output: `colab_upload/braille_fingertip_yolo.zip`

### Step 4 — Upload to Google Drive

| File | Drive path |
|------|------------|
| `braille_fingertip_yolo.zip` | `MyDrive/BrailleLens_Fingertip_Domain/braille_fingertip_yolo.zip` |
| `yolo26n_fingertip_best.pt` | `MyDrive/BrailleLens_Fingertip_Domain/yolo26n_fingertip_best.pt` |

Copy weights from `finger_cell_track/weights/yolo26n_fingertip_best.pt`.

### Step 5 — Colab

1. Upload [BrailleLens_Fingertip_Domain_Colab.ipynb](BrailleLens_Fingertip_Domain_Colab.ipynb) to [Google Colab](https://colab.research.google.com).
2. **Runtime → Change runtime type → T4 GPU**
3. Run all cells in order.
4. Optional: enable hyperparameter tuning cell (`RUN_TUNE = True`).

### Step 6 — Download results

From Drive:

- `yolo26n_fingertip_braille_best.pt` → save as `finger_cell_track/weights/yolo26n_fingertip_braille_best.pt`
- `metrics_summary.json` → paste values into [metrics_template.md](metrics_template.md) and share for review

---

## Scripts

| Script | Role |
|--------|------|
| `labelme_to_yolo.py` | Convert LabelMe JSON → YOLO `.txt` (called by `build_dataset.py`) |
| `build_dataset.py` | Split + Ultralytics layout + `data.yaml` |
| `pack_for_colab.py` | Zip dataset for Drive upload |
| `train_local.py` | Fine-tune on PC (CPU or CUDA) + export metrics |

---

## Training notes

- **Braille-only** fine-tune from existing `yolo26n_fingertip_best.pt` (not bare COCO weights).
- Augmentation, early stopping (`patience=15`), and L2 (`weight_decay`) are set in the Colab notebook.
- Optional `model.tune()` replaces sklearn RandomSearchCV — genetic search on val mAP50.
- With only 60 images, overfitting is possible; always check live video with `eval_tip.py` after deploying weights.
