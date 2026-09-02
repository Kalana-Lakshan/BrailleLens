# Quick start — Combined YOLO26 fingertip training (Option A)

Train **yolo26n.pt** on **TI1K + Roboflow + Braille_fingertip** in one run.

---

## Fast path (you have ~15 minutes setup)

### A) If you already have `fingertip_yolo26.zip` on Drive (~436 MB from earlier)

1. Pack Braille only (on PC):

```powershell
$py = "finger_cell_track\.venv\Scripts\python.exe"
& $py finger_cell_track/yolo_domain_specific/pack_for_colab.py
```

2. Upload to `MyDrive/BrailleLens_Fingertip_Combined/`:
   - `fingertip_yolo26.zip` (TI1K+Roboflow — from old training)
   - `braille_fingertip_yolo.zip` (from step 1, ~158 MB)

3. Open **`BrailleLens_Fingertip_Combined_Colab.ipynb`** in Colab → T4 GPU → Run all.

   The notebook **merges** the two zips on Colab and trains from **`yolo26n.pt`**.

---

### B) Build full combined zip on PC (all datasets local)

1. Download datasets into `finger_cell_track/datasets/`:
   - [TI1K ZIP](https://github.com/MahmudulAlam/TI1K-Dataset/archive/refs/heads/master.zip) → extract
   - [Roboflow Finger Tip Detection](https://universe.roboflow.com/first-leo0f/finger-tip-detection-i4xqf) → YOLO26 export → `Finger Tip Detection.v1i.yolo26/`

2. Build + pack:

```powershell
$py = "finger_cell_track\.venv\Scripts\python.exe"
& $py finger_cell_track/yolo_domain_specific/build_combined_dataset.py --clean
& $py finger_cell_track/yolo_domain_specific/pack_combined_for_colab.py
```

3. Upload **one file** to Drive:
   - `colab_upload/fingertip_combined_yolo.zip` → `MyDrive/BrailleLens_Fingertip_Combined/`

4. Colab notebook → Run all (skips merge if combined zip exists).

---

## After training

Download from Drive:
- `yolo26n_fingertip_combined_best.pt` → `finger_cell_track/weights/`
- `metrics_summary.json`

Replace default in app or pass `--tip-weights`.

---

## Drive folder layout

```
MyDrive/BrailleLens_Fingertip_Combined/
  fingertip_combined_yolo.zip     ← OR fingertip_yolo26.zip + braille_fingertip_yolo.zip
  runs/                           ← checkpoints (auto-created)
  yolo26n_fingertip_combined_best.pt
  metrics_summary.json
```

No need to upload `yolo26n.pt` — Colab downloads it automatically.
