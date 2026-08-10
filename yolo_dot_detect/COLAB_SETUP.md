# Google Colab + YOLO26 setup

Train the Braille-dot detector on a **Colab GPU** with **YOLO26**, saving every
epoch to Google Drive so auto-disconnects do not lose progress.

---

## 1. What to upload to Google Drive

### Recommended Drive layout (create these folders first)

```
My Drive/
└── BrailleLens_YOLO26/
    ├── braille_dots.zip          <- upload this (see step 2)
    ├── notebooks/
    │   └── BrailleLens_YOLO26_Colab.ipynb
    └── runs/                     <- created automatically by the notebook
        └── detect/
            └── braille_dot_yolo26/
                └── weights/
                    ├── best.pt
                    ├── last.pt
                    └── epoch*.pt
```

Exact Drive paths the notebook expects:

| Item | Google Drive path |
|------|-------------------|
| Project root | `MyDrive/BrailleLens_YOLO26/` |
| Dataset zip | `MyDrive/BrailleLens_YOLO26/braille_dots.zip` |
| Unpacked dataset | `MyDrive/BrailleLens_YOLO26/datasets/braille_dots/` |
| data.yaml | `MyDrive/BrailleLens_YOLO26/datasets/braille_dots/data.yaml` |
| Checkpoints / runs | `MyDrive/BrailleLens_YOLO26/runs/detect/braille_dot_yolo26/` |
| Best weights | `MyDrive/BrailleLens_YOLO26/runs/detect/braille_dot_yolo26/weights/best.pt` |
| Last (resume) | `MyDrive/BrailleLens_YOLO26/runs/detect/braille_dot_yolo26/weights/last.pt` |

In Colab after mount, these appear as:

```
/content/drive/MyDrive/BrailleLens_YOLO26/...
```

---

## 2. Pack the dataset on your PC (once)

From the **BrailleLens repo root**:

```bash
# If dataset not built yet:
py -3.11 -m yolo_dot_detect.prepare_dataset --dbsi-root "data DBSI/data" --copy-images

# Zip for Drive upload:
py -3.11 -m yolo_dot_detect.pack_for_colab
```

Output file on your PC:

```
yolo_dot_detect/colab_upload/braille_dots.zip
```

Upload that zip to:

```
Google Drive → MyDrive/BrailleLens_YOLO26/braille_dots.zip
```

Also upload the notebook:

```
PC:   yolo_dot_detect/BrailleLens_YOLO26_Colab.ipynb
Drive: MyDrive/BrailleLens_YOLO26/notebooks/BrailleLens_YOLO26_Colab.ipynb
```

You do **not** need to upload raw `data DBSI/` if you use the zip above.

---

## 3. Colab runtime

1. Open the notebook from Drive (or upload to Colab).
2. **Runtime → Change runtime type → GPU** (T4 is fine; A100 better).
3. Run cells top to bottom.
4. First run unpacks the zip into Drive (kept for next sessions).

---

## 4. Disconnect handling (weight sharing)

The notebook is designed so Colab idle disconnects are recoverable:

| Mechanism | What it does |
|-----------|--------------|
| **Drive mount** | All runs write under `MyDrive/BrailleLens_YOLO26/runs/` |
| **`save_period=1`** | Saves a checkpoint every epoch to Drive |
| **`last.pt` resume** | Re-run the train cell; it auto-resumes if `last.pt` exists |
| **`best.pt` on Drive** | Always available even if the session dies mid-epoch |
| **Keep-alive cell** | Optional JS ping to reduce idle disconnects (not 100%) |

After a disconnect:

1. Reconnect runtime → remount Drive (cell 1).
2. Re-run install + path cells.
3. Run the **Train** cell again — it detects `last.pt` and resumes.

---

## 5. After training — copy weights back to your PC

Download from Drive:

```
MyDrive/BrailleLens_YOLO26/runs/detect/braille_dot_yolo26/weights/best.pt
```

Place locally at:

```
yolo_dot_detect/runs/detect/braille_dot_yolo26/weights/best.pt
```

Then infer:

```bash
py -3.11 -m yolo_dot_detect.infer --image test-img.jpeg --weights yolo_dot_detect/runs/detect/braille_dot_yolo26/weights/best.pt --cluster
```

---

## 6. Troubleshooting

| Error | Fix |
|-------|-----|
| `ImportError: cannot import name '_Ink' from 'PIL._typing'` | Colab's preinstalled Pillow was upgraded in-place. The install cell now force-reinstalls Pillow and restarts the runtime automatically. After the restart, re-run cell 1 (Mount Drive) then cell 2. |
| `CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU` | A full page holds 1000-5000 dots and the assigner scales with boxes x anchors. Run cell **3b (tiling)** and train on the tiled dataset at `imgsz=640`. |
| `Slow image access detected` | Train from local disk, not mounted Drive. Cell 3 unpacks to `/content/braille_dots`; only checkpoints go to Drive. |
| `yolo26n.pt` not found | `ultralytics` is too old — re-run cell 2 and check `ultralytics.__version__`. |
| CUDA out of memory (still) | Lower `BATCH` to 4. Do not raise `IMGSZ` above `TILE`. |
| Session died mid-training | Re-run cells 1, 2, 3, 3b, then the Train cell — it resumes from `last.pt` on Drive. |
| Trained model finds 0 dots on a page | Tiled models need tiled inference: `--tile 640` (local) or use cell 6. |

---

## 7. Why Colab + YOLO26 helps

- **GPU** vs your CPU: hours → usually tens of minutes per full run.
- **YOLO26** (`yolo26n.pt`): NMS-free, STAL small-object assignment (better for tiny Braille dots).
- **`imgsz=1280`**: dots stay larger in the network (your local 640 run struggled here).
