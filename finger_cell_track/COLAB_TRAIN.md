# Train fingertip YOLO26n on Google Colab (T4 GPU)

## 1. On your PC — zip the dataset

The prepared folder is already at `finger_cell_track/datasets/fingertip_yolo26/` (~436 MB).

**PowerShell** (from BrailleLens repo root):

```powershell
Compress-Archive -Path "finger_cell_track\datasets\fingertip_yolo26\*" `
  -DestinationPath "finger_cell_track\datasets\fingertip_yolo26.zip" -Force
```

Or zip the folder in File Explorer → `fingertip_yolo26.zip`.

## 2. Upload to Google Drive

Upload **either**:

| What you upload | Drive path example |
|-----------------|--------------------|
| Folder `fingertip_yolo26/` (with `data.yaml` inside) | `MyDrive/BrailleLens/fingertip_yolo26/` |
| Or zip `fingertip_yolo26.zip` | `MyDrive/BrailleLens/fingertip_yolo26.zip` |

The Colab notebook accepts **both**. The error `Zip not found: ...fingertip_yolo26.zip` means only a **folder** was uploaded (or a different name/path) while the old cell looked for a zip only — re-upload the notebook cell or re-run with the updated notebook.

## 3. Open the Colab notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. **File → Upload notebook** → choose  
   `finger_cell_track/BrailleLens_Fingertip_YOLO26_Colab.ipynb`
3. **Runtime → Change runtime type**
   - Hardware accelerator: **T4 GPU**
   - Save

## 4. Run the notebook

| Cell | What it does |
|------|----------------|
| 1 | Checks CUDA / T4 |
| 2 | Mounts Drive, unzips dataset |
| 3 | Fixes `data.yaml` paths |
| 4 | Installs `ultralytics` |
| 5 | Trains `yolo26n` (~50 epochs; batch 16, lower to 8 if OOM) |
| 6 | Validation plots |
| 7 | Copies `yolo26n_fingertip_best.pt` to Drive |

If cell 2 fails on zip path, edit:

```python
ZIP_PATH = Path('/content/drive/MyDrive/YOUR_FOLDER/fingertip_yolo26.zip')
```

## 5. After training

1. Download from Drive: `BrailleLens/yolo26n_fingertip_best.pt`
2. On PC save as:  
   `finger_cell_track/weights/yolo26n_fingertip_best.pt`
3. Keep Colab open until the copy-to-Drive cell finishes (or download from the left file panel under `/content/runs/.../weights/best.pt`)

## Tips

- **Do not** select TPU — only **T4 GPU**
- If Colab disconnects: remount Drive and re-run from unzip (or keep `project/exist_ok` and resume with `model.train(resume=True)` if a `last.pt` exists)
- First run downloads `yolo26n.pt` automatically
- Expect tens of minutes to a couple of hours for 50 epochs on T4, depending on queue load
