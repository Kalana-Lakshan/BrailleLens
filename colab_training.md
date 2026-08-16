# Colab GPU training (Stage 4a + Stage 4b)

This PC cannot train the models (CPU-only PyTorch). Two **independent**
GPU jobs need to run on Google Colab. They do not share code, data, or
Drive folders. Assign **one person / one notebook / one runtime** to each
job. Do not start both in the same Colab tab.

| Job | What you train | Input zip (gitignored — send via Drive) | File you send back |
|---|---|---|---|
| **A — Stage 4a** | Single-class YOLO **cell** boxes | `braille_cells.zip` (~270 MB) | `braille_cell_best.pt` |
| **B — Stage 4b** | 64-class **CNN** on 64×64 crops | `cnn_crops.zip` (~320 MB) | `braille_cnn_mixed.pt` |

This is **not** the old dot detector (`yolo_dot_detect/`). Do not train that.

Runtime for both: **Colab → Runtime → Change runtime type → T4 GPU** (or better).
Confirm GPU before training:

```python
import torch
assert torch.cuda.is_available(), "Runtime is still CPU — switch to GPU and reconnect"
print(torch.cuda.get_device_name(0))
```

---

## Owner only — zip and upload before the other person starts

Datasets are not in git. From the **BrailleLens repo root** on the machine
that already ran `prepare_cell_dataset` and `data_pipeline.reduce`:

```powershell
mkdir colab_upload -ErrorAction SilentlyContinue

# Job A — cell detector pages + YOLO labels (~270 MB)
tar -a -c -f colab_upload/braille_cells.zip -C cell_detect/datasets braille_cells

# Job B — CNN crop archives (~320 MB)
tar -a -c -f colab_upload/cnn_crops.zip -C data_pipeline crops
```

Upload to two **separate** Google Drive folders (share the folder, not the
whole Drive):

```
MyDrive/BrailleLens_Colab/4a_cell_detector/braille_cells.zip
MyDrive/BrailleLens_Colab/4b_cnn/cnn_crops.zip
```

Also share this repo branch (or a zip of the source **without** `data DBSI/`
and `data Angelina/`):

```
https://github.com/Kalana-Lakshan/BrailleLens.git
branch: Kalana/improvement-preprocessing
```

Job B needs the Python package. Job A can train from the zip alone.

If the GitHub repo is private, the trainer either uses a GitHub token in
Colab, or you send a source zip of the repo (code only).

---

# Job A — Train the cell detector (Stage 4a)

**Goal:** YOLO finds every Braille *cell* box on a page. Not dots. Not
letters.

**Expected time on a T4:** a few hours for 80 epochs. Checkpoints land on
Drive every 5 epochs, so a disconnect is recoverable.

**Do not** turn on `fliplr` or `flipud`. Braille is not mirror-symmetric.

**If GPU memory runs out:** drop `--batch` to `2`, or `--imgsz` to `960`.
Do **not** tile the page (that is the *dot* detector's trick).

Optional: you can upload and run `cell_detect/BrailleLens_CellDetector_Colab.ipynb`
instead of pasting the cells below. The commands here are the source of truth
if the notebook and this file ever disagree.

## A1. New Colab notebook

New notebook. GPU runtime. One sentence at the top: `BrailleLens Stage 4a cell detector`.

## A2. Install

```python
!pip -q install ultralytics pyyaml opencv-python-headless pillow
```

## A3. Mount Drive and unzip the dataset

```python
from google.colab import drive
from pathlib import Path
import torch, zipfile, yaml

assert torch.cuda.is_available(), "Switch Runtime → GPU and reconnect"
drive.mount("/content/drive")

zip_path = Path("/content/drive/MyDrive/BrailleLens_Colab/4a_cell_detector/braille_cells.zip")
assert zip_path.exists(), f"Missing {zip_path} — ask the owner to upload braille_cells.zip"

out = Path("/content/braille_cells")
if not (out / "data.yaml").exists():
    with zipfile.ZipFile(zip_path) as z:
        z.extractall("/content")

# Force a Colab-local path. The zip may still contain a Windows path.
yaml_path = out / "data.yaml"
cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
cfg["path"] = str(out)
yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

print("cuda", torch.cuda.get_device_name(0))
print("pages train/val/test",
      len(list((out / "images/train").glob("*"))),
      len(list((out / "images/val").glob("*"))),
      len(list((out / "images/test").glob("*"))))
```

Expect roughly **332 / 56 / 44** pages. If those folders are empty, the zip is
wrong.

## A4. Train (this is the job)

```python
from ultralytics import YOLO
from pathlib import Path

run_dir = Path("/content/drive/MyDrive/BrailleLens_Colab/4a_cell_detector/runs")
run_dir.mkdir(parents=True, exist_ok=True)

model = YOLO("yolo26n.pt")
model.train(
    data="/content/braille_cells/data.yaml",
    epochs=80,
    imgsz=1280,
    batch=4,
    device=0,
    workers=2,
    patience=15,
    seed=42,
    max_det=800,
    fliplr=0.0,
    flipud=0.0,
    mosaic=0.5,
    mixup=0.0,
    hsv_v=0.40,
    degrees=3.0,
    translate=0.10,
    scale=0.30,
    project=str(run_dir),
    name="braille_cell_yolo26",
    exist_ok=True,
    save_period=5,
)
```

If `yolo26n.pt` fails to download, stop and tell the owner. Do not silently
switch to a different model family unless they agree.

To **resume** after Colab killed the session:

```python
from ultralytics import YOLO
last = "/content/drive/MyDrive/BrailleLens_Colab/4a_cell_detector/runs/braille_cell_yolo26/weights/last.pt"
YOLO(last).train(resume=True)
```

## A5. Copy the weight the repo expects

When training finishes (or after a decent `best.pt` if you have to stop):

```python
import shutil
from pathlib import Path

src = Path("/content/drive/MyDrive/BrailleLens_Colab/4a_cell_detector/runs/braille_cell_yolo26/weights/best.pt")
dst = Path("/content/drive/MyDrive/BrailleLens_Colab/4a_cell_detector/braille_cell_best.pt")
assert src.exists(), src
shutil.copy2(src, dst)
print("send this file back:", dst, "bytes", dst.stat().st_size)
```

Also paste the last Ultralytics metrics line into the chat with the owner
(`metrics/mAP50(B)`, precision, recall). A 1-epoch CPU smoke test on this
dataset was only **mAP50 ≈ 0.12**. A real 80-epoch GPU run should be much
higher. If mAP50 stays below ~0.40 after tens of epochs, stop and ping the
owner — do not burn the rest of the Colab quota.

## A6. Send back

Email / Drive-share **one** file:

`MyDrive/BrailleLens_Colab/4a_cell_detector/braille_cell_best.pt`

Owner drops it at:

`cell_detect/weights/braille_cell_best.pt`

---

# Job B — Train the CNN (Stage 4b)

**Goal:** 64-class `SimpleBrailleCNN` on mixed DSBI + Angelina 64×64
grayscale crops. One class per Braille code 0–63 (code 0 = blank cell).

**Expected time on a T4:** well under 2 hours for 20 epochs.

**Do not** run `--smoke-test` as the real job. That is a 256-crop sanity
check only.

## B1. New Colab notebook

**Different** notebook from Job A. GPU runtime. Title: `BrailleLens Stage 4b CNN`.

## B2. Clone the training code

If GitHub clone works:

```python
from google.colab import drive
import torch

assert torch.cuda.is_available(), "Switch Runtime → GPU and reconnect"
drive.mount("/content/drive")

%cd /content
!git clone -b Kalana/improvement-preprocessing https://github.com/Kalana-Lakshan/BrailleLens.git
%cd /content/BrailleLens
```

If the repo is private or the branch is not on GitHub yet, skip clone.
Upload a source zip of the repo to
`MyDrive/BrailleLens_Colab/4b_cnn/braillelens_code.zip` and unzip to
`/content/BrailleLens` instead.

Need these packages on disk after unzip/clone:

- `braille_cnn/` (especially `train_classifier.py`, `cnn.py`)
- `data_pipeline/` (`crop_dataset.py`, `transform.py`, `contracts.py`)

## B3. Unzip the crop archives into the repo

```python
from pathlib import Path
import zipfile, shutil

zip_path = Path("/content/drive/MyDrive/BrailleLens_Colab/4b_cnn/cnn_crops.zip")
assert zip_path.exists(), f"Missing {zip_path} — ask the owner to upload cnn_crops.zip"

crops = Path("/content/BrailleLens/data_pipeline/crops")
crops.mkdir(parents=True, exist_ok=True)
if not (crops / "crops_train.npz").exists():
    with zipfile.ZipFile(zip_path) as z:
        z.extractall("/content/BrailleLens/data_pipeline")

# The zip contains a folder named "crops". If extractall dumped files one
# level too high, move them.
if not (crops / "crops_train.npz").exists() and Path("/content/BrailleLens/data_pipeline/crops_train.npz").exists():
    for name in ("crops_train.npz", "crops_val.npz", "crops_test.npz", "crops_meta.json"):
        src = Path("/content/BrailleLens/data_pipeline") / name
        if src.exists():
            shutil.move(str(src), str(crops / name))

needed = ["crops_train.npz", "crops_val.npz", "crops_test.npz"]
for name in needed:
    p = crops / name
    assert p.exists(), f"missing {p}"
    print(name, round(p.stat().st_size / 1e6, 1), "MB")
```

Expect about **244 / 43 / 31 MB** and class counts **120,809 / 21,095 / 15,655**.

## B4. Install (use Colab's GPU PyTorch — do not pip-install a CPU wheel)

```python
%cd /content/BrailleLens
!pip -q install numpy pyyaml
```

Quick import check:

```python
%cd /content/BrailleLens
import torch
from braille_cnn.cnn import SimpleBrailleCNN
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(SimpleBrailleCNN())
```

## B5. Train (this is the job)

Paste this as **one** Colab cell (do not split the `!python` line):

```python
%cd /content/BrailleLens
!python -m braille_cnn.train_classifier --epochs 20 --device cuda --balance-domains --crops-dir /content/BrailleLens/data_pipeline/crops --out-checkpoint /content/drive/MyDrive/BrailleLens_Colab/4b_cnn/braille_cnn_mixed.pt
```

Keep **Colab connected** until it prints `best val acc:`. The script writes
the best epoch to Drive whenever val accuracy improves, so a late disconnect
can still leave a usable checkpoint.

Optional 1-minute smoke (not the real model — do this only if B5 failed to start):

```python
%cd /content/BrailleLens
!python -m braille_cnn.train_classifier --smoke-test --device cuda --crops-dir /content/BrailleLens/data_pipeline/crops --out-checkpoint /content/drive/MyDrive/BrailleLens_Colab/4b_cnn/braille_cnn_smoke.pt
```

Do **not** send `braille_cnn_smoke.pt` back as the trained model.

## B6. Send back

Share **one** file:

`MyDrive/BrailleLens_Colab/4b_cnn/braille_cnn_mixed.pt`

Also paste the last few `epoch .. val_acc=` lines. Owner drops it at:

`braille_cnn/checkpoints/braille_cnn_mixed.pt`

A useful val accuracy is high (old scan-only DBSI CNN was ~99% on that
scanner test set; mixed handheld+scan will usually land lower). If val
accuracy is stuck near random (~1.6% = 1/64) after a few epochs, stop.

---

## After both files are back (owner, not the Colab person)

```powershell
# from repo root
copy <path-to>\braille_cell_best.pt cell_detect\weights\braille_cell_best.pt
copy <path-to>\braille_cnn_mixed.pt braille_cnn\checkpoints\braille_cnn_mixed.pt

py -3.11 -m cell_detect.evaluate_detector
py -3.11 -m braille_cnn.eval_angelina
```

Do not commit the `.pt` files (they are gitignored). Keep them on disk /
Drive.

## FAQ

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available() == False` | Runtime → GPU, then reconnect. Old cells still run on CPU until you do. |
| Colab session died | Job A: resume from `last.pt`. Job B: re-run B5; best-so-far is already on Drive. |
| `data.yaml` path is `C:\Users\...` | Cell A3 rewrites it. Re-run A3. |
| OOM in 4a | `batch=2` then `imgsz=960`. Never enable flips. |
| Job A mAP looks like 0.12 after epoch 1 | Normal for a cold start. Judge after tens of epochs. |
| Wrong repo / training `yolo_dot_detect` | Stop. That is the fallback *dot* detector, not this job. |
