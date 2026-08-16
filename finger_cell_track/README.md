# finger_cell_track

PC prototype: **YOLO26 fingertip** tracking + DotNeuralNet cell pre-scan →
hit-test which Braille cell the finger covers (Learning / Testing).

Works with tip-only / no-palm views (unlike MediaPipe). Phone / glasses later.

## Setup

Uses a **local venv**:

```powershell
cd finger_cell_track
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org ultralytics torch --index-url https://download.pytorch.org/whl/cpu
cd ..
```

Needed weights:

- Braille cells: `experiments/DotNeuralNet/weights/yolov8_braille.pt`
- Fingertip: `finger_cell_track/weights/yolo26n_fingertip_best.pt` (from Colab)

## Quick start

```powershell
$py = "finger_cell_track\.venv\Scripts\python.exe"

# Tip YOLO only (webcam or IP Webcam)
& $py finger_cell_track/tip_track.py --source 0
& $py finger_cell_track/tip_track.py --source http://PHONE_IP:8080/video

# Full CellMap + tip → Learning/Testing
& $py finger_cell_track/live_app.py --source 0 --mode learning --lang en

# Auto-label tip-on-Braille photos (then correct in Roboflow + fine-tune)
& $py finger_cell_track/auto_label_tips.py --images path\to\photos
```

Train tip model on Colab T4: `BrailleLens_Fingertip_YOLO26_Colab.ipynb` (see `COLAB_TRAIN.md`).

## Keys (live_app)

| Key | Action |
|-----|--------|
| Q | Quit |
| R | Rescan page → rebuild CellMap |
| L / T | Learning / Testing mode |

## Layout

| File | Role |
|------|------|
| `tip_yolo.py` | Load tip weights, detect tip center |
| `tip_track.py` | Live tip YOLO overlay |
| `live_app.py` | Tip + CellMap Learning/Testing |
| `auto_label_tips.py` | Pseudo-label photos for domain fine-tune |
| `hand_track.py` | Legacy MediaPipe tip (palm required) |
| `cell_map.py` / `prescan.py` / `modes.py` | CellMap + modes |
