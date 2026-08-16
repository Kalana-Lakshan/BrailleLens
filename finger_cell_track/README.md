# Stage 5 — finger_cell_track

Live learning app: prescanned CellMap + fingertip hit-test → one character
per dwell in the terminal.

**Tip detector:** MediaPipe Hands is primary (`MediaPipeTip`). Skin-contour
is the fallback. TipYOLO stays as a baseline (`--tip-backend yolo`).

**Page capture:** `--auto-scan` freezes a still, hand-free frame and prints
`[PAGE] CAPTURED`. `R` is still a manual override.

Prescan still uses DotNeuralNet until `recognize_page(backend="cells")` has
trained cell-detector weights. That swap is `prescan.py` only.

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
& $py finger_cell_track/live_app.py --source 0 --mode learning --lang si --auto-scan

# Measure tip backends on recorded footage
& $py finger_cell_track/eval_tip.py

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
| `autoscan.py` | `PageWatcher` — auto page capture, no R required |
| `hand_track.py` | `MediaPipeTip` + `SkinContourTip` |
| `eval_tip.py` | Measure MediaPipe vs contour vs YOLO on video |
| `tip_yolo.py` | Old tip YOLO (baseline only) |
| `live_app.py` | Tip + CellMap Learning/Testing |
| `cell_map.py` / `prescan.py` / `modes.py` | CellMap + modes |
| `registration.py` | ORB+RANSAC; `status` is OK / LOST |
