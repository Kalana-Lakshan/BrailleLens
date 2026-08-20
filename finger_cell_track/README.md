# Stage 5 — finger_cell_track

Live learning app: prescanned CellMap + fingertip hit-test → one character
per dwell in the terminal.

**Tip detector:** `SkinContourTip` is the default (`--tip-backend skin`).
Contact is the pad deepest into the page (not the nail). Thin edge/corner
ghosts are rejected; `TipEMA` also ignores teleport jumps.
MediaPipe is optional (`--tip-backend mediapipe` or `auto`). TipYOLO is a
baseline only (`--tip-backend yolo`).

**Page capture:** `--auto-scan` freezes a still, hand-free frame and prints
`[PAGE] CAPTURED`. `R` is still a manual override.

Prescan uses `recognize_page()` (cells if our YOLO exists, else dots).
DotNeuralNet is a last-resort fallback.

## Setup

Use **`finger_cell_track/.venv`** for Stage 5. MediaPipe is broken in the
root env (`protobuf` / TensorFlow clash). That venv already imports
mediapipe 0.10.14, torch, ultralytics and OpenCV together.

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

# Tip-only check (no Braille book) — yellow dot on fingertip
& $py finger_cell_track/tip_dot_test.py --source http://PHONE_IP:8080/video
& $py finger_cell_track/tip_dot_test.py --source 0 --show-mask

# Full CellMap + tip → Learning/Testing (3 s dwell before cell print)
& $py finger_cell_track/live_app.py --source 0 --mode learning --lang si --tip-backend skin --scan-backend cells --dwell-ms 3000

# IP Webcam on phone (replace IP)
& $py finger_cell_track/live_app.py --source http://PHONE_IP:8080/video --mode learning --lang si --tip-backend skin --scan-backend cells --dwell-ms 3000

# Replay a recording in the terminal (no window)
& $py finger_cell_track/live_app.py --source path\to.mp4 --no-window --force-scan --tip-backend skin --scan-backend cells --lang si --dwell-ms 3000

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
