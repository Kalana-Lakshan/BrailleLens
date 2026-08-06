# finger_cell_track

PC prototype: **MediaPipe Hands** index-fingertip tracking + DotNeuralNet cell
pre-scan → hit-test which Braille cell the finger covers (Learning / Testing).

Runs on your computer (webcam or IP Webcam). Phone / Flutter port is later.

## Setup

Uses a **local venv** (avoids system TensorFlow/protobuf clashes with MediaPipe):

```powershell
cd finger_cell_track
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
# For DotNeuralNet pre-scan (Step 4+):
.\.venv\Scripts\python.exe -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org ultralytics torch --index-url https://download.pytorch.org/whl/cpu
cd ..
```

DotNeuralNet weights must exist at `DotNeuralNet/weights/yolov8_braille.pt`
(for pre-scan / live Learning–Testing).

## Quick start

```powershell
$py = "finger_cell_track\.venv\Scripts\python.exe"

# Step 2 — hand tip only
& $py finger_cell_track/hand_track.py --source 0

# IP Webcam
& $py finger_cell_track/hand_track.py --source http://192.168.x.x:8080/video

# Step 5 — full app (after all steps)
& $py finger_cell_track/live_app.py --source 0 --mode learning --lang en
```

## Landmark

Index fingertip = MediaPipe Hands landmark **8**.

## Keys (live apps)

| Key | Action |
|-----|--------|
| Q | Quit |
| R | Rescan page → rebuild CellMap (live_app) |
| L / T | Learning / Testing mode (live_app) |

## Layout

| File | Role |
|------|------|
| `hand_track.py` | Live MediaPipe tip overlay |
| `cell_map.py` | CellMap, hit-test, dwell, SessionMemory |
| `prescan.py` | DotNeuralNet frame/image → CellMap |
| `modes.py` | Learning / Testing state |
| `live_app.py` | Combined PC demo |
