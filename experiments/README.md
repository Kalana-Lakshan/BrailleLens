# experiments/

Parked research that is **not** the deployment path.

| Folder / file | What it was |
|---|---|
| `braille_detector/` | Single-stage cell detector (paper-style POC). Isolated from `braille_cnn/`. |
| `DotNeuralNet/` | Third-party YOLO that detects whole Braille cells. Still used by `finger_cell_track/` for page pre-scan. |
| `IMPLEMENTATION_PLAN.md` | Historical three-branch plan (camera + live Sinhala). |

The product pipeline at the repo root is: `braille_cnn/` (classifier) → optional `yolo_dot_detect/` (dots if classical detection is weak) → `camera_capture/` (live demo).
