# live_reading — finger-occlusion-aware live Braille reading

Answers the problem: a live camera frame can't classify a cell the reading
finger is physically covering — the pixels aren't there. This folder
implements the fix: **pre-scan the page once while it's still fully
visible, then track where the finger is and look up what's already known**,
rather than trying to re-classify the covered cell from the live frame.

```
pre-scan (once, unoccluded)  ─────────────►  PageScan
    yolo_dot_detect + braille_cnn's               │  {cell center, character}
    cluster_into_cells + SimpleBrailleCNN          │  for every cell on the page
                                                    │
each live frame ──► fingertip (x,y) ──registration──► same coordinate frame
    yolo_finger_detect                  (ORB + RANSAC        │
                                          homography)         ▼
                                                      PageScan.nearest_cell()
                                                                │
                                                                ▼
                                                        Cell (character)
```

## Why registration, not a fixed pixel mapping

The simplest version of this idea assumes the camera and page never move
between the pre-scan and the live frames, so a fingertip's live-frame pixel
coordinates map directly onto the pre-scan lookup table. That assumption
doesn't hold here — the camera is on the reader's head (or a phone held by
hand), and their head moves while reading. `registration.py` instead
estimates a homography between each live frame and the reference pre-scan
frame (ORB keypoint matching + RANSAC) and transforms the fingertip point
through it before doing the lookup.

This also happens to be the right tool for the occlusion problem
specifically: registration matches on keypoints spread across the *whole*
frame, most of which the hand isn't covering at any given moment. A
per-point/local-patch tracker centered on the fingertip itself would be
tracking exactly the region most likely to be partially obscured by the
hand doing the reading. Validated directly (see chat history) under both
rotation+translation+scale drift and ~25% synthetic occlusion — sub-pixel
registration error in both cases.

## Files

| File | What it does |
|---|---|
| `pre_scan.py` | `scan_page()` — preprocesses the photo (`cell_detect.preprocess`: CLAHE contrast + best-effort deskew, both on by default), then runs the adopted detection pipeline (`yolo_dot_detect.YoloDotDetector` → `braille_cnn.dot_detect.cluster_into_cells` → `SimpleBrailleCNN`) once on it, returns a `PageScan` (reference grayscale image + list of `Cell(center, code, label, confidence)`). `PageScan.nearest_cell()` finds the closest known cell to a query point, with a distance cutoff so an out-of-range point (finger off the page, or momentarily unmatched) doesn't return a bogus nearest cell. |
| `registration.py` | `FrameRegistration` — ORB + RANSAC homography between any live frame and the pre-scan's reference frame. `homography_or_last()` falls back to the last known-good homography if one frame's matching momentarily fails, rather than treating a single noisy frame as "lost". |
| `live_loop.py` | `LiveReader` — wires fingertip detection (`yolo_finger_detect`) + registration + lookup together. `process_frame()` returns a `ReadEvent` every call; `process_frame_debounced()` only returns a `Cell` when it's *different* from the last one returned, so a still finger doesn't re-emit the same character every frame. `run_on_video()` drives it from a camera index, video file, or IP-camera URL. |

## Status

Architecture fully wired and validated end-to-end (pre-scan → registration
→ lookup, including under simulated rotation/translation/occlusion — see
chat history for the exact numbers). `yolo_finger_detect`'s model is only
smoke-tested so far (2 epochs on TI1K); a real training run, and likely
fine-tuning on real photos of a finger actually reading a Braille page
(TI1K is outdoor pointing gestures, not this domain — see
`yolo_finger_detect/README.md`... TODO if not yet written), are the
remaining steps before this is usable end-to-end on real hardware.

## Usage

```bash
py -3.11 -m live_reading.live_loop \
    --page-image path/to/clear_page_photo.jpg \
    --braille-checkpoint braille_cnn/checkpoints/braille_cnn_dbsi_finetuned.pt \
    --yolo-dot-weights yolo_dot_detect/runs/detect/braille_dot_yolov8/weights/best.pt \
    --fingertip-weights yolo_finger_detect/runs/detect/fingertip_yolov8/weights/best.pt \
    --source 0
```

Pre-scan preprocessing (CLAHE + deskew, both on by default — see
`cell_detect/preprocess.py`) can be toggled with `--no-clahe` / `--no-deskew`.
CAUTION: CLAHE was measured to make *cell* detection worse, not better, on
one real phone photo (`cell_detect/README.md`'s "Preprocessing before
detection" section) -- being on by default here is so it can be checked
against real pre-scan photos too, not a claim it already helps. If your
pre-scan finds noticeably fewer cells with it on, pass `--no-clahe`.
