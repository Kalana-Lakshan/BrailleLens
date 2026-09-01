"""Stage 5.3 — measure fingertip detectors on real Braille-reading footage.

Compares MediaPipe, the skin-contour fallback, and TipYOLO (baseline only).

    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/eval_tip.py
    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/eval_tip.py --video path\\to.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

DEFAULT_VIDEO = Path(r"C:\Users\ADMIN\Documents\oCam\Record_2026_08_07_11_00_29_714.mp4")


def _run_backend(name: str, detect_fn, cap, max_frames: int) -> dict:
    hits = 0
    frames = 0
    while frames < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames += 1
        tip = detect_fn(frame)
        if tip is not None:
            hits += 1
    rate = hits / max(frames, 1)
    return {"name": name, "frames": frames, "hits": hits, "rate": rate}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fingertip backends")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--stride", type=int, default=2, help="Evaluate every Nth frame")
    parser.add_argument("--skip-yolo", action="store_true")
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    # Compare tip detectors on recorded Braille-reading video (hit rate %).
    # skin_contour usually wins when the palm is cropped out of frame.
    from tip_backends import create_tip_backend

    backends = [
        ("mediapipe", create_tip_backend("mediapipe")),
        ("skin_contour", create_tip_backend("skin")),  # live_app default
    ]
    if not args.skip_yolo:
        try:
            backends.append(("tip_yolo", create_tip_backend("yolo")))
        except FileNotFoundError as exc:
            print(f"TipYOLO skipped: {exc}")

    print(f"video: {args.video}")
    print(f"max_frames={args.max_frames}  stride={args.stride}")
    print()
    print(f"{'backend':<14} {'frames':>7} {'hits':>7} {'rate':>8}")

    for name, det in backends:
        cap = cv2.VideoCapture(str(args.video))
        seen = 0

        def detect(frame, _det=det, _name=name):
            if hasattr(_det, "detect"):
                out = _det.detect(frame)
                if isinstance(out, tuple):
                    return out[0]
                return out
            return None

        hits = frames = 0
        while frames < args.max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            seen += 1
            if seen % args.stride:
                continue
            frames += 1
            if detect(frame) is not None:
                hits += 1
        cap.release()
        if hasattr(det, "close"):
            det.close()
        rate = hits / max(frames, 1)
        print(f"{name:<14} {frames:7d} {hits:7d} {rate:8.1%}")
        if name == "mediapipe":
            if rate >= 0.90:
                print("  gate: ship MediaPipe alone; keep contour as --tip-backend")
            elif rate >= 0.60:
                print("  gate: MediaPipe primary, skin-contour automatic fallback")
            else:
                print("  gate: skin-contour primary; MediaPipe only for finger identity")


if __name__ == "__main__":
    main()
