"""PC live app: MediaPipe tip + DotNeuralNet CellMap + Learning/Testing.

    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/live_app.py --source 0 --mode learning --lang en

Keys:
  Q     quit
  R     rescan current frame → CellMap
  L / T learning / testing mode
  In testing: after a dwell prompt, type the letter in the terminal and press Enter
              (or press the letter key in the OpenCV window for a–z / 0–9).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_DNN = _ROOT / "DotNeuralNet"
for p in (_HERE, _ROOT, _DNN):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from cell_map import CellMap, DwellFilter, TipEMA, hit_test  # noqa: E402
from hand_track import INDEX_FINGERTIP, open_source, process_frame  # noqa: E402
from modes import LearningMode, TestingMode  # noqa: E402
from prescan import draw_cellmap, prescan_bgr  # noqa: E402

_DEFAULT_WEIGHTS = _DNN / "weights" / "yolov8_braille.pt"
_mp_hands = mp.solutions.hands


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p = argparse.ArgumentParser(description="Finger → Braille cell live Learning/Testing")
    p.add_argument("--source", default="0")
    p.add_argument("--weights", type=Path, default=_DEFAULT_WEIGHTS)
    p.add_argument("--mode", choices=("learning", "testing"), default="learning")
    p.add_argument("--lang", choices=("en", "si"), default="en")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--dwell-ms", type=float, default=400.0)
    p.add_argument("--margin", type=float, default=0.15)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--display-width", type=int, default=960)
    args = p.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}")

    from ultralytics import YOLO

    print(f"Loading YOLO {args.weights} ...", flush=True)
    yolo = YOLO(str(args.weights))
    print(f"Opening {args.source!r} ...", flush=True)
    cap = open_source(args.source)
    hands = _mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    cell_map = CellMap()
    ema = TipEMA(0.35)
    dwell = DwellFilter(args.dwell_ms)
    learn = LearningMode()
    test = TestingMode()
    mode = args.mode
    last_frame = None
    highlight_id = None
    status = "Press R to scan page"

    win = "finger_cell_track — Live (Q quit, R rescan, L/T mode)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(
        f"Mode={mode}. Press R over a Braille page to build CellMap. Q quit.",
        flush=True,
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame grab failed.", flush=True)
                time.sleep(0.2)
                continue
            last_frame = frame

            out, tip_raw = process_frame(frame.copy(), hands)
            tip = ema.update(tip_raw)
            hit = hit_test(tip, cell_map, margin_frac=args.margin) if tip else None
            highlight_id = hit.id if hit else None

            if hit is None:
                if mode == "learning":
                    learn.on_leave()
                else:
                    test.on_leave()
                dwell.update(None)
            else:
                fired = dwell.update(hit)
                if fired is not None:
                    if mode == "learning":
                        ev = learn.on_dwell(fired)
                    else:
                        ev = test.on_dwell(fired)
                    if ev:
                        print(ev.message, flush=True)
                        status = ev.message

            if cell_map.cells:
                out = draw_cellmap(out, cell_map, highlight_id=highlight_id)
            if tip is not None:
                cv2.circle(out, (int(tip[0]), int(tip[1])), 10, (0, 255, 255), -1)

            hud1 = (
                f"mode={mode}  cells={len(cell_map)}  "
                f"hit={hit.char if hit else '-'}  tip={'Y' if tip else 'N'}"
            )
            cv2.putText(
                out, hud1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2
            )
            cv2.putText(
                out,
                status[:80],
                (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 255),
                1,
            )

            h, w = out.shape[:2]
            if w > args.display_width:
                scale = args.display_width / w
                out = cv2.resize(
                    out,
                    (args.display_width, int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            cv2.imshow(win, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                print("Scanning page...", flush=True)
                status = "Scanning..."
                cell_map = prescan_bgr(
                    last_frame,
                    yolo,
                    conf=args.conf,
                    lang=args.lang,
                    imgsz=args.imgsz,
                    device=args.device,
                )
                dwell.reset()
                ema.reset()
                learn = LearningMode()
                test = TestingMode()
                status = f"Scanned {len(cell_map)} cells"
                print(status, flush=True)
            if key in (ord("l"), ord("L")):
                mode = "learning"
                status = "Learning mode"
                print(status, flush=True)
            if key in (ord("t"), ord("T")):
                mode = "testing"
                status = "Testing mode"
                print(status, flush=True)
            # Single-key answers a–z / 0–9 while testing prompt is active
            if mode == "testing" and test.awaiting_answer and key != 255:
                ch = chr(key) if 32 <= key < 127 else ""
                if ch.isalnum():
                    ev = test.submit_answer(ch)
                    print(ev.message, flush=True)
                    status = ev.message
                    dwell.reset()
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped.", flush=True)


if __name__ == "__main__":
    main()
