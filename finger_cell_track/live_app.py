"""PC live app: YOLO fingertip + DotNeuralNet CellMap + Learning/Testing.

    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/live_app.py --source 0 --mode learning --lang en
    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/live_app.py --source http://PHONE_IP:8080/video

Keys:
  Q     quit
  R     (re)scan current frame -> CellMap + new registration reference frame
  L / T learning / testing mode
  In testing: after a dwell prompt, type the letter in the terminal and press Enter
              (or press the letter key in the OpenCV window for a–z / 0–9).

R only needs to be pressed once at the start, not every time the page or
camera shifts: every live frame is automatically re-aligned back onto the
scanned reference frame (registration.py, ORB + RANSAC homography), so the
CellMap stays valid under drift on its own. Press R again only to scan a
genuinely different/new page, or as a manual reset if the HUD's `reg=`
status shows LOST for an extended stretch (e.g. camera pointed somewhere
with too little texture to match against, like a blank area of the page).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_DNN = _ROOT / "DotNeuralNet"
for p in (_HERE, _ROOT, _DNN):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

import numpy as np  # noqa: E402

from cell_map import Cell, CellMap, DwellFilter, TipEMA, hit_test  # noqa: E402
from hand_track import open_source  # noqa: E402
from modes import LearningMode, TestingMode  # noqa: E402
from prescan import draw_cellmap, prescan_bgr  # noqa: E402
from registration import FrameRegistration  # noqa: E402
from tip_yolo import TipYOLO  # noqa: E402

_DEFAULT_CELL_WEIGHTS = _DNN / "weights" / "yolov8_braille.pt"


def _cellmap_for_display(cell_map: CellMap, homography: np.ndarray | None) -> CellMap:
    """Cell boxes are stored in the pre-scan's reference frame. Drawing
    them directly onto a live frame that's since drifted would show boxes
    in the wrong place even though hit-testing (which goes through
    registration) is still correct -- so for display only, project each
    box from reference space back into the current live frame via the
    inverse homography. Axis-aligned bounding box of the 4 transformed
    corners (a box doesn't stay a box under a perspective transform, but
    this is a fine approximation for small frame-to-frame drift)."""
    if homography is None or not cell_map.cells:
        return cell_map
    try:
        inv_h = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return cell_map

    out_cells = []
    for c in cell_map.cells:
        x0, y0, x1, y1 = c.xyxy
        corners = np.array([[[x0, y0]], [[x1, y0]], [[x0, y1]], [[x1, y1]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(corners, inv_h).reshape(-1, 2)
        nx0, ny0 = mapped.min(axis=0)
        nx1, ny1 = mapped.max(axis=0)
        out_cells.append(
            Cell(
                id=c.id, xyxy=(float(nx0), float(ny0), float(nx1), float(ny1)),
                char=c.char, pattern=c.pattern, code=c.code, conf=c.conf,
                line=c.line, col=c.col,
            )
        )
    return CellMap(cells=out_cells)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p = argparse.ArgumentParser(description="Finger → Braille cell live Learning/Testing")
    p.add_argument("--source", default="0")
    p.add_argument("--weights", type=Path, default=_DEFAULT_CELL_WEIGHTS, help="Braille cell YOLO")
    p.add_argument("--tip-weights", type=Path, default=None, help="Fingertip YOLO (default: weights/)")
    p.add_argument("--mode", choices=("learning", "testing"), default="learning")
    p.add_argument("--lang", choices=("en", "si"), default="en")
    p.add_argument("--conf", type=float, default=0.25, help="Cell YOLO conf")
    p.add_argument("--tip-conf", type=float, default=0.25, help="Tip YOLO conf")
    p.add_argument("--dwell-ms", type=float, default=400.0)
    p.add_argument("--margin", type=float, default=0.15)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--display-width", type=int, default=960)
    args = p.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"Cell weights not found: {args.weights}")

    from ultralytics import YOLO

    print(f"Loading cell YOLO {args.weights} ...", flush=True)
    yolo = YOLO(str(args.weights))
    print("Loading tip YOLO ...", flush=True)
    tipper = TipYOLO(
        weights=args.tip_weights,
        conf=args.tip_conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    print(f"Tip weights: {tipper.weights}", flush=True)
    print(f"Opening {args.source!r} ...", flush=True)
    cap = open_source(args.source)

    cell_map = CellMap()
    ema = TipEMA(0.35)
    dwell = DwellFilter(args.dwell_ms)
    learn = LearningMode()
    test = TestingMode()
    mode = args.mode
    last_frame = None
    highlight_id = None
    status = "Press R to scan page"
    # Reference-frame registration: maps each live frame's tip position
    # back onto the coordinate frame the CellMap was scanned in, so the
    # map stays valid as the page/camera drifts instead of needing a
    # manual rescan every time. None until the first successful scan.
    registration: FrameRegistration | None = None

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

            tip_raw, tip_box, tip_conf = tipper.detect(frame)
            tip = ema.update(tip_raw)

            # Track the live<->reference homography once per frame (used
            # both to map the tip for hit-testing, and to re-project the
            # CellMap's boxes for display -- without this, hit-testing
            # would silently assume the camera hasn't moved since the last
            # R rescan, and the drawn boxes would visibly lag behind).
            live_h = None
            reg_inliers = 0
            if registration is not None:
                live_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                live_h = registration.homography_or_last(live_gray)
                reg_inliers = registration.last_inliers

            ref_tip = registration.transform_point(tip, live_h) if (tip is not None and live_h is not None) else None
            hit = hit_test(ref_tip, cell_map, margin_frac=args.margin) if ref_tip else None
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

            out = frame.copy()
            if tip_box is not None:
                x1, y1, x2, y2 = tip_box
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
            if cell_map.cells:
                display_map = _cellmap_for_display(cell_map, live_h)
                out = draw_cellmap(out, display_map, highlight_id=highlight_id)
            if tip is not None:
                cv2.circle(out, (int(tip[0]), int(tip[1])), 10, (0, 255, 255), -1)

            reg_status = "-" if registration is None else (f"OK({reg_inliers})" if live_h is not None else "LOST")
            hud1 = (
                f"mode={mode}  cells={len(cell_map)}  reg={reg_status}  "
                f"hit={hit.char if hit else '-'}  tip={'Y' if tip else 'N'}"
                + (f" {tip_conf:.2f}" if tip_raw else "")
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
                # New reference frame for registration -- every subsequent
                # live frame gets aligned back to *this* snapshot, so the
                # CellMap keeps working as the page/camera drifts instead
                # of needing another manual rescan.
                registration = FrameRegistration(cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY))
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
            if mode == "testing" and test.awaiting_answer and key != 255:
                ch = chr(key) if 32 <= key < 127 else ""
                if ch.isalnum():
                    ev = test.submit_answer(ch)
                    print(ev.message, flush=True)
                    status = ev.message
                    dwell.reset()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped.", flush=True)


if __name__ == "__main__":
    main()
