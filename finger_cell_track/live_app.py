"""PC live app: prescanned CellMap + fingertip hit-test (Learning / Testing).

    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/live_app.py --source 0 --lang si

Auto-scan is on by default. Press R only to force a rescan.

Keys:
  Q     quit
  R     (re)scan current frame -> CellMap + new registration reference frame
  L / T learning / testing mode
  In testing: after a dwell prompt, type a-z / 0-9 in the OpenCV window
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_HERE, _ROOT):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

import numpy as np  # noqa: E402

from autoscan import PageWatcher  # noqa: E402
from cell_map import Cell, CellMap, DwellFilter, TipEMA, hit_test  # noqa: E402
from camera_source import open_source  # noqa: E402
from tip_backends import create_tip_backend  # noqa: E402

from modes import LearningMode, TestingMode  # noqa: E402
from prescan import draw_cellmap, prescan_bgr  # noqa: E402
from registration import FrameRegistration  # noqa: E402

_OWN_CELL_WEIGHTS = _ROOT / "cell_detect" / "weights" / "braille_cell_best.pt"
_DNN_WEIGHTS = _ROOT / "experiments" / "DotNeuralNet" / "weights" / "yolov8_braille.pt"


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
    p.add_argument("--weights", type=Path, default=None, help="Cell YOLO weights (own or DotNeuralNet)")
    p.add_argument("--checkpoint", type=Path, default=None, help="CNN checkpoint for recognize_page")
    p.add_argument(
        "--scan-backend",
        choices=("auto", "cells", "dots", "dnn"),
        default="cells",
        help="cells = our YOLO cell detector + CNN (default once weights exist)",
    )
    p.add_argument(
        "--tip-weights",
        type=Path,
        default=None,
        help="Fingertip YOLO26 weights (default: yolo26n_fingertip_braille_best.pt)",
    )
    # Tip detector: YOLO26 default, SkinContourTip if YOLO misses the frame.
    p.add_argument(
        "--tip-backend",
        choices=("auto", "yolo", "skin", "mediapipe"),
        default="auto",
        help="auto = YOLO26 then SkinContourTip fallback (default); yolo/skin/mediapipe = that detector only",
    )
    p.add_argument(
        "--auto-scan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture the page when the camera is still and no hand is visible (default on).",
    )
    p.add_argument("--mode", choices=("learning", "testing"), default="learning")
    p.add_argument("--lang", choices=("en", "si"), default="si")
    p.add_argument("--conf", type=float, default=0.25, help="Cell YOLO conf")
    p.add_argument("--tip-conf", type=float, default=0.25, help="Tip YOLO conf")
    p.add_argument(
        "--dwell-ms",
        type=float,
        default=3000.0,
        help="Hold the same cell this many ms before printing it (default 3000 = 3 s)",
    )
    p.add_argument("--margin", type=float, default=0.15)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="cpu")
    p.add_argument("--display-width", type=int, default=960)
    p.add_argument(
        "--no-window",
        action="store_true",
        help="No OpenCV window (use with a video file for a terminal-only run)",
    )
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = unlimited)")
    p.add_argument(
        "--force-scan",
        action="store_true",
        help="Prescan the first frame immediately (skip waiting for a hand-free still)",
    )
    args = p.parse_args()

    if args.scan_backend == "dnn":
        weights = args.weights or _DNN_WEIGHTS
        if not weights.exists():
            raise SystemExit(f"DotNeuralNet weights not found: {weights}")
        from ultralytics import YOLO

        print(f"Loading DotNeuralNet {weights} ...", flush=True)
        yolo = YOLO(str(weights))
    else:
        yolo = None
        print(f"Scan backend={args.scan_backend} (recognize_page; dnn only if own weights missing)", flush=True)

    # --- Build tip detector via tip_backends (auto = YOLO26 + skin fallback) ---
    print(f"Tip backend={args.tip_backend} ...", flush=True)
    tipper = create_tip_backend(
        args.tip_backend,
        tip_weights=args.tip_weights,
        tip_conf=args.tip_conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    yolo_det = getattr(tipper, "primary", tipper)
    if hasattr(yolo_det, "weights"):
        print(f"Tip weights: {yolo_det.weights}", flush=True)
    print(f"Opening {args.source!r} ...", flush=True)
    cap = open_source(args.source)
    source_is_file = Path(str(args.source)).is_file()

    cell_map = CellMap()
    # TipEMA smooths YOLO / SkinContourTip jitter and rejects teleports (cell_map.py).
    ema = TipEMA(alpha=0.35, max_jump_px=180.0, lost_frames_to_retarget=8, coast_frames=5)
    dwell = DwellFilter(args.dwell_ms)
    learn = LearningMode()
    test = TestingMode()
    mode = args.mode
    last_frame = None
    highlight_id = None
    n_frames = 0
    pending_force_scan = bool(args.force_scan)
    status = "Hold still over the page" if args.auto_scan else "Press R to scan page"
    print(
        f"Dwell={args.dwell_ms:.0f} ms — cell is printed only after holding that long.",
        flush=True,
    )

    def _do_scan(frame):
        return prescan_bgr(
            frame,
            yolo,
            conf=args.conf,
            lang=args.lang,
            imgsz=args.imgsz,
            device=args.device,
            backend=args.scan_backend,
            cell_weights=args.weights,
            cnn_checkpoint=args.checkpoint,
        )

    # Force-scan owns capture; the watcher would otherwise drop the map
    # as soon as the finger (and camera motion) appears.
    watcher = PageWatcher(scan_fn=_do_scan) if (args.auto_scan and not args.force_scan) else None
    # Reference-frame registration: maps each live frame's tip position
    # back onto the coordinate frame the CellMap was scanned in, so the
    # map stays valid as the page/camera drifts instead of needing a
    # manual rescan every time. None until the first successful scan.
    registration: FrameRegistration | None = None

    win = "finger_cell_track — Live (Q quit, R rescan, L/T mode)"
    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if args.auto_scan and not pending_force_scan:
        print(f"Mode={mode}. Auto-scan on. Hold still, hand off the page. R = force rescan. Q quit.", flush=True)
    elif pending_force_scan:
        print(f"Mode={mode}. Force-scan on first frame. YOLO26 tip (skin fallback) → cell char on the terminal.", flush=True)
    else:
        print(f"Mode={mode}. Press R over a Braille page to build CellMap. Q quit.", flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if source_is_file:
                    print("End of video.", flush=True)
                    break
                print("Frame grab failed.", flush=True)
                time.sleep(0.2)
                continue
            last_frame = frame
            n_frames += 1
            if args.max_frames and n_frames > args.max_frames:
                print(f"Reached --max-frames {args.max_frames}.", flush=True)
                break

            if pending_force_scan:
                print("Scanning page (first frame)...", flush=True)
                cell_map = _do_scan(frame)
                registration = FrameRegistration(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                registration.assume_identity()
                dwell.reset()
                ema.reset()
                learn = LearningMode()
                test = TestingMode()
                status = f"Scanned {len(cell_map)} cells"
                print("", flush=True)
                print("=" * 56, flush=True)
                print(f"[PAGE] CAPTURED OK — {len(cell_map)} cells (force-scan)", flush=True)
                print(
                    f"Hold finger on a cell for {args.dwell_ms / 1000:.0f}s to print it.",
                    flush=True,
                )
                print("=" * 56, flush=True)
                print("", flush=True)
                if watcher is not None:
                    watcher.cell_map = cell_map
                    watcher.state = "TRACKING"
                    watcher._last_capture_t = time.time()
                    watcher._lost_since = None
                    watcher._tracking_since = time.time()
                    watcher._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    watcher._last_kind = "captured"
                pending_force_scan = False

            # Per-frame tip: YOLO26 (SkinContour fallback) → TipEMA → homography + hit_test.
            tip_raw, tip_box, tip_conf = tipper.detect(frame)
            tip = ema.update(tip_raw)
            hand_visible = bool(getattr(tipper, "hand_visible", tip_raw is not None))

            if watcher is not None and (cell_map is None or len(cell_map) == 0 or watcher.state != "TRACKING"):
                ev = watcher.update(
                    frame,
                    hand_visible=hand_visible,
                    registration_lost=False,
                )
                if ev:
                    print(ev.message, flush=True)
                    status = ev.message
                if ev and ev.kind == "captured" and watcher.cell_map is not None:
                    cell_map = watcher.cell_map
                    registration = FrameRegistration(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    registration.assume_identity()
                    dwell.reset()
                    ema.reset()
                    learn = LearningMode()
                    test = TestingMode()
                    print("", flush=True)
                    print("=" * 56, flush=True)
                    print(
                        f"[PAGE] AUTO-CAPTURE OK — {len(cell_map)} cells ready",
                        flush=True,
                    )
                    print(
                        f"Hold finger on a cell for {args.dwell_ms / 1000:.0f}s to print it.",
                        flush=True,
                    )
                    print("=" * 56, flush=True)
                    print("", flush=True)
                    status = f"Page captured ({len(cell_map)} cells) — place finger"

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
                if watcher is not None:
                    lost_ev = watcher.update(
                        frame,
                        hand_visible=hand_visible,
                        registration_lost=registration.status == "LOST",
                    )
                    if lost_ev and lost_ev.kind == "lost":
                        print(lost_ev.message, flush=True)
                        status = lost_ev.message
                        cell_map = CellMap()
                        registration = None

            ref_tip = None
            if tip is not None:
                if registration is not None and live_h is not None:
                    ref_tip = FrameRegistration.transform_point(tip, live_h)
                elif registration is not None:
                    ref_tip = (float(tip[0]), float(tip[1]))
            hit = hit_test(ref_tip, cell_map, margin_frac=args.margin) if ref_tip else None
            highlight_id = hit.id if hit else None

            if hit is None:
                if mode == "learning":
                    learn.on_leave()
                else:
                    test.on_leave()
                dwell.update(None)
            else:
                # Print only after the finger stays on the same cell for dwell_ms.
                fired = dwell.update(hit)
                if fired is not None:
                    print(
                        f"[CELL] id={fired.id}  L{fired.line}C{fired.col}  "
                        f"code={fired.code}  char={fired.char!r}  conf={fired.conf:.2f}",
                        flush=True,
                    )
                    if mode == "learning":
                        ev = learn.on_dwell(fired)
                    else:
                        ev = test.on_dwell(fired)
                    if ev:
                        print(ev.message, flush=True)
                        status = ev.message

            if args.no_window:
                continue

            out = frame.copy()
            if tip_box is not None:
                x1, y1, x2, y2 = tip_box
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
            if cell_map.cells:
                display_map = _cellmap_for_display(cell_map, live_h)
                out = draw_cellmap(out, display_map, highlight_id=highlight_id)
            if tip is not None:
                cv2.circle(out, (int(tip[0]), int(tip[1])), 10, (0, 255, 255), -1)

            if registration is None:
                reg_status = "-"
            else:
                reg_status = f"{registration.status}({reg_inliers})"
            page_state = watcher.state if watcher is not None else "manual"
            tip_src = getattr(tipper, "last_backend", args.tip_backend)
            hud1 = (
                f"mode={mode}  cells={len(cell_map)}  page={page_state}  reg={reg_status}  "
                f"hit={hit.char if hit else '-'}  tip={'Y' if tip else 'N'}"
                + (f" {tip_conf:.2f}" if tip_raw else "")
                + (f" [{tip_src}]" if tip else "")
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
                cell_map = _do_scan(last_frame)
                # New reference frame for registration -- every subsequent
                # live frame gets aligned back to *this* snapshot, so the
                # CellMap keeps working as the page/camera drifts instead
                # of needing another manual rescan.
                registration = FrameRegistration(cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY))
                registration.assume_identity()
                dwell.reset()
                ema.reset()
                learn = LearningMode()
                test = TestingMode()
                status = f"Scanned {len(cell_map)} cells"
                print("", flush=True)
                print("=" * 56, flush=True)
                print(f"[PAGE] CAPTURED OK — {len(cell_map)} cells (manual R)", flush=True)
                print(
                    f"Hold finger on a cell for {args.dwell_ms / 1000:.0f}s to print it.",
                    flush=True,
                )
                print("=" * 56, flush=True)
                print("", flush=True)
                if watcher is not None:
                    watcher.cell_map = cell_map
                    watcher.state = "TRACKING"
                    watcher._last_capture_t = time.time()
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
        if hasattr(tipper, "close"):
            tipper.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped.", flush=True)


if __name__ == "__main__":
    main()
