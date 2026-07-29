"""Live camera capture loop for BrailleLens.

Opens a webcam or IP Webcam stream, converts frames to grayscale PIL Images,
runs a frame-stability check, and feeds stable frames into run_auto_transcribe.

Controls:
    q   quit
    s   force inference on current frame (bypasses stability check once)
    d   toggle detection boxes on preview
"""

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from braille_cnn.infer_page import load_model, run_auto_transcribe  # noqa: E402


# ------------------------------------------------------------------ helpers

def _open_source(source) -> cv2.VideoCapture:
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
    except (ValueError, TypeError):
        cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera source: {source!r}\n"
            "For IP Webcam, make sure the app is running and the URL is reachable "
            "(e.g. http://192.168.1.x:8080/video)."
        )
    return cap


def _frame_to_gray_pil(bgr_frame: np.ndarray) -> Image.Image:
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    return Image.fromarray(gray)


def _motion_score(prev: Optional[np.ndarray], curr: np.ndarray) -> float:
    if prev is None:
        return 0.0
    return float(cv2.absdiff(prev, curr).mean())


def _fit_for_display(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    scale = max_width / w
    new_w = max_width
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def _scale_box(box, scale: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (
        int(round(x0 * scale)),
        int(round(y0 * scale)),
        int(round(x1 * scale)),
        int(round(y1 * scale)),
    )


def _draw_detections(
    preview: np.ndarray,
    result: Optional[dict],
    scale: float,
    show_detections: bool,
) -> None:
    if not show_detections or result is None:
        return

    for c in result["clusters"]:
        if c["merged"]:
            color = (0, 0, 255)
        else:
            color = (0, 200, 0)
        x0, y0, x1, y1 = c["bbox"]
        sx0, sy0, sx1, sy1 = _scale_box((x0, y0, x1, y1), scale)
        cv2.rectangle(preview, (sx0, sy0), (sx1, sy1), color, 1)

    for box in result["boxes"]:
        sx0, sy0, sx1, sy1 = _scale_box(box, scale)
        cv2.rectangle(preview, (sx0, sy0), (sx1, sy1), (255, 180, 0), 1)


def _overlay_status_lines(result: Optional[dict], motion: float) -> list[str]:
    if result is None:
        return ["(waiting for stable frame …)", "Hold camera over Braille page"]
    lines = result["lines"]
    if not lines:
        return ["(no braille lines detected)", "Try better light / press S"]
    avg_conf = float(result["confidences"].mean()) if len(result["confidences"]) else 0.0
    return [
        f"cells: {result['num_valid_cells']}  lines: {len(lines)}  conf: {avg_conf:.2f}",
        f"motion: {motion:.1f}",
        "Sinhala -> terminal (live update below)",
    ]


def _print_live_output(result: dict, verbose: bool) -> None:
    """Refresh Sinhala transcription in-place in the terminal."""
    sentence = result["sentence"].replace("\n", " | ")
    stats = (
        f"[{time.strftime('%H:%M:%S')}] "
        f"cells={result['num_valid_cells']} lines={len(result['lines'])} "
        f"dots={result['num_dots']}"
    )

    if verbose:
        print(f"\n{stats}")
        for line in result["lines"]:
            print(f"  {line}")
        return

    # In-place refresh: stats on one line, sentence on the next
    width = 100
    sys.stdout.write("\r" + stats.ljust(width))
    sys.stdout.write("\n\r" + sentence[: width - 1].ljust(width))
    sys.stdout.write("\033[1A")  # move cursor back up one line
    sys.stdout.flush()


def _draw_overlay(
    frame: np.ndarray,
    status: str,
    fps: float,
    last_lines: list[str],
    frame_size: Optional[tuple[int, int]] = None,
    motion: float = 0.0,
    stable_streak: int = 0,
) -> np.ndarray:
    display = frame.copy()
    h, w = display.shape[:2]

    bar_color = (0, 180, 0) if status == "STABLE" else (0, 80, 200)
    cv2.rectangle(display, (0, 0), (w, 36), bar_color, -1)

    status_text = f"{status}   FPS: {fps:.1f}   motion: {motion:.1f}   stable: {stable_streak}"
    if frame_size is not None:
        status_text += f"   src: {frame_size[0]}x{frame_size[1]}"
    cv2.putText(display, status_text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    y = 56
    for line in last_lines[-4:]:
        cv2.putText(display, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1, cv2.LINE_AA)
        y += 20

    cv2.putText(display, "Q quit | S infer | D toggle boxes",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return display


# ----------------------------------------------------------------- main loop

def run_camera(args) -> None:
    source = getattr(args, "source", 0)
    motion_threshold: float = getattr(args, "motion_threshold", 8.0)
    preview_only: bool = getattr(args, "preview_only", False)
    display_width: int = getattr(args, "display_width", 960)
    infer_interval: float = getattr(args, "infer_interval", 1.5)
    stable_frames_required: int = getattr(args, "stable_frames", 8)
    verbose: bool = getattr(args, "verbose", False)

    model = None
    device = None
    if preview_only:
        print("Preview-only mode: camera feed will open without loading a model.")
    else:
        print(f"Loading model from {args.checkpoint} …")
        if not Path(args.checkpoint).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {args.checkpoint}\n"
                "Train first: py -3.11 -m braille_cnn.finetune_dbsi --scratch --dbsi-root \"data DBSI/data\""
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(args.checkpoint, device)
        print(f"Model loaded on {device}.")
        if not verbose:
            print("\nLive Sinhala transcription (updates in place — use --verbose for full log):\n")

    print(f"Opening camera source: {source!r} …")
    cap = _open_source(source)
    print("Camera opened.")
    cv2.namedWindow("BrailleLens — Live Camera", cv2.WINDOW_NORMAL)

    prev_gray: Optional[np.ndarray] = None
    logged_resolution = False
    last_infer_time: float = 0.0
    stable_streak = 0
    last_result: Optional[dict] = None
    last_lines: list[str] = ["(preview only)" if preview_only else "(waiting for stable frame …)"]
    force_infer = False
    show_detections = True

    fps_counter = 0
    fps_clock = time.time()
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nWARNING: failed to grab frame — retrying …")
            time.sleep(0.05)
            continue

        if not logged_resolution:
            h, w = frame.shape[:2]
            print(f"Camera frame size: {w}x{h} (preview max width {display_width}px)")
            logged_resolution = True

        gray_np = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = _motion_score(prev_gray, gray_np)
        prev_gray = gray_np
        now = time.time()

        fps_counter += 1
        if now - fps_clock >= 1.0:
            fps_display = fps_counter / (now - fps_clock)
            fps_counter = 0
            fps_clock = now

        is_stable = motion <= motion_threshold
        if is_stable:
            stable_streak += 1
        else:
            stable_streak = 0

        status = "STABLE" if stable_streak >= stable_frames_required else "MOVING"

        should_infer = (
            not preview_only
            and (
                force_infer
                or (
                    stable_streak >= stable_frames_required
                    and (now - last_infer_time) >= infer_interval
                )
            )
        )
        force_infer = False

        if should_infer:
            try:
                pil_image = _frame_to_gray_pil(frame)
                last_result = run_auto_transcribe(pil_image, args, model=model, device=device)
                last_lines = _overlay_status_lines(last_result, motion)
                _print_live_output(last_result, verbose=verbose)
            except Exception as exc:
                last_lines = [f"[inference error] {exc}"]
                if verbose:
                    print(f"\n[inference error] {exc}")
            last_infer_time = now

        src_h, src_w = frame.shape[:2]
        preview, scale = _fit_for_display(frame, display_width)
        _draw_detections(preview, last_result, scale, show_detections)
        display = _draw_overlay(
            preview, status, fps_display, last_lines,
            frame_size=(src_w, src_h), motion=motion, stable_streak=stable_streak,
        )
        cv2.imshow("BrailleLens — Live Camera", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            force_infer = True
        if key == ord("d"):
            show_detections = not show_detections

    cap.release()
    cv2.destroyAllWindows()
    if not verbose:
        sys.stdout.write("\n\n")
    if last_result and last_result["sentence"].strip():
        print("Final transcription:")
        print(last_result["sentence"])
    print("Camera closed.")
