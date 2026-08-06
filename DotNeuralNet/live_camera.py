"""Live IP Webcam / webcam preview with DotNeuralNet → Sinhala.

Run from the BrailleLens project root (Python 3.11):

    py -3.11 DotNeuralNet/live_camera.py
    py -3.11 DotNeuralNet/live_camera.py --source http://192.168.8.126:8080/video

Controls (preview window):
    Q       quit
    S       force inference on current frame
    D       toggle box drawing
    T       toggle Sinhala text panel
    + / -   raise / lower confidence (step 0.05)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parent.parent
_DNN = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_DNN) not in sys.path:
    sys.path.insert(0, str(_DNN))

_DEFAULT_WEIGHTS = _DNN / "weights" / "yolov8_braille.pt"
_DEFAULT_SOURCE = "http://192.168.8.126:8080/video"
_BOX_COLOR = (0, 0, 255)  # BGR red
_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\Nirmala.ttc"),
    Path(r"C:\Windows\Fonts\iskpota.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSansSinhala-Regular.ttf"),
)


def _open_source(source: str) -> cv2.VideoCapture:
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
    except (ValueError, TypeError):
        cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera source: {source!r}\n"
            "For IP Webcam use: http://PHONE_IP:8080/video\n"
            "Phone and PC must be on the same Wi-Fi."
        )
    return cap


def _fit(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    scale = max_width / w
    return (
        cv2.resize(
            frame,
            (max_width, max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        ),
        scale,
    )


def _load_sinhala_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_sinhala_panel(
    preview_bgr: np.ndarray,
    text: str,
    *,
    panel_height: int,
    font: ImageFont.ImageFont,
) -> np.ndarray:
    """Stack a dark panel under the preview with PIL-rendered Sinhala lines."""
    w = preview_bgr.shape[1]
    panel = Image.new("RGB", (w, panel_height), (18, 18, 22))
    draw = ImageDraw.Draw(panel)
    y = 8
    lines = (text or "(no text yet — hold page steady)").splitlines() or [""]
    for line in lines[:12]:
        draw.text((12, y), line, font=font, fill=(240, 240, 245))
        y += 28
        if y > panel_height - 28:
            break
    panel_bgr = cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)
    return np.vstack([preview_bgr, panel_bgr])


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p = argparse.ArgumentParser(description="Live DotNeuralNet Braille → Sinhala")
    p.add_argument(
        "--source",
        default=_DEFAULT_SOURCE,
        help=f"IP Webcam URL or webcam index (default: {_DEFAULT_SOURCE})",
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=_DEFAULT_WEIGHTS,
        help="Path to yolov8_braille.pt",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Below this, decode emits '_' instead of a glyph",
    )
    p.add_argument("--lang", choices=("si", "en"), default="si")
    p.add_argument(
        "--infer-interval",
        type=float,
        default=1.0,
        help="Seconds between YOLO + Sinhala decode runs",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="cpu", help="cpu or 0 for CUDA")
    p.add_argument("--display-width", type=int, default=960)
    p.add_argument("--panel-height", type=int, default=220)
    p.add_argument(
        "--labels",
        action="store_true",
        help="Draw pattern/conf on each box (can clutter)",
    )
    args = p.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}")

    from ultralytics import YOLO
    from transcribe_image import transcribe

    print(f"Loading {args.weights} ...", flush=True)
    model = YOLO(str(args.weights))
    print(f"Opening {args.source} ...", flush=True)
    cap = _open_source(args.source)
    print("Camera opened. Waiting for first frame...", flush=True)

    font = _load_sinhala_font(22)
    conf = args.conf
    show_boxes = True
    show_text = True
    show_labels = args.labels
    last_cells: list = []
    last_n = 0
    last_n_lines = 0
    last_sentence = ""
    last_infer_t = 0.0
    last_infer_ms = 0.0
    force = False

    win = "DotNeuralNet — Live Sinhala (Q quit, S infer, D boxes, T text)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("Live. Press Q to quit. Sinhala also prints here each inference.", flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame grab failed — check IP Webcam URL / Wi-Fi.", flush=True)
                time.sleep(0.3)
                continue

            now = time.time()
            should_infer = force or (now - last_infer_t) >= args.infer_interval
            if should_infer:
                force = False
                t0 = time.time()
                result = transcribe(
                    frame,
                    model,
                    conf=conf,
                    conf_threshold=args.conf_threshold,
                    lang=args.lang,
                    imgsz=args.imgsz,
                    device=args.device,
                )
                last_infer_ms = (time.time() - t0) * 1000.0
                last_infer_t = now
                last_cells = result["cells"]
                last_n = result["n_cells"]
                last_n_lines = result["n_lines"]
                last_sentence = result["sentence"]
                print(
                    f"\n--- cells={last_n} lines={last_n_lines} "
                    f"({last_infer_ms:.0f}ms) ---",
                    flush=True,
                )
                print(last_sentence or "(empty)", flush=True)

            preview, scale = _fit(frame, args.display_width)
            if show_boxes and last_cells:
                for cell in last_cells:
                    x0, y0, x1, y1 = cell["xyxy"]
                    sx0, sy0 = int(round(x0 * scale)), int(round(y0 * scale))
                    sx1, sy1 = int(round(x1 * scale)), int(round(y1 * scale))
                    cv2.rectangle(preview, (sx0, sy0), (sx1, sy1), _BOX_COLOR, 2)
                    if show_labels:
                        label = f"{cell.get('pattern', '')} {cell['conf']:.2f}"
                        cv2.putText(
                            preview,
                            label,
                            (sx0, max(sy0 - 2, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            _BOX_COLOR,
                            1,
                            cv2.LINE_AA,
                        )

            hud = (
                f"cells={last_n}  lines={last_n_lines}  conf={conf:.2f}  "
                f"infer={last_infer_ms:.0f}ms"
            )
            cv2.putText(
                preview,
                hud,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 120),
                2,
                cv2.LINE_AA,
            )

            display = (
                _draw_sinhala_panel(
                    preview,
                    last_sentence,
                    panel_height=args.panel_height,
                    font=font,
                )
                if show_text
                else preview
            )

            cv2.imshow(win, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                force = True
            if key in (ord("d"), ord("D")):
                show_boxes = not show_boxes
            if key in (ord("t"), ord("T")):
                show_text = not show_text
            if key in (ord("+"), ord("=")):
                conf = min(0.95, conf + 0.05)
            if key in (ord("-"), ord("_")):
                conf = max(0.05, conf - 0.05)
            if key in (ord("l"), ord("L")):
                show_labels = not show_labels
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped.", flush=True)


if __name__ == "__main__":
    main()
