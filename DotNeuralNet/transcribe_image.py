"""Transcribe a Braille page photo with DotNeuralNet → Sinhala (Parts 1+2).

Pipeline:
  YOLO cells → Otsu line grouping → word gaps → Sinhala decode_sequence

From BrailleLens repo root:

    py -3.11 DotNeuralNet/transcribe_image.py --image path/to/page.jpg
    py -3.11 DotNeuralNet/transcribe_image.py --image path/to/page.jpg --lang en --conf 0.2
    py -3.11 DotNeuralNet/transcribe_image.py --image path/to/page.jpg --save-overlay out.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
_DNN = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_DNN) not in sys.path:
    sys.path.insert(0, str(_DNN))

_DEFAULT_WEIGHTS = _DNN / "weights" / "yolov8_braille.pt"


def _decode_line(codes: list[int], confs: list[float], lang: str, conf_threshold: float) -> str:
    from src.sinhala_bridge import decode_patterns

    # decode_patterns accepts codes as ints via pattern_to_code
    return decode_patterns(
        codes,
        lang=lang,
        confidences=confs,
        conf_threshold=conf_threshold,
    )


def transcribe(
    image_bgr,
    model,
    *,
    conf: float = 0.25,
    conf_threshold: float = 0.25,
    lang: str = "si",
    imgsz: int = 640,
    device: str = "cpu",
) -> dict:
    from src.layout import boxes_to_cells, codes_with_word_spaces, group_into_lines

    res = model.predict(
        source=image_bgr,
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    r0 = res[0]
    cells = boxes_to_cells(r0.boxes, r0.names)
    lines = group_into_lines(cells)

    text_lines = []
    for line in lines:
        codes, confs = codes_with_word_spaces(line)
        text_lines.append(_decode_line(codes, confs, lang, conf_threshold))

    sentence = "\n".join(text_lines)
    return {
        "cells": cells,
        "lines": lines,
        "text_lines": text_lines,
        "sentence": sentence,
        "n_cells": len(cells),
        "n_lines": len(lines),
    }


def _draw_overlay(bgr, result: dict) -> any:
    out = bgr.copy()
    palette = [
        (0, 200, 80),
        (255, 140, 0),
        (200, 80, 255),
        (80, 180, 255),
        (0, 220, 220),
    ]
    for li, line in enumerate(result["lines"]):
        color = palette[li % len(palette)]
        for cell in line:
            x0, y0, x1, y1 = (int(round(v)) for v in cell["xyxy"])
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 1)
        if line:
            cx = int(round(line[0]["center"][0]))
            cy = int(round(line[0]["center"][1]))
            cv2.putText(
                out,
                f"L{li + 1}",
                (cx, max(cy - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    cv2.putText(
        out,
        f"cells={result['n_cells']} lines={result['n_lines']}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 120),
        2,
        cv2.LINE_AA,
    )
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p = argparse.ArgumentParser(description="DotNeuralNet page → Sinhala transcription")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--weights", type=Path, default=_DEFAULT_WEIGHTS)
    p.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence")
    p.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Below this, emit '_' instead of a Sinhala glyph",
    )
    p.add_argument("--lang", choices=("si", "en"), default="si")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--save-overlay", type=Path, default=None)
    args = p.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")
    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}")

    from ultralytics import YOLO

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    model = YOLO(str(args.weights))
    result = transcribe(
        bgr,
        model,
        conf=args.conf,
        conf_threshold=args.conf_threshold,
        lang=args.lang,
        imgsz=args.imgsz,
        device=args.device,
    )

    print(f"cells={result['n_cells']}  lines={result['n_lines']}  lang={args.lang}")
    print("=" * 40)
    print(result["sentence"])
    print("=" * 40)

    if args.save_overlay:
        args.save_overlay.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_overlay), _draw_overlay(bgr, result))
        print(f"overlay -> {args.save_overlay}")


if __name__ == "__main__":
    main()
