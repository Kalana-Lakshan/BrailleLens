"""Build a CellMap from an image/frame via DotNeuralNet + BrailleLens decode.

From repo root (venv with ultralytics + torch):

    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/prescan.py --image path.jpg --lang en
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_DNN = _ROOT / "experiments" / "DotNeuralNet"
for p in (_HERE, _ROOT, _DNN):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from cell_map import Cell, CellMap  # noqa: E402

_DEFAULT_WEIGHTS = _DNN / "weights" / "yolov8_braille.pt"


def _decode_char(code: int, lang: str) -> str:
    from braille_cnn.labels import code_to_label, decode_sequence

    if lang == "en":
        text = decode_sequence([code], lang="en")
        return text if text else code_to_label(code, lang="en")
    return code_to_label(code, lang="si")


def cells_from_yolo_result(boxes, names, lang: str = "en") -> CellMap:
    from src.layout import boxes_to_cells, group_into_lines

    raw = boxes_to_cells(boxes, names)
    lines = group_into_lines(raw)
    out: list[Cell] = []
    cid = 0
    for li, line in enumerate(lines):
        for col, c in enumerate(line):
            code = int(c["code"])
            char = _decode_char(code, lang)
            if not char or str(char).startswith("["):
                char = c.get("pattern") or str(code)
            out.append(
                Cell(
                    id=cid,
                    xyxy=tuple(float(v) for v in c["xyxy"]),
                    char=str(char),
                    pattern=str(c.get("pattern", "")),
                    code=code,
                    conf=float(c.get("conf", 1.0)),
                    line=li,
                    col=col,
                )
            )
            cid += 1
    return CellMap(cells=out)


def prescan_bgr(
    image_bgr: np.ndarray,
    model,
    *,
    conf: float = 0.25,
    lang: str = "en",
    imgsz: int = 640,
    device: str = "cpu",
) -> CellMap:
    res = model.predict(
        source=image_bgr,
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    r0 = res[0]
    return cells_from_yolo_result(r0.boxes, r0.names, lang=lang)


def draw_cellmap(
    bgr: np.ndarray,
    cell_map: CellMap,
    *,
    highlight_id: int | None = None,
) -> np.ndarray:
    out = bgr.copy()
    for c in cell_map.cells:
        x0, y0, x1, y1 = (int(round(v)) for v in c.xyxy)
        color = (0, 0, 255) if c.id != highlight_id else (0, 255, 255)
        thick = 2 if c.id == highlight_id else 1
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thick)
        label = (c.char or "?")[:6]
        cv2.putText(
            out,
            label,
            (x0, max(y0 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p = argparse.ArgumentParser(description="DotNeuralNet image → CellMap")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--weights", type=Path, default=_DEFAULT_WEIGHTS)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--lang", choices=("en", "si"), default="en")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save-overlay", type=Path, default=None)
    args = p.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")
    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}")

    from ultralytics import YOLO

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise SystemExit(f"Failed to read {args.image}")

    model = YOLO(str(args.weights))
    cell_map = prescan_bgr(
        bgr,
        model,
        conf=args.conf,
        lang=args.lang,
        imgsz=args.imgsz,
        device=args.device,
    )
    print(f"cells={len(cell_map)}  lang={args.lang}")
    for c in cell_map.cells[:40]:
        print(f"  id={c.id:3d} L{c.line}C{c.col} char={c.char!r} pat={c.pattern}")
    if len(cell_map) > 40:
        print(f"  ... +{len(cell_map) - 40} more")

    if args.save_overlay:
        args.save_overlay.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_overlay), draw_cellmap(bgr, cell_map))
        print(f"overlay -> {args.save_overlay}")


if __name__ == "__main__":
    main()
