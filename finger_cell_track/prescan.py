"""Build a CellMap from a camera frame.

Primary path: braille_cnn.recognize.recognize_page (Stage 4e).
Fallback: experiments/DotNeuralNet, used only when our cell weights are missing.

    py -3.11 -m finger_cell_track.prescan --image test-img.jpeg --lang si
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cell_map import Cell, CellMap  # noqa: E402

_OWN_CELL_WEIGHTS = _ROOT / "cell_detect" / "weights" / "braille_cell_best.pt"
_DNN_WEIGHTS = _ROOT / "experiments" / "DotNeuralNet" / "weights" / "yolov8_braille.pt"


def _decode_char(code: int, lang: str) -> str:
    from braille_cnn.labels import code_to_label, decode_sequence

    if int(code) == 0:
        return " "
    if lang == "en":
        text = decode_sequence([code], lang="en")
        return text if text else code_to_label(code, lang="en")
    return code_to_label(code, lang="si")


def cellmap_from_recognize(cells: list[dict], lang: str = "si") -> CellMap:
    """Stage 4e dicts -> CellMap. line/col already assigned by recognize_page."""
    from data_pipeline.contracts import code_to_dot_string

    out: list[Cell] = []
    for cid, c in enumerate(cells):
        code = int(c["code"])
        char = c.get("char") or _decode_char(code, lang)
        out.append(
            Cell(
                id=cid,
                xyxy=tuple(float(v) for v in c["xyxy"]),
                char=str(char),
                pattern=code_to_dot_string(code),
                code=code,
                conf=float(c.get("conf", 1.0)),
                line=int(c.get("line", 0)),
                col=int(c.get("col", 0)),
            )
        )
    return CellMap(cells=out)


def _prescan_recognize(
    image_bgr: np.ndarray,
    *,
    backend: str,
    lang: str,
    device: str,
    cell_weights: Path | None,
    cnn_checkpoint: Path | None,
    cell_conf: float,
) -> CellMap:
    from braille_cnn.recognize import recognize_page

    cells = recognize_page(
        image_bgr,
        backend=backend,
        lang=lang,
        device=device,
        cnn_checkpoint=cnn_checkpoint,
        cell_weights=cell_weights,
        cell_conf=cell_conf,
    )
    return cellmap_from_recognize(cells, lang=lang)


def _prescan_dotneuralnet(
    image_bgr: np.ndarray,
    model,
    *,
    conf: float,
    lang: str,
    imgsz: int,
    device: str,
) -> CellMap:
    dnn = _ROOT / "experiments" / "DotNeuralNet"
    if str(dnn) not in sys.path:
        sys.path.insert(0, str(dnn))
    from src.layout import boxes_to_cells, group_into_lines

    res = model.predict(
        source=image_bgr, conf=conf, imgsz=imgsz, device=device, verbose=False
    )
    raw = boxes_to_cells(res[0].boxes, res[0].names)
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
    model=None,
    *,
    conf: float = 0.25,
    lang: str = "si",
    imgsz: int = 640,
    device: str = "cpu",
    backend: str = "auto",
    cell_weights: Path | None = None,
    cnn_checkpoint: Path | None = None,
) -> CellMap:
    """Scan one frame into a CellMap.

    backend:
      cells  — our YOLO cell detector + CNN
      dots   — classical/YOLO dots + CNN (infer_page)
      dnn    — third-party DotNeuralNet
      auto   — cells if weights exist, else dots, else dnn
    """
    own = Path(cell_weights) if cell_weights else _OWN_CELL_WEIGHTS
    chosen = backend
    if backend == "auto":
        if own.exists():
            chosen = "cells"
        else:
            chosen = "dots"

    if chosen in ("cells", "dots"):
        try:
            return _prescan_recognize(
                image_bgr,
                backend=chosen,
                lang=lang,
                device=device,
                cell_weights=own if chosen == "cells" else None,
                cnn_checkpoint=cnn_checkpoint,
                cell_conf=conf,
            )
        except FileNotFoundError as exc:
            print(f"recognize_page ({chosen}) unavailable: {exc}", flush=True)
            if backend != "auto":
                raise
            chosen = "dnn"

    if model is None:
        if not _DNN_WEIGHTS.exists():
            raise FileNotFoundError(
                "No cell-detector weights and no DotNeuralNet fallback.\n"
                "Train cell_detect on Colab, or keep experiments/DotNeuralNet/weights/"
                "yolov8_braille.pt"
            )
        from ultralytics import YOLO

        model = YOLO(str(_DNN_WEIGHTS))
    return _prescan_dotneuralnet(
        image_bgr, model, conf=conf, lang=lang, imgsz=imgsz, device=device
    )


def cells_from_yolo_result(boxes, names, lang: str = "en") -> CellMap:
    """DotNeuralNet Ultralytics boxes → CellMap. Used only by the dnn fallback."""
    dnn = _ROOT / "experiments" / "DotNeuralNet"
    if str(dnn) not in sys.path:
        sys.path.insert(0, str(dnn))
    from src.layout import boxes_to_cells, group_into_lines

    return _cells_from_grouped(group_into_lines(boxes_to_cells(boxes, names)), lang)


def _cells_from_grouped(lines, lang: str) -> CellMap:
    out: list[Cell] = []
    cid = 0
    for li, line in enumerate(lines):
        for col, c in enumerate(line):
            code = int(c["code"])
            char = _decode_char(code, lang)
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

    p = argparse.ArgumentParser(description="Image → CellMap via recognize_page")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--backend", choices=("auto", "cells", "dots", "dnn"), default="auto")
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--lang", choices=("en", "si"), default="si")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save-overlay", type=Path, default=None)
    args = p.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise SystemExit(f"Failed to read {args.image}")

    cell_map = prescan_bgr(
        bgr,
        conf=args.conf,
        lang=args.lang,
        imgsz=args.imgsz,
        device=args.device,
        backend=args.backend,
        cell_weights=args.weights,
        cnn_checkpoint=args.checkpoint,
    )
    print(f"cells={len(cell_map)}  lang={args.lang}  backend={args.backend}")
    for c in cell_map.cells[:40]:
        print(
            f"  id={c.id:3d} L{c.line}C{c.col} code={c.code:2d} "
            f"char={c.char!r} conf={c.conf:.2f}"
        )
    if len(cell_map) > 40:
        print(f"  ... +{len(cell_map) - 40} more")

    if args.save_overlay:
        args.save_overlay.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_overlay), draw_cellmap(bgr, cell_map))
        print(f"overlay -> {args.save_overlay}")


if __name__ == "__main__":
    main()
