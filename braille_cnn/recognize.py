"""Stage 4e — one page-recognition API for training eval and the live app.

    recognize_page(image, backend="cells")
        -> list of {xyxy, code, char, conf, line, col}

backend="cells"  YOLO cell boxes + SimpleBrailleCNN  (deployment path)
backend="dots"   existing infer_page.run_auto_transcribe  (baseline)

line / col come from group_into_lines() in this file, so dropping
DotNeuralNet does not lose reading order.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from .infer_page import load_model, run_auto_transcribe
from .labels import code_to_label
from .normalize import normalize_crop

NUM_CLASSES = 64
IMG_SIZE = 64
DEFAULT_CNN = Path("braille_cnn/checkpoints/braille_cnn_mixed.pt")
ALT_CNN = Path("braille_cnn/checkpoints/braille_cnn_angelina_finetuned.pt")
DBSI_CNN = Path("braille_cnn/checkpoints/braille_cnn_dbsi_finetuned.pt")


def group_into_lines(cells: list[dict], gap_factor: float = 0.7) -> list[list[dict]]:
    """Cluster cell centres into reading-order lines (top→bottom, left→right).

    Ported out of DotNeuralNet's group_into_lines so recognize_page and the
    live app share one implementation. A new line starts when the y-gap
    exceeds gap_factor * median cell height.
    """
    if not cells:
        return []
    heights = [max(c["xyxy"][3] - c["xyxy"][1], 1.0) for c in cells]
    cell_h = float(np.median(heights))
    by_y = sorted(cells, key=lambda c: (c["xyxy"][1] + c["xyxy"][3]) / 2.0)
    line_gap = cell_h * gap_factor
    lines = [[by_y[0]]]
    for cell in by_y[1:]:
        prev = lines[-1][-1]
        prev_y = (prev["xyxy"][1] + prev["xyxy"][3]) / 2.0
        y = (cell["xyxy"][1] + cell["xyxy"][3]) / 2.0
        if y - prev_y > line_gap:
            lines.append([cell])
        else:
            lines[-1].append(cell)
    for line in lines:
        line.sort(key=lambda c: (c["xyxy"][0] + c["xyxy"][2]) / 2.0)
    lines.sort(key=lambda line: np.mean([(c["xyxy"][1] + c["xyxy"][3]) / 2.0 for c in line]))
    return lines


def _insert_word_gaps(
    line: list[dict], pitch: float, lang: str, gap_factor: float = 1.6, max_spaces: int = 5
) -> list[dict]:
    """Synthesize blank-cell ("space") entries at within-line x-gaps that are
    wide relative to the page's normal cell-to-cell pitch.

    The cell detector (YOLO, or the dot-grid path) only ever proposes a box
    where it sees raised dots -- an all-blank word-gap cell has nothing to
    detect, so it is structurally invisible to both backends. This recovers
    it after the fact from geometry: a word space is ~1 extra blank cell
    width beyond the normal letter-to-letter pitch, so a gap of ~2x pitch
    implies one missing blank cell, ~3x implies two, etc.
    """
    if len(line) < 2 or pitch <= 0:
        return line
    centers = [(c["xyxy"][0] + c["xyxy"][2]) / 2.0 for c in line]
    y0 = min(c["xyxy"][1] for c in line)
    y1 = max(c["xyxy"][3] for c in line)
    out = [line[0]]
    for i in range(len(line) - 1):
        delta = centers[i + 1] - centers[i]
        ratio = delta / pitch
        if ratio > gap_factor:
            n_spaces = max(1, min(round(ratio) - 1, max_spaces))
            for k in range(1, n_spaces + 1):
                cx = centers[i] + delta * k / (n_spaces + 1)
                out.append(
                    {
                        "xyxy": (cx - pitch * 0.3, y0, cx + pitch * 0.3, y1),
                        "code": 0,
                        "char": code_to_label(0, lang=lang),
                        "conf": 1.0,
                    }
                )
        out.append(line[i + 1])
    return out


def _assign_line_col(cells: list[dict], lang: str = "en") -> list[dict]:
    lines = group_into_lines(cells)
    all_deltas = []
    for line in lines:
        centers = [(c["xyxy"][0] + c["xyxy"][2]) / 2.0 for c in line]
        all_deltas.extend(centers[i + 1] - centers[i] for i in range(len(centers) - 1))
    pitch = float(np.median(all_deltas)) if all_deltas else 0.0

    out = []
    for line_i, line in enumerate(lines):
        augmented = _insert_word_gaps(line, pitch, lang)
        for col_i, cell in enumerate(augmented):
            cell["line"] = line_i
            cell["col"] = col_i
        out.extend(augmented)
    return out


def _resolve_cnn(path: str | Path | None) -> Path:
    if path:
        return Path(path)
    for cand in (DEFAULT_CNN, ALT_CNN, DBSI_CNN):
        if cand.exists():
            return cand
    return DEFAULT_CNN


def _classify_boxes(image: Image.Image, boxes: list[tuple], model, device, img_size: int):
    if not boxes:
        return [], []
    crops = []
    for x0, y0, x1, y1 in boxes:
        crop = image.crop((int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))))
        crop = crop.resize((img_size, img_size), Image.Resampling.BICUBIC)
        crops.append(crop)
    batch = torch.stack(
        [torch.from_numpy(normalize_crop(c)).unsqueeze(0) for c in crops]
    ).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
    return preds.cpu().tolist(), confs.cpu().tolist()


def _to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("L")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("L")
    arr = np.asarray(image)
    if arr.ndim == 3:
        import cv2

        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return Image.fromarray(arr)


def recognize_page(
    image,
    backend: str = "cells",
    lang: str = "si",
    device=None,
    cnn_checkpoint: str | Path | None = None,
    cell_weights: str | Path | None = None,
    cell_conf: float = 0.25,
    img_size: int = IMG_SIZE,
    model=None,
) -> list[dict]:
    """Detect cells on a page and classify each one.

    Returns a list of dicts:
        xyxy, code, char, conf, line, col
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pil = _to_pil(image)

    if backend == "dots":
        args = SimpleNamespace(
            checkpoint=str(_resolve_cnn(cnn_checkpoint)),
            img_size=img_size,
            lang=lang,
            conf_threshold=0.0,
            link_distance=None,
            use_cell_grid=True,
            filter_ruler_lines=True,
            crop_shape="angelina",
            cell_margin_scale=0.15,
            dot_backend="classical",
            dot_z_threshold=3.0,
            dot_footprint=9,
            dot_peak_y_offset=0.0,
            dot_classifier_checkpoint=None,
        )
        result = run_auto_transcribe(pil, args, model=model, device=device)
        cells = []
        for box, code, conf in zip(result["boxes"], result["preds"], result["confidences"]):
            code_i = int(code)
            cells.append(
                {
                    "xyxy": tuple(float(v) for v in box),
                    "code": code_i,
                    "char": code_to_label(code_i, lang=lang) if code_i else " ",
                    "conf": float(conf),
                    "line": 0,
                    "col": 0,
                }
            )
        return _assign_line_col(cells, lang=lang)

    if backend != "cells":
        raise ValueError(f"Unknown backend {backend!r}; use 'cells' or 'dots'")

    from cell_detect.detect_cells import CellDetector

    if model is None:
        ckpt = _resolve_cnn(cnn_checkpoint)
        if not Path(ckpt).exists():
            raise FileNotFoundError(
                f"CNN checkpoint not found: {ckpt}\n"
                "Train first: py -3.11 -m braille_cnn.train_classifier"
            )
        model = load_model(str(ckpt), device)

    detector = CellDetector(weights=cell_weights, conf=cell_conf, device=str(device))
    detections = detector.detect_boxes(image)
    boxes = [d["xyxy"] for d in detections]
    det_confs = [d["conf"] for d in detections]
    preds, cnn_confs = _classify_boxes(pil, boxes, model, device, img_size)

    cells = []
    for box, det_c, code, cnn_c in zip(boxes, det_confs, preds, cnn_confs):
        code_i = int(code)
        cells.append(
            {
                "xyxy": tuple(float(v) for v in box),
                "code": code_i,
                "char": code_to_label(code_i, lang=lang) if code_i else " ",
                "conf": float(min(det_c, cnn_c)),
                "line": 0,
                "col": 0,
            }
        )
    return _assign_line_col(cells, lang=lang)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4e — recognise one Braille page")
    parser.add_argument("--image", required=True)
    parser.add_argument("--backend", choices=("cells", "dots"), default="cells")
    parser.add_argument("--lang", choices=("en", "si"), default="si")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cell-weights", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cells = recognize_page(
        args.image,
        backend=args.backend,
        lang=args.lang,
        device=device,
        cnn_checkpoint=args.checkpoint,
        cell_weights=args.cell_weights,
    )
    print(f"{len(cells)} cells  backend={args.backend}")
    current_line = -1
    for cell in cells:
        if cell["line"] != current_line:
            current_line = cell["line"]
            print(f"\n--- line {current_line} ---")
        print(
            f"  col={cell['col']:3d}  code={cell['code']:2d}  "
            f"char={cell['char']}  conf={cell['conf']:.2f}"
        )


if __name__ == "__main__":
    main()
