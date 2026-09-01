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
from collections import Counter
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


def _drop_ruler_lines(cells: list[dict], min_cells: int = 15, top2_fraction: float = 0.55) -> list[dict]:
    """Drop cells belonging to a decorative divider/ruler row -- a long,
    dense horizontal line some Braille pages use between sections, which
    reads as a row of raised dots to the detector even though it is not a
    real cell. Gold Dataset/ANNOTATION_GUIDELINES.md explicitly excludes
    these from ground truth ("Skip blank space and decorative divider /
    ruler rows"), and pages 1-8 (the gold train pages) do contain some,
    correctly unboxed -- so the detector has *some* negative exposure, but
    not enough of the 12-image gold set for it to fully generalize (same
    root constraint behind most of this session's findings), and it still
    fires on ones it hasn't seen before (confirmed: an 18-box false-positive
    row on held-out pg-11, see "Ruler-line filter" in
    reports/eval/gold_cell_detector_finetune.md).

    Requires backend="cells" (needs the classifier's predicted codes, cells
    (backend="dots") already has its own point-level equivalent,
    dot_detect.filter_ruler_lines). This is the classify-then-check-code-
    uniformity idea data_pipeline/clean.py already uses for DBSI/Angelina
    manifest cleaning (_ruler_mask, RULER_SAME_CODE_FRACTION=0.80 on known
    ground-truth codes), ported to live inference with a looser threshold:
    a single-dominant-code check does not reproduce reliably here, because
    individual YOLO boxes crop the divider's repeating pattern at slightly
    different phases, producing 2-3 similar-looking predicted codes rather
    than collapsing to exactly one. The top-2-codes combined fraction is
    what actually separates a divider from real text at this noise level --
    validated directly against every line with >=10 cells on both held-out
    gold pages: the one confirmed real divider came out to 0.67, every real
    line 0.18-0.40 (see the same report section for the full table). 0.55
    leaves real margin on both sides of that gap without being anywhere
    near data_pipeline's 0.80 (tuned for clean ground-truth codes, not noisy
    inferred ones -- it would never fire here).
    """
    lines = group_into_lines(cells)
    keep = []
    for line in lines:
        if len(line) >= min_cells:
            counts = Counter(c["code"] for c in line)
            top2 = sum(n for _, n in counts.most_common(2))
            if top2 / len(line) >= top2_fraction:
                continue  # flagged as a ruler/divider row -- drop the whole line
        keep.extend(line)
    return keep


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
    spine_boost: bool = False,
    drop_ruler_lines: bool = True,
    apply_clahe: bool = True,
    clahe_clip_limit: float = 2.5,
    deskew: bool = True,
) -> list[dict]:
    """Detect cells on a page and classify each one.

    Returns a list of dicts:
        xyxy, code, char, conf, line, col

    spine_boost (backend="cells" only) re-detects the spine-proximal strip
    of the page at higher effective resolution and merges the result --
    recovers cells the open-book page curvature near the spine otherwise
    suppresses the confidence of (see CellDetector.detect_boxes's docstring
    and reports/eval/gold_cell_detector_finetune.md's failure analysis).
    Default off: it's a real, validated win on genuine open-book-spread
    photos, but adds a second inference pass and is untested on flat
    scans/single loose pages, where there's no spine effect to recover from.

    drop_ruler_lines (backend="cells" only) removes decorative divider/ruler
    rows the detector mistakes for real cells (see _drop_ruler_lines).
    Default on: validated against all 12 gold pages, not just the two
    held-out ones it was tuned against -- fires on exactly 2, both times
    with zero change to true positives (see "Ruler-line filter" in
    reports/eval/gold_cell_detector_finetune.md). Pass False to disable.

    apply_clahe / clahe_clip_limit / deskew (backend="cells" only) are
    forwarded to CellDetector -- see cell_detect/preprocess.py for what each
    does. Both default ON for live phone testing. CAUTION: apply_clahe was
    directly measured to make cell detection WORSE (not better) on one real
    phone photo, monotonically with clip strength -- being on by default
    here is so it can be checked against more real phone photos, not a
    reversal of that finding. Pass apply_clahe=False if it hurts on yours.
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

    detector = CellDetector(
        weights=cell_weights,
        conf=cell_conf,
        device=str(device),
        apply_clahe=apply_clahe,
        clahe_clip_limit=clahe_clip_limit,
        deskew=deskew,
    )
    detections = detector.detect_boxes(image, spine_boost=spine_boost)
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
    if drop_ruler_lines:
        cells = _drop_ruler_lines(cells)
    return _assign_line_col(cells, lang=lang)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4e — recognise one Braille page")
    parser.add_argument("--image", required=True)
    parser.add_argument("--backend", choices=("cells", "dots"), default="cells")
    parser.add_argument("--lang", choices=("en", "si"), default="si")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cell-weights", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--spine-boost", action="store_true",
                         help="Re-detect the spine-proximal strip at higher resolution and merge -- for genuine open-book-spread photos (see CellDetector.detect_boxes)")
    parser.add_argument("--drop-ruler-lines", action=argparse.BooleanOptionalAction, default=True,
                         help="Remove decorative divider/ruler rows the detector mistakes for cells (see _drop_ruler_lines). On by default; pass --no-drop-ruler-lines to disable")
    parser.add_argument("--clahe", action=argparse.BooleanOptionalAction, default=True,
                         help="CLAHE contrast correction before cell detection (see cell_detect/preprocess.py). On by default -- CAUTION: measured to HURT detection on one real phone photo; pass --no-clahe if it hurts on yours")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.5,
                         help="CLAHE clip limit -- higher pushes more local contrast, at more noise-amplification risk on blank regions")
    parser.add_argument("--deskew", action=argparse.BooleanOptionalAction, default=True,
                         help="Best-effort perspective deskew before cell detection (see cell_detect/preprocess.py::deskew_page). On by default; safe no-op when no confident page quad is found. Pass --no-deskew to disable")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cells = recognize_page(
        args.image,
        backend=args.backend,
        lang=args.lang,
        device=device,
        cnn_checkpoint=args.checkpoint,
        cell_weights=args.cell_weights,
        spine_boost=args.spine_boost,
        drop_ruler_lines=args.drop_ruler_lines,
        apply_clahe=args.clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        deskew=args.deskew,
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
