"""Convert LabelMe fingertip rectangles to YOLO detection labels.

Reads JSON from Gold Dataset/Braille_fingertip/ (or --source) and writes
normalized YOLO lines (class 0 = fingertip) to a staging labels folder.

Usage (from repo root)::

    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/yolo_domain_specific/labelme_to_yolo.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_DEFAULT_SOURCE = _REPO / "Gold Dataset" / "Braille_fingertip"
_DEFAULT_OUT = _HERE / "staging" / "labels"

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_CLASS_ID = 0
_VALID_LABELS = {"fingertip", "Fingertip", "FINGERTIP"}


def _load_image_size(json_path: Path) -> tuple[int, int] | None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    h = data.get("imageHeight")
    w = data.get("imageWidth")
    if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
        return w, h
    return None


def labelme_rect_to_yolo(
    points: list,
    img_w: int,
    img_h: int,
    class_id: int = _CLASS_ID,
) -> str | None:
    """LabelMe rectangle (two corners) -> YOLO normalized line."""
    if len(points) < 2:
        return None
    xs = [float(p[0]) for p in points[:2]]
    ys = [float(p[1]) for p in points[:2]]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    nw = bw / img_w
    nh = bh / img_h
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 1e-6), 1.0)
    nh = min(max(nh, 1e-6), 1.0)
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def convert_json(json_path: Path, img_w: int, img_h: int) -> list[str]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        label = shape.get("label", "")
        if label not in _VALID_LABELS:
            continue
        line = labelme_rect_to_yolo(shape.get("points", []), img_w, img_h)
        if line:
            lines.append(line)
    return lines


def find_image_for_json(json_path: Path, source: Path) -> Path | None:
    stem = json_path.stem
    for ext in _IMG_EXTS:
        cand = source / f"{stem}{ext}"
        if cand.is_file():
            return cand
        cand = source / f"{stem}{ext.upper()}"
        if cand.is_file():
            return cand
    return None


def resolve_image_size(json_path: Path, source: Path) -> tuple[int, int]:
    size = _load_image_size(json_path)
    if size:
        return size
    img = find_image_for_json(json_path, source)
    if img is None:
        raise FileNotFoundError(f"No image for {json_path.name}")
    import cv2

    bgr = cv2.imread(str(img))
    if bgr is None:
        raise ValueError(f"Unreadable image: {img}")
    h, w = bgr.shape[:2]
    return w, h


def convert_folder(source: Path, out_labels: Path) -> dict[str, int]:
    out_labels.mkdir(parents=True, exist_ok=True)
    stats = {"json": 0, "labeled": 0, "empty": 0, "skipped_no_json": 0}

    images = sorted(
        f for f in source.iterdir() if f.suffix.lower() in _IMG_EXTS and f.is_file()
    )
    for img_path in images:
        json_path = source / f"{img_path.stem}.json"
        out_txt = out_labels / f"{img_path.stem}.txt"
        if not json_path.exists():
            stats["skipped_no_json"] += 1
            continue
        stats["json"] += 1
        w, h = resolve_image_size(json_path, source)
        lines = convert_json(json_path, w, h)
        out_txt.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        if lines:
            stats["labeled"] += 1
        else:
            stats["empty"] += 1
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="LabelMe fingertip JSON -> YOLO txt")
    p.add_argument("--source", type=Path, default=_DEFAULT_SOURCE)
    p.add_argument("--out-labels", type=Path, default=_DEFAULT_OUT)
    args = p.parse_args()

    if not args.source.is_dir():
        raise SystemExit(f"Source folder not found: {args.source}")

    stats = convert_folder(args.source, args.out_labels)
    print(f"Source: {args.source}")
    print(f"Labels: {args.out_labels}")
    print(
        f"JSON converted: {stats['json']}  "
        f"with boxes: {stats['labeled']}  "
        f"empty: {stats['empty']}  "
        f"images missing JSON: {stats['skipped_no_json']}"
    )
    if stats["json"] == 0:
        print(
            "\nNo LabelMe JSON found. Annotate first — see ANNOTATION_GUIDE.md",
            flush=True,
        )


if __name__ == "__main__":
    main()
