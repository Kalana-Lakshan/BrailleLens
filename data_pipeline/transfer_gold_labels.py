"""Stage 1b — copy High-quality LabelMe boxes onto the Low-quality twins.

The two Gold folders are the same 12 physical pages under different lighting.
Annotate High quality only, then run this. A page whose homography is weak
is refused and must be labelled by hand.

    py -3.11 -m data_pipeline.transfer_gold_labels
    py -3.11 -m data_pipeline.transfer_gold_labels --min-inliers 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .contracts import repo_root

ROOT = repo_root()
HIGH = ROOT / "Gold Dataset" / "High quality dataset"
LOW = ROOT / "Gold Dataset" / "Low quality dataset"
QC_DIR = ROOT / "reports" / "gold_transfer"


def _load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _homography(src_gray: np.ndarray, dst_gray: np.ndarray, min_inliers: int):
    orb = cv2.ORB_create(nfeatures=3000)
    kp1, d1 = orb.detectAndCompute(src_gray, None)
    kp2, d2 = orb.detectAndCompute(dst_gray, None)
    if d1 is None or d2 is None or len(kp1) < 12 or len(kp2) < 12:
        return None, 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(d1, d2, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None, 0
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if H is None or inliers < min_inliers:
        return None, inliers
    return H, inliers


def _map_rect(points, H) -> list[list[float]]:
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in mapped]


def _overlay(high_bgr, low_bgr, src_shapes, dst_shapes, out_path: Path) -> None:
    def draw(img, shapes, color):
        out = img.copy()
        for shape in shapes:
            (x0, y0), (x1, y1) = shape["points"][:2]
            cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), color, 1)
        return out

    left = draw(high_bgr, src_shapes, (0, 200, 0))
    right = draw(low_bgr, dst_shapes, (0, 180, 255))
    h = min(left.shape[0], right.shape[0])
    left = cv2.resize(left, (int(left.shape[1] * h / left.shape[0]), h))
    right = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))
    canvas = np.hstack([left, right])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer Gold High labels onto Low twins")
    parser.add_argument("--high", type=Path, default=HIGH)
    parser.add_argument("--low", type=Path, default=LOW)
    parser.add_argument("--qc-dir", type=Path, default=QC_DIR)
    parser.add_argument("--min-inliers", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    jsons = sorted(args.high.glob("pg-*.json"))
    if not jsons:
        print(
            f"No LabelMe JSON in {args.high}.\n"
            "Annotate the 12 High quality pages first "
            "(Gold Dataset/ANNOTATION_GUIDELINES.md)."
        )
        return

    ok = fail = 0
    for src_json in jsons:
        stem = src_json.stem
        src_img = next((p for p in (args.high / f"{stem}.jpeg", args.high / f"{stem}.jpg") if p.exists()), None)
        dst_img = next((p for p in (args.low / f"{stem}.jpeg", args.low / f"{stem}.jpg") if p.exists()), None)
        dst_json = args.low / f"{stem}.json"
        if src_img is None or dst_img is None:
            print(f"  {stem}: missing image, skipped")
            fail += 1
            continue
        if dst_json.exists() and not args.overwrite:
            print(f"  {stem}: {dst_json.name} already exists (pass --overwrite)")
            continue

        doc = json.loads(src_json.read_text(encoding="utf-8"))
        H, inliers = _homography(_load_gray(src_img), _load_gray(dst_img), args.min_inliers)
        if H is None:
            print(f"  {stem}: REJECTED (inliers={inliers} < {args.min_inliers}) — label Low by hand")
            fail += 1
            continue

        new_shapes = []
        for shape in doc.get("shapes", []):
            if shape.get("shape_type") != "rectangle" or len(shape.get("points", [])) < 2:
                continue
            cloned = dict(shape)
            cloned["points"] = _map_rect(shape["points"][:2], H)
            new_shapes.append(cloned)

        low_bgr = cv2.imread(str(dst_img))
        h, w = low_bgr.shape[:2]
        out_doc = {
            "version": doc.get("version", "5.0.1"),
            "flags": {},
            "shapes": new_shapes,
            "imagePath": dst_img.name,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w,
        }
        dst_json.write_text(json.dumps(out_doc, indent=2), encoding="utf-8")
        high_bgr = cv2.imread(str(src_img))
        _overlay(high_bgr, low_bgr, doc.get("shapes", []), new_shapes, args.qc_dir / f"{stem}.png")
        print(f"  {stem}: transferred {len(new_shapes)} boxes  inliers={inliers}  qc={stem}.png")
        ok += 1

    print(f"\nDone: {ok} transferred, {fail} refused.")
    print(f"Inspect overlays in {args.qc_dir} before training on Gold.")


if __name__ == "__main__":
    main()
