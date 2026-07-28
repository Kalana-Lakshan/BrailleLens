"""Inference on a photo of a real (unannotated) Braille page.

Two cropping strategies, chosen with --auto:

Fixed-grid (default): divides --region into --rows x --cols evenly spaced
cells. Only correct if the page is flat and the grid is regular (a clean
scan or a straight-on, undistorted photo) -- any skew, curvature, page
perspective, or variable-length lines (real prose with word spacing) breaks
the "equal spacing everywhere" assumption. Always check --debug-out before
trusting the predictions.

Automatic (--auto): detects actual dot highlights (dot_detect.py) and groups
them into per-cell clusters from their measured positions, instead of a
guessed uniform grid. Tolerates variable-length lines and mild skew, since
it only crops where dots actually are. Still assumes one global within-cell
dot-linking distance (--link-distance) -- if perspective makes the pitch
shrink a lot across the page, some clusters merge multiple real cells; those
are flagged as "?" rather than guessed at (see dot_detect.MAX_DOTS_PER_CELL).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from .cnn import SimpleBrailleCNN
from .dot_detect import cluster_into_cells, detect_dot_centers
from .labels import code_to_label

NUM_CLASSES = 64


def _parse_region(region_str, img_w, img_h):
    if region_str is None:
        return (0, 0, img_w, img_h)
    x0, y0, x1, y1 = (float(v) for v in region_str.split(","))
    return (x0, y0, x1, y1)


# ---------------------------------------------------------------- fixed grid

def _cell_crops(image, rows, cols, region, img_size, margin_scale):
    x0, y0, x1, y1 = region
    cell_w = (x1 - x0) / cols
    cell_h = (y1 - y0) / rows

    boxes = []
    crops = []
    for r in range(rows):
        for c in range(cols):
            cx0 = x0 + c * cell_w
            cx1 = cx0 + cell_w
            cy0 = y0 + r * cell_h
            cy1 = cy0 + cell_h
            margin_x = cell_w * margin_scale
            margin_y = cell_h * margin_scale
            box = (
                max(cx0 - margin_x, 0),
                max(cy0 - margin_y, 0),
                min(cx1 + margin_x, image.width),
                min(cy1 + margin_y, image.height),
            )
            boxes.append(box)
            crop = image.crop(tuple(int(round(v)) for v in box))
            crop = crop.resize((img_size, img_size), Image.Resampling.BICUBIC)
            crops.append(crop)
    return crops, boxes


def _save_grid_debug_overlay(image, region, boxes, out_path):
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    x0, y0, x1, y1 = region
    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
    for box in boxes:
        draw.rectangle(box, outline=(0, 255, 0), width=1)
    overlay.save(out_path)
    print(f"saved debug overlay: {out_path} (red = --region, green = per-cell crop boxes)")


def run_fixed_grid(image, args):
    region = _parse_region(args.region, image.width, image.height)
    crops, boxes = _cell_crops(image, args.rows, args.cols, region, args.img_size, args.margin_scale)

    if args.debug_out:
        _save_grid_debug_overlay(image, region, boxes, Path(args.debug_out))

    preds, confidences = _classify(crops, args)

    print(f"\npredicted grid ({args.rows} rows x {args.cols} cols), checkpoint={args.checkpoint}:\n")
    idx = 0
    for r in range(args.rows):
        row_labels = []
        for c in range(args.cols):
            label = code_to_label(preds[idx].item(), lang=args.lang)
            row_labels.append(f"{label}({confidences[idx].item():.2f})")
            idx += 1
        print("  " + " ".join(row_labels))


# ------------------------------------------------------------------- auto

def _estimate_cell_size(clusters):
    sizes = [
        (c["bbox"][2] - c["bbox"][0], c["bbox"][3] - c["bbox"][1])
        for c in clusters if not c["merged"] and len(c["points"]) >= 2
    ]
    if not sizes:
        return 20.0, 20.0  # fallback if too few multi-dot clusters to measure
    widths, heights = zip(*sizes)
    return float(np.median(widths)), float(np.median(heights))


def _cluster_crop_box(center, cell_w, cell_h, margin_scale, img_w, img_h):
    cx, cy = center
    half_w = cell_w * (0.5 + margin_scale)
    half_h = cell_h * (0.5 + margin_scale)
    return (
        max(cx - half_w, 0),
        max(cy - half_h, 0),
        min(cx + half_w, img_w),
        min(cy + half_h, img_h),
    )


def _line_gap_threshold(sorted_ys, cell_h_med):
    """Otsu-split the y-center gaps into "same line" vs "line break" groups.

    Consecutive clusters within one physical line still have some y jitter
    (skew, dot row within the cell), but the jump to the next line is
    consistently much bigger -- Otsu's threshold (usually used for pixel
    intensity, here reused on 1D gap values) finds that split point
    automatically instead of guessing a fixed multiple of cell size.
    """
    diffs = np.diff(sorted_ys)
    if len(diffs) < 2 or diffs.max() <= 0:
        return cell_h_med * 1.5
    scaled = np.clip(diffs / diffs.max() * 255, 0, 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = (otsu_val / 255.0) * diffs.max()
    return max(threshold, cell_h_med * 1.3)


def _group_into_lines(clusters):
    """Gap-based 1D grouping of cluster centers by y, then by x within a line."""
    if not clusters:
        return []
    cell_h_med = float(np.median([c["bbox"][3] - c["bbox"][1] for c in clusters if not c["merged"]] or [20.0]))

    by_y = sorted(clusters, key=lambda c: c["center"][1])
    line_gap = _line_gap_threshold([c["center"][1] for c in by_y], cell_h_med)
    lines = [[by_y[0]]]
    for c in by_y[1:]:
        if c["center"][1] - lines[-1][-1]["center"][1] > line_gap:
            lines.append([c])
        else:
            lines[-1].append(c)

    for line in lines:
        line.sort(key=lambda c: c["center"][0])
    lines.sort(key=lambda line: np.mean([c["center"][1] for c in line]))
    return lines


def _assemble_line_text(line, labels):
    gaps = [
        line[i + 1]["bbox"][0] - line[i]["bbox"][2]
        for i in range(len(line) - 1)
    ]
    word_gap = np.median(gaps) * 1.8 if gaps else 0.0

    out = [labels[0]]
    for i in range(1, len(line)):
        if gaps[i - 1] > max(word_gap, 1.0):
            out.append(" ")
        out.append(labels[i])
    return "".join(out)


def run_auto(image, args):
    gray = np.asarray(image, dtype=np.float32)
    points = detect_dot_centers(gray, percentile=args.dot_percentile, footprint=args.dot_footprint)
    clusters = cluster_into_cells(points, link_distance=args.link_distance)
    print(f"detected {len(points)} candidate dots -> {len(clusters)} cell clusters "
          f"({sum(c['merged'] for c in clusters)} flagged as merged/uncertain)")

    cell_w, cell_h = _estimate_cell_size(clusters)
    print(f"estimated single-cell size: {cell_w:.1f} x {cell_h:.1f} px")

    valid = [c for c in clusters if not c["merged"]]
    boxes = [
        _cluster_crop_box(c["center"], cell_w, cell_h, args.cell_margin_scale, image.width, image.height)
        for c in valid
    ]
    crops = [
        image.crop(tuple(int(round(v)) for v in box)).resize((args.img_size, args.img_size), Image.Resampling.BICUBIC)
        for box in boxes
    ]
    preds, confidences = _classify(crops, args)
    labels = [code_to_label(p.item(), lang=args.lang) for p in preds]

    lines = _group_into_lines(valid)
    label_by_id = {id(c): (lbl, conf.item()) for c, lbl, conf in zip(valid, labels, confidences)}

    print(f"\ntranscription attempt ({len(lines)} lines detected), checkpoint={args.checkpoint}:\n")
    for line in lines:
        line_labels = [label_by_id[id(c)][0] for c in line]
        print("  " + _assemble_line_text(line, line_labels))

    if args.debug_out:
        overlay = image.convert("RGB").copy()
        draw = ImageDraw.Draw(overlay)
        for c in clusters:
            color = (255, 0, 0) if c["merged"] else (0, 255, 0)
            x0, y0, x1, y1 = c["bbox"]
            draw.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=color, width=1)
        for c, box in zip(valid, boxes):
            lbl, conf = label_by_id[id(c)]
            draw.rectangle(box, outline=(0, 128, 255), width=1)
            draw.text((box[0], max(box[1] - 10, 0)), f"{lbl}", fill=(0, 128, 255))
        overlay.save(args.debug_out)
        print(f"\nsaved debug overlay: {args.debug_out} "
              f"(green = single-cell cluster, red = flagged merged cluster, blue = classified crop box)")


# --------------------------------------------------------------- shared

def _classify(crops, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleBrailleCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    if not crops:
        return torch.empty(0, dtype=torch.long), torch.empty(0)

    batch = torch.stack([
        torch.from_numpy(np.asarray(crop, dtype=np.float32) / 255.0).unsqueeze(0)
        for crop in crops
    ]).to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        confidences, preds = probs.max(dim=1)
    return preds, confidences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="path to the page photo/scan")
    parser.add_argument("--auto", action="store_true",
                         help="detect dots and cluster into cells automatically instead of a fixed --rows/--cols grid")
    parser.add_argument("--rows", type=int, default=None, help="[fixed-grid] number of braille-cell rows in --region")
    parser.add_argument("--cols", type=int, default=None, help="[fixed-grid] number of braille-cell columns in --region")
    parser.add_argument("--region", type=str, default=None,
                         help="[fixed-grid] x0,y0,x1,y1 pixel box containing the grid; default is the whole image")
    parser.add_argument("--link-distance", type=float, default=15.0,
                         help="[auto] max pixel distance between dots to link them into the same cell")
    parser.add_argument("--dot-percentile", type=float, default=99.3,
                         help="[auto] brightness-difference percentile cutoff for a peak to count as a dot")
    parser.add_argument("--dot-footprint", type=int, default=9,
                         help="[auto] non-max-suppression window in pixels (~one dot's diameter)")
    parser.add_argument("--checkpoint", type=str,
                         default="braille_cnn/checkpoints/braille_cnn_dbsi_finetuned.pt")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--margin-scale", type=float, default=0.2,
                         help="[fixed-grid] extra padding around each cell as a fraction of cell PITCH")
    parser.add_argument("--cell-margin-scale", type=float, default=0.8,
                         help="[auto] extra padding around each cell as a fraction of the measured dot-span "
                              "(0.8 matches DBSIDataset's convention, i.e. what braille_cnn_dbsi_finetuned.pt "
                              "was actually trained on -- see dbsi_dataset.py's _cell_box)")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "si"])
    parser.add_argument("--debug-out", type=str, default=None,
                         help="optional path to save an overlay image showing the detected/assumed grid")
    args = parser.parse_args()

    if not args.auto and (args.rows is None or args.cols is None):
        parser.error("--rows and --cols are required unless --auto is set")

    image = Image.open(args.image).convert("L")

    if args.auto:
        run_auto(image, args)
    else:
        run_fixed_grid(image, args)


if __name__ == "__main__":
    main()
