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
from .dot_classifier import DotPatchCNN
from .dot_detect import (
    cluster_by_grid,
    cluster_into_cells,
    detect_dot_centers,
    estimate_link_distance,
    filter_ruler_lines,
    fit_cell_grid,
    grid_cell_center,
)
from .normalize import normalize_crop
from .labels import (
    CODE_TO_SINHALA,
    INDICATOR_CODES,
    TWO_CELL_VOWEL_SIGNS,
    _is_combining_vowel_sign,
    code_to_label,
)

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


def run_fixed_grid(image, args, model=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = load_model(args.checkpoint, device)

    region = _parse_region(args.region, image.width, image.height)
    crops, boxes = _cell_crops(image, args.rows, args.cols, region, args.img_size, args.margin_scale)

    if args.debug_out:
        _save_grid_debug_overlay(image, region, boxes, Path(args.debug_out))

    preds, confidences = _classify(crops, model, device)

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


def _grid_crop_box(center, dx, dy, margin_scale, img_w, img_h):
    """Crop box matching DBSIDataset._cell_box's exact (asymmetric!) formula:
    width = dx*(1 + 2*margin_scale), height = 2*dy*(1 + margin_scale) -- the
    vertical margin there is scaled off a single row-gap (y1-y0), not the
    full top-to-bottom dot span (y2-y0), so width and height do NOT use the
    same multiplier. braille_cnn_dbsi_finetuned.pt was trained on crops built
    with this exact formula (via DBSIDataset); _cluster_crop_box's symmetric
    formula only approximates it via a single averaged cell_w/cell_h, which
    measurably hurt accuracy (47.8% vs 96.2% on the same held-out cells with
    an exact-formula crop -- see chat history / README). Only usable where a
    grid fit succeeded (need dx, dy from dot_detect.fit_cell_grid).
    """
    cx, cy = center
    half_w = dx * (0.5 + margin_scale)
    half_h = dy * (1.0 + margin_scale)
    return (
        max(cx - half_w, 0),
        max(cy - half_h, 0),
        min(cx + half_w, img_w),
        min(cy + half_h, img_h),
    )


def _angelina_grid_crop_box(center, dx, dy, margin_scale, img_w, img_h):
    """Crop box matching AngelinaDataset's box convention: its annotated
    cell boxes are consistently ~2*dx wide by ~3*dy tall (confirmed by
    measuring true box width/height against a fitted grid's dx/dy on a real
    Angelina photo: width/dx=1.98, height/dy=3.07), unlike DBSI's tighter,
    asymmetric _grid_crop_box formula. Using the DBSI formula's crop shape
    on an Angelina photo starved the classifier of most of its cell margin
    (~60% the expected crop size) and measurably wrecked accuracy end-to-end
    (10% vs 43% on the same detected cells with this formula, same
    checkpoint) even though braille_cnn_angelina_finetuned.pt saw both crop
    conventions during its mixed fine-tuning -- crop *shape*, not just
    classifier training domain, matters. margin_scale is applied the same
    way AngelinaDataset itself applies it (symmetric fraction of box
    width/height), not DBSI's per-axis-different convention.
    """
    cx, cy = center
    half_w = dx * (1.0 + 2 * margin_scale)
    half_h = 1.5 * dy * (1.0 + 2 * margin_scale)
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


def _group_into_lines_by_grid(valid, centers, grid):
    """Same output shape as _group_into_lines, but groups by the fitted
    grid's own line index instead of a separate y-gap heuristic -- exact
    (derived from the same page-wide model already used for crop centering)
    instead of approximate, and avoids the two disagreeing (confirmed: the
    y-gap heuristic reported 40 "lines" on a page the grid fit correctly
    resolved to 28, once noisy/near-miss cluster y-jitter got amplified by
    grid-based clustering producing more, cleaner, more tightly-spaced
    clusters per true line than before).
    """
    Py, phase_y = grid["Py"], grid["phase_y"]
    by_line = {}
    for c, center in zip(valid, centers):
        line_idx = int(round((center[1] - phase_y) / Py))
        by_line.setdefault(line_idx, []).append(c)
    lines = [by_line[k] for k in sorted(by_line)]
    for line in lines:
        line.sort(key=lambda c: c["center"][0])
    return lines


def _line_word_gaps(line):
    gaps = [
        line[i + 1]["bbox"][0] - line[i]["bbox"][2]
        for i in range(len(line) - 1)
    ]
    word_gap = np.median(gaps) * 1.8 if gaps else 0.0
    return gaps, word_gap


def _assemble_line_text(line, labels):
    gaps, word_gap = _line_word_gaps(line)

    out = [labels[0]]
    for i in range(1, len(line)):
        if gaps[i - 1] > max(word_gap, 1.0):
            out.append(" ")
        out.append(labels[i])
    return "".join(out)


def _decode_line_with_confidence(line, code_by_id, lang, conf_threshold):
    """Assemble one braille line into readable text, marking low-confidence cells as '_'."""
    gaps, word_gap = _line_word_gaps(line)

    if lang != "si":
        labels = []
        for c in line:
            code, conf = code_by_id[id(c)]
            if conf < conf_threshold:
                labels.append("_")
            else:
                labels.append(code_to_label(code, lang=lang))
        return _assemble_line_text(line, labels)

    parts: list[str] = []
    i = 0
    while i < len(line):
        if i > 0 and gaps[i - 1] > max(word_gap, 1.0):
            parts.append(" ")

        c = line[i]
        code, conf = code_by_id[id(c)]

        if conf < conf_threshold:
            parts.append("_")
            i += 1
            continue

        if code in INDICATOR_CODES and i + 1 < len(line):
            c2 = line[i + 1]
            code2, conf2 = code_by_id[id(c2)]
            if conf2 >= conf_threshold:
                pair = (code, code2)
                if pair in TWO_CELL_VOWEL_SIGNS:
                    sign = TWO_CELL_VOWEL_SIGNS[pair]
                    if parts and _is_combining_vowel_sign(sign):
                        parts[-1] = parts[-1] + sign
                    else:
                        parts.append(sign)
                    i += 2
                    continue

        if code == 0:
            parts.append(" ")
        else:
            parts.append(CODE_TO_SINHALA.get(code, f"#{code}"))
        i += 1

    return "".join(parts)


def _assemble_transcription(lines, valid, preds, confidences, lang, conf_threshold):
    code_by_id = {
        id(c): (preds[i].item(), confidences[i].item()) for i, c in enumerate(valid)
    }
    text_lines = [
        _decode_line_with_confidence(line, code_by_id, lang, conf_threshold)
        for line in lines
    ]
    sentence = "\n".join(text_lines)
    return text_lines, sentence


def load_dot_classifier(checkpoint: str, device: torch.device):
    model = DotPatchCNN().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


def verify_dots(points, image, dot_model, device, patch_size=32, threshold=0.5):
    """Filters candidate dot points through a trained DotPatchCNN, dropping
    ones it doesn't recognize as a real dot. See dot_patch_dataset.py /
    train_dot_classifier.py -- this is the learned replacement for a
    hand-tuned brightness threshold (mirrors the DSBI paper's own
    Haar+Adaboost verification stage). Confirmed on held-out DBSI pages to
    take raw-detector precision from ~44-53% to ~99% at a modest recall cost
    (~86% -> ~90% depending on settings); NOT yet confirmed to transfer to
    real handheld photos (trained on DBSI only) -- see chat history/README.
    """
    if len(points) == 0:
        return points
    half = patch_size // 2
    patches, keep_idx = [], []
    for i, (x, y) in enumerate(points):
        xi, yi = int(round(x)), int(round(y))
        box = (xi - half, yi - half, xi - half + patch_size, yi - half + patch_size)
        if box[0] < 0 or box[1] < 0 or box[2] > image.width or box[3] > image.height:
            continue
        patches.append(np.asarray(image.crop(box), dtype=np.uint8))
        keep_idx.append(i)
    if not patches:
        return np.empty((0, 2))
    batch = torch.from_numpy(np.stack(patches)).float().unsqueeze(1).to(device) / 255.0
    with torch.no_grad():
        probs = torch.softmax(dot_model(batch), dim=1)[:, 1].cpu().numpy()
    keep_idx = np.array(keep_idx)[probs > threshold]
    return points[keep_idx]


def run_auto_transcribe(image, args, model=None, device=None, dot_model=None):
    """Detect, classify, and return structured transcription data (no printing)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = load_model(args.checkpoint, device)

    dot_classifier_checkpoint = getattr(args, "dot_classifier_checkpoint", None)
    if dot_model is None and dot_classifier_checkpoint:
        dot_model = load_dot_classifier(dot_classifier_checkpoint, device)

    conf_threshold = getattr(args, "conf_threshold", 0.0)

    gray = np.asarray(image, dtype=np.float32)
    points = detect_dot_centers(gray, z_threshold=args.dot_z_threshold, footprint=args.dot_footprint,
                                 peak_y_offset=getattr(args, "dot_peak_y_offset", 0.0))
    if dot_model is not None:
        points = verify_dots(points, image, dot_model, device,
                              threshold=getattr(args, "dot_classifier_threshold", 0.5))

    if getattr(args, "filter_ruler_lines", True):
        points = filter_ruler_lines(points)

    # Fit the page's regular cell grid from the (ideally classifier-verified,
    # i.e. clean) points first, since it changes BOTH how points get grouped
    # into cells and where each cell's crop is centered:
    #
    # - Clustering: cluster_into_cells' distance threshold can silently fuse
    #   two different real cells into one "valid" (not merged-flagged)
    #   cluster whenever their combined point count stays under the 6-dot
    #   limit (e.g. a 1-dot cell next to a 2-dot cell). Confirmed on a
    #   held-out DBSI page: this cost ~150 of 618 true cells their own
    #   cluster. cluster_by_grid assigns each point to whichever grid slot
    #   it's actually closest to, so two real cells can't merge as long as
    #   the grid fit itself is accurate -- pushed matched-cell coverage from
    #   76% to 99% on that same page.
    # - Centering: a cluster's own point centroid is a bad crop center for
    #   asymmetric dot patterns (e.g. only the right-column dots active
    #   pulls the centroid off the true cell center); the grid's fitted
    #   center doesn't have that problem.
    #
    # Only trust either on a clean point set -- the same fit attempted on
    # raw noisy (unverified) detections failed outright (see chat history).
    grid = fit_cell_grid(points) if getattr(args, "use_cell_grid", True) else None

    if grid is not None:
        clusters = cluster_by_grid(points, grid)
        link_distance = None
    else:
        link_distance = args.link_distance
        if link_distance is None:
            link_distance = estimate_link_distance(points)
        clusters = cluster_into_cells(points, link_distance=link_distance)

    cell_w, cell_h = _estimate_cell_size(clusters)
    valid = [c for c in clusters if not c["merged"]]

    centers = []
    used_grid_box = []
    for c in valid:
        center = c["center"]
        on_grid = False
        if grid is not None:
            fitted = grid_cell_center(center, grid)
            if fitted is not None:
                center = fitted
                on_grid = True
        centers.append(center)
        used_grid_box.append(on_grid)

    crop_shape = getattr(args, "crop_shape", "dbsi")
    crop_shape_fn = _angelina_grid_crop_box if crop_shape == "angelina" else _grid_crop_box
    cell_margin_scale = getattr(args, "cell_margin_scale", None)
    if cell_margin_scale is None:
        cell_margin_scale = 0.15 if crop_shape == "angelina" else 0.8
    boxes = [
        crop_shape_fn(center, grid["dx"], grid["dy"], cell_margin_scale, image.width, image.height)
        if on_grid else
        _cluster_crop_box(center, cell_w, cell_h, cell_margin_scale, image.width, image.height)
        for center, on_grid in zip(centers, used_grid_box)
    ]
    crops = [
        image.crop(tuple(int(round(v)) for v in box)).resize((args.img_size, args.img_size), Image.Resampling.BICUBIC)
        for box in boxes
    ]
    preds, confidences = _classify(crops, model, device)
    if grid is not None:
        lines = _group_into_lines_by_grid(valid, centers, grid)
    else:
        lines = _group_into_lines(valid)
    text_lines, sentence = _assemble_transcription(
        lines, valid, preds, confidences, args.lang, conf_threshold
    )

    return {
        "num_dots": len(points),
        "link_distance": link_distance,
        "num_clusters": len(clusters),
        "num_merged": sum(c["merged"] for c in clusters),
        "cell_size": (cell_w, cell_h),
        "lines": text_lines,
        "sentence": sentence,
        "num_valid_cells": len(valid),
        "clusters": clusters,
        "valid": valid,
        "boxes": boxes,
        "preds": preds,
        "confidences": confidences,
        "grouped_lines": lines,
    }


def run_auto(image, args, model=None, device=None, quiet=False):
    result = run_auto_transcribe(image, args, model=model, device=device)

    if not quiet:
        link_desc = f"{result['link_distance']:.1f}px" if result["link_distance"] is not None else "n/a (grid-based clustering)"
        print(f"detected {result['num_dots']} candidate dots -> {result['num_clusters']} cell clusters "
              f"({result['num_merged']} flagged as merged/uncertain), link_distance={link_desc}")
        cell_w, cell_h = result["cell_size"]
        print(f"estimated single-cell size: {cell_w:.1f} x {cell_h:.1f} px")
        print(f"\ntranscription ({len(result['lines'])} lines detected), checkpoint={args.checkpoint}:\n")
        for line in result["lines"]:
            print(f"  {line}")

    if args.debug_out:
        overlay = image.convert("RGB").copy()
        draw = ImageDraw.Draw(overlay)
        for c in result["clusters"]:
            color = (255, 0, 0) if c["merged"] else (0, 255, 0)
            x0, y0, x1, y1 = c["bbox"]
            draw.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=color, width=1)
        cell_label_by_id = {
            id(c): (code_to_label(result["preds"][i].item(), lang=args.lang), result["confidences"][i].item())
            for i, c in enumerate(result["valid"])
        }
        for c, box in zip(result["valid"], result["boxes"]):
            lbl, _conf = cell_label_by_id[id(c)]
            draw.rectangle(box, outline=(0, 128, 255), width=1)
            draw.text((box[0], max(box[1] - 10, 0)), f"{lbl}", fill=(0, 128, 255))
        overlay.save(args.debug_out)
        if not quiet:
            print(f"\nsaved debug overlay: {args.debug_out} "
                  f"(green = single-cell cluster, red = flagged merged cluster, blue = classified crop box)")

    return result


# --------------------------------------------------------------- shared

def load_model(checkpoint: str, device: torch.device):
    """Load the CNN once at startup and return the model ready for inference."""
    model = SimpleBrailleCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


def _classify(crops, model, device):
    """Run inference on a list of PIL crops using a pre-loaded model."""
    if not crops:
        return torch.empty(0, dtype=torch.long), torch.empty(0)

    batch = torch.stack([
        torch.from_numpy(normalize_crop(crop)).unsqueeze(0)
        for crop in crops
    ]).to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        confidences, preds = probs.max(dim=1)
    return preds, confidences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="path to the page photo/scan")
    parser.add_argument("--camera", action="store_true",
                         help="open a live camera feed instead of a static image (uses camera_capture/)")
    parser.add_argument("--auto", action="store_true",
                         help="detect dots and cluster into cells automatically instead of a fixed --rows/--cols grid")
    parser.add_argument("--rows", type=int, default=None, help="[fixed-grid] number of braille-cell rows in --region")
    parser.add_argument("--cols", type=int, default=None, help="[fixed-grid] number of braille-cell columns in --region")
    parser.add_argument("--region", type=str, default=None,
                         help="[fixed-grid] x0,y0,x1,y1 pixel box containing the grid; default is the whole image")
    parser.add_argument("--link-distance", type=float, default=None,
                         help="[auto] max pixel distance between dots to link them into the same cell; "
                              "default (unset) auto-estimates this per-image from the photo's own dot "
                              "spacing instead of assuming one fixed pixel value (see "
                              "dot_detect.estimate_link_distance)")
    parser.add_argument("--dot-z-threshold", type=float, default=3.0,
                         help="[auto] local z-score cutoff for a peak to count as a dot (see dot_detect.py); "
                              "adapts per-region instead of one global brightness percentile, so it transfers "
                              "across photos/scans with different lighting without retuning")
    parser.add_argument("--dot-footprint", type=int, default=9,
                         help="[auto] non-max-suppression window in pixels (~one dot's diameter)")
    parser.add_argument("--dot-peak-y-offset", type=float, default=0.0,
                         help="[auto] added to every detected dot's y-coordinate. Confirmed empirically on 5 "
                              "different DBSI books that this detector's peak lands ~2-6px ABOVE the true dot "
                              "center (a real asymmetric-lighting effect in the underlying signal, not a bug a "
                              "smarter peak-finder fixes -- see dot_detect.detect_dot_centers docstring). 5.5 is "
                              "a well-supported default for DBSI-style flatbed scans (cut end-to-end "
                              "misclassification roughly in half across 3 held-out pages); defaults to 0.0 "
                              "(off) here since it's unverified on other capture setups (e.g. phone photos) -- "
                              "re-measure before assuming it transfers.")
    parser.add_argument("--dot-classifier-checkpoint", type=str, default=None,
                         help="[auto] optional path to a trained DotPatchCNN (train_dot_classifier.py) to verify "
                              "candidate dots before clustering, replacing brightness-threshold-only filtering. "
                              "Validated on held-out DBSI pages (~44-53%% -> ~99%% precision); NOT yet confirmed "
                              "to transfer to real handheld photos (DBSI-only training data) -- disabled by "
                              "default for that reason.")
    parser.add_argument("--dot-classifier-threshold", type=float, default=0.5,
                         help="[auto] softmax probability cutoff for the dot classifier, if enabled")
    parser.add_argument("--crop-shape", type=str, default="dbsi", choices=["dbsi", "angelina"],
                         help="[auto] which crop-box geometry to use when a grid fit succeeds: 'dbsi' (dx wide x "
                              "2*dy tall, asymmetric margin -- matches DBSIDataset/_grid_crop_box, what "
                              "braille_cnn_dbsi_finetuned.pt was built on) or 'angelina' (2*dx wide x 3*dy tall, "
                              "symmetric margin -- matches AngelinaDataset's own box convention). Getting this "
                              "wrong for the photo's actual domain silently starves the classifier of the crop "
                              "shape it was trained on -- confirmed to cost ~33 accuracy points end-to-end on a "
                              "real Angelina photo even with an otherwise-correct grid fit and checkpoint. Use "
                              "'angelina' for real handheld phone photos, 'dbsi' for flatbed scans.")
    parser.add_argument("--no-cell-grid", dest="use_cell_grid", action="store_false",
                         help="[auto] disable grid-fitted crop centering (dot_detect.fit_cell_grid) and fall "
                              "back to the raw cluster centroid; grid-fitting needs a reasonably clean point "
                              "set to work well (pair with --dot-classifier-checkpoint) or it may do nothing "
                              "useful (falls back silently per-cell where the fit doesn't cover it)")
    parser.add_argument("--no-ruler-line-filter", dest="filter_ruler_lines", action="store_false",
                         help="[auto] disable filtering out long, dense, near-straight horizontal point runs "
                              "(dot_detect.filter_ruler_lines) before grid-fitting/clustering -- some real "
                              "Braille pages use a decorative/structural divider line between sections that "
                              "isn't part of any cell but reads as many closely-spaced real dots to the "
                              "detector, corrupting nearby cells' clustering and the page-wide grid fit if left "
                              "in. On by default; disable only if it's incorrectly stripping real dots from a "
                              "genuinely dense page.")
    parser.add_argument("--checkpoint", type=str,
                         default="braille_cnn/checkpoints/braille_cnn_dbsi_finetuned.pt")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--margin-scale", type=float, default=0.2,
                         help="[fixed-grid] extra padding around each cell as a fraction of cell PITCH")
    parser.add_argument("--cell-margin-scale", type=float, default=None,
                         help="[auto] extra padding around each cell as a fraction of the measured dot-span. "
                              "Defaults depend on --crop-shape: 0.8 for 'dbsi' (matches DBSIDataset's convention, "
                              "see dbsi_dataset.py's _cell_box) or 0.15 for 'angelina' (matches AngelinaDataset's "
                              "own margin_scale default -- confirmed empirically to be near-optimal on a real "
                              "photo; both 0.3 and 0.8 measurably hurt accuracy there).")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "si"])
    parser.add_argument("--conf-threshold", type=float, default=0.0,
                        help="cells with softmax confidence below this are shown as '_' in transcription")
    parser.add_argument("--debug-out", type=str, default=None,
                         help="optional path to save an overlay image showing the detected/assumed grid")
    args = parser.parse_args()

    if args.camera:
        from camera_capture.camera import run_camera
        run_camera(args)
        return

    if args.image is None:
        parser.error("--image is required unless --camera is set")
    if not args.auto and (args.rows is None or args.cols is None):
        parser.error("--rows and --cols are required unless --auto is set")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    image = Image.open(args.image).convert("L")

    if args.auto:
        run_auto(image, args, model=model, device=device)
    else:
        run_fixed_grid(image, args, model=model, device=device)


if __name__ == "__main__":
    main()
