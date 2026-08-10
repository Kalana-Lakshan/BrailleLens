"""Runs a trained BrailleDetector on a full page image in one forward pass
(no dot detection, no grid-fitting, no per-cell cropping) and evaluates
against ground truth using the same matched-center methodology used
throughout this session for the existing braille_cnn/ pipeline, so the
numbers are directly comparable.
"""

import argparse

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

from .boxes import nms
from .data import ANGELINA_SCALE, DBSI_SCALE, parse_angelina_csv, parse_dbsi_txt
from .model import BrailleDetector


def load_model(checkpoint_path, device):
    model = BrailleDetector().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def detect_page(image, model, device, scale, conf_threshold=0.3, nms_iou=0.02, tile=1024):
    """image: PIL grayscale image (full resolution, NOT pre-scaled).
    Returns boxes (N,4) and labels (N,) (codes, 1-63) in ORIGINAL image
    pixel coordinates, and scores (N,).

    Tiles the page if larger than `tile` (in scaled pixels) so memory use
    stays bounded on a big photo -- the model is fully convolutional so
    each tile is just an independent forward pass, stitched back together
    by offsetting box coordinates; NMS runs globally afterward to clean up
    any duplicate detections near tile seams.
    """
    w, h = image.size
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    scaled = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

    all_boxes, all_scores, all_labels = [], [], []
    for y0 in range(0, new_h, tile):
        for x0 in range(0, new_w, tile):
            x1, y1 = min(x0 + tile, new_w), min(y0 + tile, new_h)
            crop = scaled.crop((x0, y0, x1, y1))
            arr = np.asarray(crop, dtype=np.float32)
            arr = (arr - arr.mean()) / (arr.std() + 1e-6)
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(device)

            box_out, cls_out = model(tensor)
            boxes = model.decode_boxes(box_out)[0]  # (Hf,Wf,4)
            probs = torch.sigmoid(cls_out)[0]  # (C,Hf,Wf)
            max_prob, max_label = probs.max(dim=0)  # (Hf,Wf)

            keep = max_prob > conf_threshold
            if keep.sum() == 0:
                continue
            b = boxes[keep].cpu().numpy()
            b[:, [0, 2]] += x0
            b[:, [1, 3]] += y0
            all_boxes.append(b)
            all_scores.append(max_prob[keep].cpu().numpy())
            all_labels.append((max_label[keep] + 1).cpu().numpy())

    if not all_boxes:
        return np.empty((0, 4)), np.empty((0,), dtype=np.int64), np.empty((0,))

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    keep_idx = nms(boxes, scores, iou_threshold=nms_iou)
    boxes, scores, labels = boxes[keep_idx], scores[keep_idx], labels[keep_idx]
    boxes = boxes / scale  # back to original image pixel coords
    return boxes, labels, scores


def evaluate(image_path, ann_path, model, device, source, conf_threshold=0.3, match_tol=15.0):
    image = Image.open(image_path).convert("L")
    w, h = image.size
    scale = DBSI_SCALE if source == "dbsi" else ANGELINA_SCALE

    if source == "dbsi":
        true_boxes = parse_dbsi_txt(ann_path)
    else:
        true_boxes = parse_angelina_csv(ann_path, w, h)
    true_centers = np.array([((x0 + x1) / 2, (y0 + y1) / 2) for x0, y0, x1, y1, c in true_boxes])
    true_codes = np.array([c for *_, c in true_boxes])

    pred_boxes, pred_labels, pred_scores = detect_page(image, model, device, scale, conf_threshold=conf_threshold)
    pred_centers = np.array([((x0 + x1) / 2, (y0 + y1) / 2) for x0, y0, x1, y1 in pred_boxes]) \
        if len(pred_boxes) else np.empty((0, 2))

    if len(pred_centers) == 0 or len(true_centers) == 0:
        return {"n_true": len(true_centers), "n_pred": len(pred_centers), "matched": 0, "correct": 0}

    tree = cKDTree(true_centers)
    d, idx = tree.query(pred_centers, k=1)
    matched = d < match_tol
    correct = (pred_labels[matched] == true_codes[idx[matched]]).sum()
    return {
        "n_true": len(true_centers), "n_pred": len(pred_centers),
        "matched": int(matched.sum()), "correct": int(correct),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--annotation", type=str, required=True)
    parser.add_argument("--source", choices=["dbsi", "angelina"], required=True)
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    result = evaluate(args.image, args.annotation, model, device, args.source,
                       conf_threshold=args.conf_threshold)
    n_true = result["n_true"]
    print(f"true cells: {n_true}  detected: {result['n_pred']}  "
          f"matched: {result['matched']}/{n_true}  correct: {result['correct']}  "
          f"overall: {result['correct']}/{n_true} ({result['correct']/max(n_true,1)*100:.1f}%)")


if __name__ == "__main__":
    main()
