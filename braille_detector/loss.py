"""Loss for BrailleDetector: FocalLoss for classification (per RetinaNet,
matching the paper's L = L_loc + lambda_cls * L_cls) + smooth-L1 for box
regression, both computed only where a ground-truth box was assigned.

Target assignment: each GT box is assigned to the single feature-map cell
containing its center -- matching the paper's "one anchor per cell, every
character covered by at least one grid cell" simplification, so no IOU-
based multi-anchor matching is needed.
"""

import torch
import torch.nn.functional as F

from .boxes import NUM_CLASSES


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return alpha_t * ce * (1 - p_t).pow(gamma)


def build_targets(model, box_out_shape, boxes_list, labels_list, device):
    """boxes_list/labels_list: per-image lists of (N_i,4)/(N_i,) tensors in
    pixel coords. Returns cls_target (B,C,Hf,Wf), box_target (B,4,Hf,Wf),
    pos_mask (B,Hf,Wf) bool."""
    b, _, hf, wf = box_out_shape
    cls_target = torch.zeros(b, NUM_CLASSES, hf, wf, device=device)
    box_target = torch.zeros(b, 4, hf, wf, device=device)
    pos_mask = torch.zeros(b, hf, wf, dtype=torch.bool, device=device)

    for i in range(b):
        boxes = boxes_list[i].to(device)
        labels = labels_list[i].to(device)
        if len(boxes) == 0:
            continue
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        cell_ix = (cx / model.stride).long().clamp(0, wf - 1)
        cell_iy = (cy / model.stride).long().clamp(0, hf - 1)

        reg_targets = model.encode_boxes(boxes, cell_ix, cell_iy)  # (N,4)
        for n in range(len(boxes)):
            iy, ix = cell_iy[n].item(), cell_ix[n].item()
            pos_mask[i, iy, ix] = True
            cls_target[i, :, iy, ix] = 0.0
            cls_target[i, labels[n].item() - 1, iy, ix] = 1.0
            box_target[i, :, iy, ix] = reg_targets[n]

    return cls_target, box_target, pos_mask


def compute_loss(model, box_out, cls_out, boxes_list, labels_list, lambda_cls=1.0):
    device = box_out.device
    cls_target, box_target, pos_mask = build_targets(model, box_out.shape, boxes_list, labels_list, device)

    num_pos = max(pos_mask.sum().item(), 1)

    cls_loss = focal_loss(cls_out, cls_target).sum() / num_pos

    pos_mask_4 = pos_mask.unsqueeze(1).expand(-1, 4, -1, -1)
    if pos_mask.any():
        box_loss = F.smooth_l1_loss(box_out[pos_mask_4], box_target[pos_mask_4], reduction="sum") / num_pos
    else:
        box_loss = box_out.sum() * 0.0

    total = box_loss + lambda_cls * cls_loss
    return total, box_loss.detach(), cls_loss.detach(), num_pos
