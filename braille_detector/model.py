"""Single-stage Braille-cell detector: a small CNN backbone at stride 16
(one 16x16-pixel feature-map cell per expected character, matching the
Ovodov paper's simplification of RetinaNet -- one feature-map scale, one
anchor per cell, since every Braille character is small and nearly the
same size, unlike general object detection which needs a multi-scale
pyramid) feeding a box-regression head and a class head.

This replaces braille_cnn/'s whole detect -> grid-fit -> cluster -> crop ->
classify pipeline with one forward pass: no dot detection, no grid model,
no per-cell cropping. See boxes.py's module docstring for why this lives
in its own folder instead of extending braille_cnn/.
"""

import torch
from torch import nn

from .boxes import NUM_CLASSES

STRIDE = 16
ANCHOR_W = 22.0
ANCHOR_H = 38.0


def _conv_block(in_ch, out_ch, stride):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Backbone(nn.Module):
    """4 stride-2 blocks after one stride-1 stem = total stride 16."""

    def __init__(self):
        super().__init__()
        self.stem = _conv_block(1, 32, 1)
        self.down1 = _conv_block(32, 32, 2)
        self.down2 = _conv_block(32, 64, 2)
        self.down3 = _conv_block(64, 96, 2)
        self.down4 = _conv_block(96, 128, 2)
        self.refine = _conv_block(128, 128, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.refine(x)
        return x


class BrailleDetector(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, anchor_w=ANCHOR_W, anchor_h=ANCHOR_H, stride=STRIDE):
        super().__init__()
        self.num_classes = num_classes
        self.anchor_w = anchor_w
        self.anchor_h = anchor_h
        self.stride = stride
        self.backbone = Backbone()
        self.box_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, 1),
        )
        self.cls_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )
        # Standard RetinaNet init: cls bias set so initial sigmoid outputs
        # are small (~0.01), since the vast majority of cells are
        # background -- avoids the loss being dominated by early large
        # gradients from ~2600 confidently-wrong negative predictions.
        prior = 0.01
        nn.init.constant_(self.cls_head[-1].bias, -torch.log(torch.tensor((1 - prior) / prior)).item())

    def forward(self, x):
        feat = self.backbone(x)
        box_out = self.box_head(feat)   # (B, 4, Hf, Wf)
        cls_out = self.cls_head(feat)   # (B, C, Hf, Wf)
        return box_out, cls_out

    def anchor_centers(self, feat_h, feat_w, device):
        ys = (torch.arange(feat_h, device=device, dtype=torch.float32) + 0.5) * self.stride
        xs = (torch.arange(feat_w, device=device, dtype=torch.float32) + 0.5) * self.stride
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return gx, gy  # each (Hf, Wf)

    def decode_boxes(self, box_out):
        """box_out: (B,4,Hf,Wf) regression targets -> (B,Hf,Wf,4) pixel boxes (x0,y0,x1,y1)."""
        b, _, hf, wf = box_out.shape
        gx, gy = self.anchor_centers(hf, wf, box_out.device)
        tx, ty, tw, th = box_out.unbind(dim=1)  # each (B,Hf,Wf)
        cx = gx.unsqueeze(0) + tx * self.anchor_w
        cy = gy.unsqueeze(0) + ty * self.anchor_h
        w = self.anchor_w * torch.exp(tw.clamp(-4, 4))
        h = self.anchor_h * torch.exp(th.clamp(-4, 4))
        x0, y0 = cx - w / 2, cy - h / 2
        x1, y1 = cx + w / 2, cy + h / 2
        return torch.stack([x0, y0, x1, y1], dim=-1)

    def encode_boxes(self, boxes, cell_ix, cell_iy):
        """boxes: (N,4) pixel (x0,y0,x1,y1) ground truth assigned to grid
        cells (cell_ix, cell_iy) (each (N,)) -> (N,4) regression targets."""
        gx = (cell_ix.float() + 0.5) * self.stride
        gy = (cell_iy.float() + 0.5) * self.stride
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        tx = (cx - gx) / self.anchor_w
        ty = (cy - gy) / self.anchor_h
        tw = torch.log(w.clamp(min=1e-3) / self.anchor_w)
        th = torch.log(h.clamp(min=1e-3) / self.anchor_h)
        return torch.stack([tx, ty, tw, th], dim=-1)
