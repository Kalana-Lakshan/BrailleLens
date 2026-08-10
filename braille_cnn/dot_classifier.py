"""Small binary CNN: is this patch centered on a real Braille dot, or not?

Verification stage for dot_detect.py, mirroring the DSBI paper's own
approach (Haar+Adaboost dot detector, F1 0.948-0.970) with a learned
classifier instead of a hand-tuned brightness threshold -- see
dot_patch_dataset.py's docstring for why: no single threshold value gets
both good precision and good recall (confirmed empirically), so this
replaces the threshold decision with a model trained on real dot appearance
vs. the existing detector's own actual false positives.
"""

import torch.nn as nn


class DotPatchCNN(nn.Module):
    """Input: (B, 1, 32, 32) grayscale patch. Output: 2-class logits (not-dot, dot)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1), nn.BatchNorm2d(12), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(24 * 4 * 4, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
