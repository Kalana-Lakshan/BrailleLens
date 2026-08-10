"""Shared crop normalization for the CNN's input, used identically by
synthetic training (dataset.py), DBSI fine-tuning (dbsi_dataset.py), and
inference (infer_page.py) -- so the network is trained and evaluated on the
same representation regardless of where a crop came from.

Motivation: absolute crop brightness varies enormously across sources (DBSI's
tan-paper flatbed scans vs. a phone photo of white paper under uneven room
lighting can differ by 50+ gray levels), but the raw pixel value was
previously fed to the network unchanged (just divided by 255). A model's
early conv filters implicitly learn the *absolute* brightness range they were
trained on, so a crop lifted from a much brighter or dimmer region than that
range confuses classification -- independent of whether the dot pattern
itself is perfectly legible.
"""

import numpy as np


def normalize_crop(arr, std_floor: float = 10.0, span_std: float = 4.0) -> np.ndarray:
    """Standardizes one Braille-cell crop's brightness/contrast to a
    canonical range, independent of its absolute background tone.

    Subtracts the crop's own mean (removes absolute brightness/background
    tone) and divides by its own std (removes absolute contrast), floored at
    std_floor so a blank/no-dot cell -- which is naturally near-flat, with
    only faint background/sensor noise -- doesn't get that noise amplified
    into fake high-contrast texture. std_floor sits above typical blank-cell
    noise and below typical populated-cell contrast (empirically: synthetic
    code=0 crops have raw std up to ~5, populated codes typically 6-14+).

    Returns a float32 array clipped to [0, 1], mean-centered at 0.5.
    """
    arr = np.asarray(arr, dtype=np.float32)
    mean = arr.mean()
    std = arr.std()
    scale = max(std, std_floor)
    out = (arr - mean) / (scale * span_std) + 0.5
    return np.clip(out, 0.0, 1.0).astype(np.float32)
