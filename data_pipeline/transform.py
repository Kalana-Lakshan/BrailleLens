"""Stage 2d - Transformation: page pixels -> the tensor the CNN actually sees.

Two groups of functions:

  geometry   extract_crop()   box + margin -> fixed-size grayscale crop
  values     to_model_input() uint8 crop -> normalized float32, and
             Augmenter        photometric + geometric jitter for real crops

On scale normalization
---------------------
There is no explicit "rescale to a common dot pitch" step, because resizing
every cell box to the same output size already achieves it, *provided* a box
means the same thing in every dataset. Stage 2a guarantees that: DSBI's
dot-extent box is expanded to a full cell, Angelina's boxes are annotated
full-cell, so both are the same object before the resize.

What genuinely differs is how much blank paper each dataset's annotator left
around the dots, which changes the fraction of the 64x64 crop the dots fill.
That is what SOURCE_MARGINS corrects, and analyze.py measures the dot-fill
fraction per source so these numbers can be tuned against real data instead of
guessed at.

On augmenting real crops
-----------------------
Until now only synthetic crops were augmented (braille_cnn/render.py) while
DSBI and Angelina crops went in raw. That is backwards for domain transfer: the
real crops are the scarce ones and the ones whose capture conditions vary, so
they are the ones that benefit most from jitter.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from braille_cnn.normalize import normalize_crop

IMG_SIZE_DEFAULT = 64

# Extra context around a cell box, as a fraction of box size. Tuned per source
# so the dots fill a comparable share of the output crop. DSBI boxes already
# carry an 0.8 grid margin from Stage 2a, so they need almost none here.
SOURCE_MARGINS = {
    "dbsi": 0.05,
    "angelina": 0.15,
    "gold": 0.15,
}
DEFAULT_MARGIN = 0.15


def margin_for(source: str) -> float:
    return SOURCE_MARGINS.get(str(source), DEFAULT_MARGIN)


def extract_crop(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    margin: float = DEFAULT_MARGIN,
    img_size: int = IMG_SIZE_DEFAULT,
) -> np.ndarray | None:
    """Crop one cell from a grayscale page and resize to img_size square.

    `image` must be a 2-D uint8 array. Returns None when the box lies wholly
    outside the page, which the caller should count rather than crash on.
    """
    height, width = image.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in box)
    box_w, box_h = x1 - x0, y1 - y0
    if box_w <= 0 or box_h <= 0:
        return None

    mx, my = box_w * margin, box_h * margin
    ix0 = int(round(max(x0 - mx, 0)))
    iy0 = int(round(max(y0 - my, 0)))
    ix1 = int(round(min(x1 + mx, width)))
    iy1 = int(round(min(y1 + my, height)))
    if ix1 - ix0 < 2 or iy1 - iy0 < 2:
        return None

    crop = image[iy0:iy1, ix0:ix1]
    # INTER_AREA when shrinking avoids the aliasing that makes small dots
    # disappear; INTER_CUBIC when growing keeps them smooth.
    interp = cv2.INTER_AREA if crop.shape[0] > img_size else cv2.INTER_CUBIC
    return cv2.resize(crop, (img_size, img_size), interpolation=interp)


def to_model_input(crop_uint8: np.ndarray) -> np.ndarray:
    """uint8 crop -> float32 in [0, 1], mean-centred at 0.5.

    Thin wrapper over braille_cnn.normalize.normalize_crop so training,
    evaluation and live inference cannot drift apart on preprocessing.
    """
    return normalize_crop(crop_uint8)


@dataclass
class Augmenter:
    """Photometric and mild geometric jitter for one 64x64 uint8 crop.

    Geometric ranges are deliberately small. A Braille cell is a 2x3 dot
    lattice, so a large rotation or shear can move a dot into a neighbouring
    lattice position and turn the crop into a genuinely different class - the
    augmentation would be relabelling the data.
    """

    rotation_deg: float = 6.0
    perspective: float = 0.02
    translate_frac: float = 0.06
    scale_jitter: float = 0.08
    brightness: float = 0.25
    contrast: float = 0.25
    blur_prob: float = 0.25
    noise_std: float = 6.0
    probability: float = 0.8

    def __call__(self, crop: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        if rng.random() > self.probability:
            return crop
        out = self._geometry(crop, rng)
        return self._photometry(out, rng)

    def _geometry(self, crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        size = crop.shape[0]
        centre = (size / 2.0, size / 2.0)
        angle = rng.uniform(-self.rotation_deg, self.rotation_deg)
        scale = 1.0 + rng.uniform(-self.scale_jitter, self.scale_jitter)
        matrix = cv2.getRotationMatrix2D(centre, angle, scale)
        matrix[0, 2] += rng.uniform(-self.translate_frac, self.translate_frac) * size
        matrix[1, 2] += rng.uniform(-self.translate_frac, self.translate_frac) * size
        out = cv2.warpAffine(
            crop, matrix, (size, size),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        )

        if self.perspective > 0:
            jitter = self.perspective * size
            src = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
            dst = src + rng.uniform(-jitter, jitter, src.shape).astype(np.float32)
            transform = cv2.getPerspectiveTransform(src, dst)
            out = cv2.warpPerspective(
                out, transform, (size, size),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
            )
        return out

    def _photometry(self, crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = crop.astype(np.float32)
        out *= 1.0 + rng.uniform(-self.contrast, self.contrast)
        out += rng.uniform(-self.brightness, self.brightness) * 255.0
        if rng.random() < self.blur_prob:
            out = cv2.GaussianBlur(out, (3, 3), sigmaX=rng.uniform(0.3, 1.0))
        if self.noise_std > 0:
            out += rng.normal(0.0, self.noise_std, out.shape)
        return np.clip(out, 0, 255).astype(np.uint8)


def dot_fill_fraction(crop: np.ndarray) -> float:
    """Share of the crop occupied by locally bright structure (embossed dots).

    A rough but source-comparable measure of how much of the crop the dots
    take up. Used by analyze.py to check whether SOURCE_MARGINS actually put
    the datasets on the same footing.
    """
    arr = crop.astype(np.float32)
    background = cv2.GaussianBlur(arr, (0, 0), sigmaX=max(crop.shape[0] / 8.0, 1.0))
    residual = arr - background
    std = float(residual.std())
    if std < 1e-6:
        return 0.0
    return float((residual > 2.0 * std).mean())
