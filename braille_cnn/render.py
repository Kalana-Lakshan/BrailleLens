"""Procedural renderer for synthetic Braille cell images.

Each Braille cell has 6 dot positions numbered:
    1 4
    2 5
    3 6
A cell's raised-dot pattern is encoded as a 6-bit integer 0-63, where
bit (dot - 1) is set if that dot is raised. This gives all 64 combinations,
including 0 (blank / space).
"""

import numpy as np
from PIL import Image, ImageFilter

IMG_SIZE_DEFAULT = 64

DOT_GRID = {  # dot number -> (row, col); row 0=top..2=bottom, col 0=left,1=right
    1: (0, 0), 2: (1, 0), 3: (2, 0),
    4: (0, 1), 5: (1, 1), 6: (2, 1),
}


def active_dots(code: int) -> list:
    return [dot for dot in range(1, 7) if code & (1 << (dot - 1))]


def _perspective_coeffs(source_pts, target_pts):
    """Coefficients for PIL's Image.transform(..., Image.PERSPECTIVE, ...),
    which for each output pixel (x, y) samples from
    ((a x + b y + c) / (g x + h y + 1), (d x + e y + f) / (g x + h y + 1))
    in the input image -- i.e. an output->input mapping. Passing
    source_pts=output rectangle corners and target_pts=perturbed input
    corners gives exactly that.
    """
    matrix = []
    for (x, y), (X, Y) in zip(source_pts, target_pts):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=np.float64)
    B = np.array(target_pts, dtype=np.float64).reshape(8)
    return np.linalg.solve(A, B)


def _apply_perspective(pil_img, strength, rng, fillcolor):
    """Warps the image as if the camera were viewing it from an angle
    (independent per-corner jitter -> arbitrary trapezoid/skew), rather than
    the simple in-plane rotation .rotate() applies.
    """
    w, h = pil_img.size
    max_shift_x = strength * w
    max_shift_y = strength * h
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    perturbed = [
        (x + rng.uniform(-max_shift_x, max_shift_x), y + rng.uniform(-max_shift_y, max_shift_y))
        for x, y in corners
    ]
    coeffs = _perspective_coeffs(corners, perturbed)
    return pil_img.transform(
        (w, h), Image.Transform.PERSPECTIVE, coeffs,
        resample=Image.Resampling.BICUBIC, fillcolor=fillcolor,
    )


def _dot_centers(img_size, cell_scale, jitter, rng):
    cx, cy = img_size / 2, img_size / 2
    col_gap = 0.30 * cell_scale * img_size
    row_gap = 0.28 * cell_scale * img_size
    centers = {}
    for dot, (row, col) in DOT_GRID.items():
        x = cx + (col - 0.5) * col_gap + rng.uniform(-jitter, jitter) * img_size
        y = cy + (row - 1) * row_gap + rng.uniform(-jitter, jitter) * img_size
        centers[dot] = (x, y)
    return centers


def render_braille_cell(
    code: int, img_size: int = IMG_SIZE_DEFAULT, augment: bool = True, rng=None, max_perspective: float = 0.20,
) -> Image.Image:
    """Render one Braille cell as a grayscale PIL image.

    augment=True applies random pose/lighting/noise so repeated calls with a
    fresh `rng` produce visually varied samples of the same class.
    max_perspective caps how strong the simulated camera-tilt warp can be
    (as a fraction of image size); set to 0 to disable it, e.g. to reproduce
    pre-perspective-augmentation behavior for comparison.
    """
    rng = rng if rng is not None else np.random.default_rng()
    h = w = img_size
    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)

    bg_level = rng.uniform(150, 200) if augment else 175.0
    background = np.full((h, w), bg_level, dtype=np.float32)
    noise_sigma = rng.uniform(1.0, 6.0) if augment else 2.0
    background += rng.normal(0, noise_sigma, size=(h, w)).astype(np.float32)

    cell_scale = rng.uniform(0.85, 1.15) if augment else 1.0
    jitter = rng.uniform(0.0, 0.02) if augment else 0.0
    base_radius = (rng.uniform(0.075, 0.10) if augment else 0.085) * img_size

    centers = _dot_centers(img_size, cell_scale, jitter, rng)

    height_field = np.zeros((h, w), dtype=np.float32)
    for dot in active_dots(code):
        cx, cy = centers[dot]
        r = base_radius * (rng.uniform(0.9, 1.1) if augment else 1.0)
        height_field += np.exp(-(((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * r ** 2)))

    grad_y, grad_x = np.gradient(height_field)
    light_x, light_y = (rng.uniform(-0.8, -0.3), rng.uniform(-0.8, -0.3)) if augment else (-0.6, -0.6)
    shading = -(grad_x * light_x + grad_y * light_y) * 60.0
    highlight = height_field * (rng.uniform(20, 45) if augment else 30.0)

    image = np.clip(background + shading + highlight, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image, mode="L")

    if not augment:
        return pil_img

    angle = rng.uniform(-10, 10)
    pil_img = pil_img.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=int(bg_level))

    perspective_strength = rng.uniform(0.0, max_perspective) if max_perspective > 0 else 0.0
    if perspective_strength > 0.02:
        pil_img = _apply_perspective(pil_img, perspective_strength, rng, fillcolor=int(bg_level))

    blur_radius = rng.uniform(0, 1.0)
    if blur_radius > 0.05:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(blur_radius))

    arr = np.asarray(pil_img).astype(np.float32)
    contrast = rng.uniform(0.85, 1.15)
    brightness = rng.uniform(-15, 15)
    arr = (arr - 128) * contrast + 128 + brightness
    arr += rng.normal(0, rng.uniform(0, 4), size=arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode="L")
