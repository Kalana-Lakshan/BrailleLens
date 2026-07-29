"""Automatic Braille-dot detection and per-cell clustering for real photos.

Alternative to infer_page.py's fixed-grid mode: instead of the caller
guessing --rows/--cols (which assumes uniform pixel spacing across the whole
image -- wrong for a handheld photo with skew/perspective, or for prose text
where line length varies), this finds actual dot highlights in the image and
groups nearby ones into per-cell clusters directly from their measured
positions. No knowledge of the page layout is required upfront.

Two-stage approach:
1. detect_dot_centers: raised dots catch light and read as local brightness
   peaks once the broad lighting gradient (vignette/raking light) is divided
   out. A percentile threshold on the flattened image, plus non-max
   suppression, gives candidate dot pixel coordinates.
2. cluster_into_cells: dots belonging to the same cell are much closer to
   each other (within-cell dot pitch) than dots in neighboring cells or
   lines, so connected components under a distance threshold recovers per-
   cell groups. A cell has at most 6 dots -- any cluster bigger than that is
   necessarily two or more real cells that got merged (happens where local
   pitch shrinks relative to the fixed threshold, e.g. from perspective) and
   is flagged rather than guessed at.
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

MAX_DOTS_PER_CELL = 6


def detect_dot_centers(gray, smooth_sigma=1.5, background_sigma=25, percentile=99.3, footprint=9):
    """Returns (N, 2) array of (x, y) pixel coordinates of candidate dot highlights.

    gray: HxW uint8/float array (grayscale image).
    smooth_sigma: blurs away fine paper-grain/JPEG noise before peak-finding.
    background_sigma: estimates the broad lighting gradient to divide out;
        must be much larger than a single dot so it doesn't blur away the
        dots themselves, only the slow shading across the page.
    percentile: brightness-difference cutoff for a peak to count as a dot;
        tuned empirically (99.3) against a real embossed-page test photo --
        lower picks up paper texture as false dots, higher misses faint ones.
    footprint: non-max-suppression window (pixels); should be roughly one
        dot's diameter so one blob isn't counted as multiple peaks.
    """
    img = np.asarray(gray, dtype=np.float32)
    smooth = cv2.GaussianBlur(img, (0, 0), sigmaX=smooth_sigma)
    background = cv2.GaussianBlur(img, (0, 0), sigmaX=background_sigma)
    diff = smooth - background

    thresh = np.percentile(diff, percentile)
    local_max = ndimage.maximum_filter(diff, size=footprint)
    peaks = (diff == local_max) & (diff > thresh)
    ys, xs = np.nonzero(peaks)
    return np.stack([xs, ys], axis=1).astype(np.float64)


def cluster_into_cells(points, link_distance=15.0):
    """Groups nearby dot centers into per-cell clusters via connected components.

    Two points are linked if they're within link_distance of each other,
    which should be set a bit above the true within-cell dot pitch (measure
    it from a nearest-neighbor-distance histogram of detect_dot_centers'
    output) and a bit below the pitch to the nearest dot in an adjacent cell.

    Returns a list of dicts, each: points (k,2) array, center (x, y),
    bbox (x0, y0, x1, y1), merged (bool -- True if the cluster has more
    dots than a single cell can have, i.e. this is probably 2+ real cells
    that got linked together and should not be trusted as one crop).
    """
    n = len(points)
    if n == 0:
        return []
    tree = cKDTree(points)
    pairs = list(tree.query_pairs(r=link_distance))
    if pairs:
        rows = [p[0] for p in pairs] + [p[1] for p in pairs]
        cols = [p[1] for p in pairs] + [p[0] for p in pairs]
        data = [1] * len(rows)
        graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    else:
        graph = csr_matrix((n, n))
    n_comp, labels = connected_components(graph, directed=False)

    clusters = []
    for label in range(n_comp):
        pts = points[labels == label]
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        cx, cy = pts.mean(axis=0)
        clusters.append({
            "points": pts,
            "bbox": (float(x0), float(y0), float(x1), float(y1)),
            "center": (float(cx), float(cy)),
            "merged": len(pts) > MAX_DOTS_PER_CELL,
        })
    return clusters
