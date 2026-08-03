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
   out. Peaks are scored by a LOCAL z-score (how many local-noise-std above
   its own neighborhood's mean) rather than one global brightness
   percentile/threshold -- a global cutoff tuned on one photo's lighting
   fails hard on another (confirmed empirically: settings tuned on a phone
   photo found ~1000 real dots there but only ~140 -- almost all false --
   on a much higher-resolution, evenly-lit DBSI flatbed scan, because that
   scan's dot contrast sits nowhere near the same global brightness range).
   The local z-score adapts per-region instead of needing to be retuned per
   photo/scan type.
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


def detect_dot_centers(gray, smooth_sigma=1.5, background_sigma=25, local_norm_sigma=20.0,
                        z_threshold=3.0, footprint=9, border_margin=30, peak_y_offset=0.0):
    """Returns (N, 2) array of (x, y) pixel coordinates of candidate dot highlights.

    gray: HxW uint8/float array (grayscale image).
    smooth_sigma: blurs away fine paper-grain/JPEG noise before peak-finding.
    background_sigma: estimates the broad lighting gradient to divide out;
        must be much larger than a single dot so it doesn't blur away the
        dots themselves, only the slow shading across the page.
    local_norm_sigma: neighborhood size (pixels) used to compute each pixel's
        local mean/std of the flattened signal, i.e. how "normal" the local
        noise floor is right there -- should span several cells so it's a
        meaningful noise estimate, not just the candidate dot's own signal.
    z_threshold: how many local standard deviations above the local mean a
        peak must be to count as a dot. Tuned empirically (~3.0) -- lower
        picks up paper texture as false dots, higher misses fainter ones.
    footprint: non-max-suppression window (pixels); should be roughly one
        dot's diameter so one blob isn't counted as multiple peaks.
    border_margin: pixels near the image edge to exclude from detection. The
        physical page/scan boundary (or a photo's frame edge) is a strong,
        sharp brightness transition that reads as a highlight peak just like
        a real dot, but isn't one -- confirmed empirically to be a real,
        distinct source of false positives (~8% of all detections on a test
        DBSI scan), separate from paper texture or lighting noise.
    peak_y_offset: added to every detected y-coordinate. Confirmed empirically
        (averaging the z-field over ~1000+ ground-truth dot positions, on 5
        different DBSI books) that this detector's peak lands a consistent
        ~2-4px ABOVE the true dot center, not sub-pixel noise -- std as low
        as 0.64px on one page. The z-field is elevated in a broad band above
        the true center and cleanly negative below it (visible directly in
        the averaged window), i.e. this is a real asymmetric brightness
        signature (a raking/directional-light effect, despite DBSI's own
        paper describing "uniform illumination"), not a detector bug a
        smarter peak-finder (e.g. sub-pixel weighted centroid) can correct --
        the underlying signal itself is genuinely off-center, so it has to
        be corrected as a calibration offset instead. Defaults to 0.0 (no
        correction) since this was only measured on DBSI's specific scanner;
        do not assume the same bias, direction, or magnitude on a different
        capture setup (e.g. a phone camera) without re-measuring it there
        first the same way (see chat history for the measurement method).
    """
    img = np.asarray(gray, dtype=np.float32)
    smooth = cv2.GaussianBlur(img, (0, 0), sigmaX=smooth_sigma)
    background = cv2.GaussianBlur(img, (0, 0), sigmaX=background_sigma)
    diff = smooth - background

    local_mean = cv2.GaussianBlur(diff, (0, 0), sigmaX=local_norm_sigma)
    local_sqmean = cv2.GaussianBlur(diff * diff, (0, 0), sigmaX=local_norm_sigma)
    local_std = np.sqrt(np.clip(local_sqmean - local_mean ** 2, 1e-6, None))
    z = (diff - local_mean) / (local_std + 1.0)

    local_max = ndimage.maximum_filter(z, size=footprint)
    peaks = (z == local_max) & (z > z_threshold)
    if border_margin > 0:
        peaks[:border_margin, :] = False
        peaks[-border_margin:, :] = False
        peaks[:, :border_margin] = False
        peaks[:, -border_margin:] = False
    ys, xs = np.nonzero(peaks)
    ys = ys.astype(np.float64) + peak_y_offset
    return np.stack([xs, ys], axis=1).astype(np.float64)


def estimate_link_distance(points, diagonal_factor=1.5, fallback=15.0, min_points=20):
    """Estimates the within-cell dot-linking distance from the image's own
    dot spacing, instead of assuming one fixed pixel value works for every
    photo.

    Real Braille dot pitch is roughly fixed physically, but its size *in
    pixels* varies hugely with photo resolution and camera distance -- a
    handheld photo of a full page can have a dot pitch several times smaller
    than a flatbed scan of one page region (confirmed empirically: a 200dpi
    DBSI scan has a ~14-18px median 1st/2nd-nearest-neighbor dot spacing
    with a clear jump to ~29px for the next cell, while a lower-resolution
    phone photo of a full page measured ~11-15.6px with no such clean gap --
    right on top of the old fixed default of 15, causing cells to fragment).

    Each point's 1st-nearest-neighbor distance is used as a proxy for the
    tighter of the two straight (row or column) within-cell dot pitches --
    for a populated cell, the closest dot to any interior point is almost
    always a straight neighbor, not a diagonal or cross-cell one. Scaling
    the median of these up by diagonal_factor (~sqrt(2)) extends the radius
    to also reliably link diagonal same-cell pairs (e.g. only dots 1 and 5
    active), which real Braille geometry places only a bit closer than the
    gap to the next cell -- so this is a real, if narrow, margin rather than
    a fully robust guarantee (see run_auto_transcribe's "merged" flag for the
    remaining failure mode: cells still fusing together on tight photos).
    """
    n = len(points)
    if n < min_points:
        return fallback
    tree = cKDTree(points)
    nn_dist, _ = tree.query(points, k=2)
    base_pitch = float(np.median(nn_dist[:, 1]))
    if base_pitch <= 0:
        return fallback
    return base_pitch * diagonal_factor


def filter_ruler_lines(points, y_tol=3.0, max_gap_factor=1.5, min_chain_points=10,
                        min_span_fraction=0.35):
    """Removes points belonging to long, dense, near-straight horizontal
    runs -- decorative/structural separator lines some real Braille pages
    use between sections, which are not part of any cell but read as many
    closely-spaced raised dots to the detector. Confirmed on a real user
    photo (test-img3.jpeg): a visible dashed horizontal divider line
    produced a solid run of detector points dense enough to flag every
    cluster along it as "merged", and risks injecting noise into
    fit_cell_grid's periodicity search for any real text line near it (its
    points land in the same horizontal/vertical displacement histograms
    used to estimate dx/dy/Px/Py).

    Points are linked (mirroring cluster_into_cells' connected-components
    approach) if they're within y_tol vertically and max_gap horizontally
    of each other -- chaining like this (rather than one fixed y-bin)
    tolerates a photo's slight overall tilt, since each individual link
    only needs a small y-difference even though the whole chain can drift
    further over its length. A resulting chain long AND wide enough to
    only plausibly be a deliberate straight line is dropped entirely.

    max_gap_factor is deliberately tight (1.5x the overall nearest-neighbor
    scale, not a looser multiple like ~3.5x): a wider gap bridges real
    same-subrow dots across DIFFERENT cells too (adjacent cells sharing,
    say, their top-left dot are Px apart, not dx) -- confirmed this was a
    real, severe false-positive mode on DBSI at max_gap_factor=3.5: dense
    real text easily produces 14-25+ dots at the exact same sub-row height
    across most of a line's width (there are only 3 possible sub-row
    heights per line, so many cells sharing one by chance is normal, not
    exceptional), which looks identical to a real divider line under a
    loose gap/span/count check alone and cost DBSI ~37 accuracy points
    (97.4% to 60.4%) by stripping real dots wholesale. The measured true
    divider-line pitch on test-img3.jpeg was ~8-13px, well under any real
    dx/Px on either DBSI or Angelina, so a tight max_gap (comfortably above
    the divider's own pitch, comfortably below a real cross-cell gap)
    empirically separates the two cleanly: confirmed to remove 0 points on
    both DBSI and Angelina test pages while still fully removing the
    test-img3.jpeg divider line's ~48 points.
    """
    n = len(points)
    if n < min_chain_points:
        return points
    tree = cKDTree(points)
    nn_dist, _ = tree.query(points, k=2)
    scale = float(np.median(nn_dist[:, 1]))
    if scale <= 0:
        return points
    max_gap = scale * max_gap_factor
    page_width = points[:, 0].max() - points[:, 0].min()
    if page_width <= 0:
        return points

    search_r = max(y_tol, max_gap)
    pairs = tree.query_pairs(r=search_r, output_type="ndarray")
    if len(pairs) == 0:
        return points
    dxp = np.abs(points[pairs[:, 0], 0] - points[pairs[:, 1], 0])
    dyp = np.abs(points[pairs[:, 0], 1] - points[pairs[:, 1], 1])
    linked = pairs[(dyp <= y_tol) & (dxp <= max_gap)]
    if len(linked) == 0:
        return points

    rows = np.concatenate([linked[:, 0], linked[:, 1]])
    cols = np.concatenate([linked[:, 1], linked[:, 0]])
    data = np.ones(len(rows))
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(graph, directed=False)

    ruler_mask = np.zeros(n, dtype=bool)
    for label in range(n_comp):
        idx = np.nonzero(labels == label)[0]
        if len(idx) < min_chain_points:
            continue
        span = points[idx, 0].max() - points[idx, 0].min()
        if span >= min_span_fraction * page_width:
            ruler_mask[idx] = True

    return points[~ruler_mask]


def cluster_into_cells(points, link_distance=None):
    """Groups nearby dot centers into per-cell clusters via connected components.

    Two points are linked if they're within link_distance of each other,
    which should be set a bit above the true within-cell dot pitch and a bit
    below the pitch to the nearest dot in an adjacent cell. If link_distance
    is None (default), it's estimated per-image from the points' own
    nearest-neighbor spacing -- see estimate_link_distance's docstring for
    why a single fixed pixel constant doesn't transfer across photos taken
    at different resolutions/distances.

    Returns a list of dicts, each: points (k,2) array, center (x, y),
    bbox (x0, y0, x1, y1), merged (bool -- True if the cluster has more
    dots than a single cell can have, i.e. this is probably 2+ real cells
    that got linked together and should not be trusted as one crop).
    """
    n = len(points)
    if n == 0:
        return []
    if link_distance is None:
        link_distance = estimate_link_distance(points)
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


# ---------------------------------------------------------------------------
# Cell-grid fitting: finds true cell-center positions from clean dot points,
# for crop centering (see infer_page.py's _cluster_crop_box).
#
# A cluster's own point centroid is a bad proxy for its cell's true center:
# a cell with only e.g. the right-column dots active (4,5,6) has all its
# points on one side, pulling the centroid ~half the intra-cell pitch away
# from the true center. Confirmed on a real DBSI page: centroid-based crop
# centers were off by a systematic ~10px (out of a ~57px crop) versus true
# cell centers, whereas a fitted-grid-based center gets that down to ~3-6px.
#
# This only works well on a clean (high-precision) point set -- e.g. after
# dot_classifier.py's verification step. Earlier attempts at this exact idea
# on raw, noisy (~44-53% precision) detector output failed outright (see
# chat history / README): the noise was dense enough to corrupt the pitch
# and phase estimates themselves, not just add a few stray points.
# ---------------------------------------------------------------------------

def _robust_peak(values, lo, hi, tol=2.0, step=0.5):
    """Value in [lo, hi] with the most other values within `tol` of it --
    a tolerance-based mode finder, robust to a real cluster of values being
    split across adjacent histogram bins (which a plain histogram isn't)."""
    values = values[(values >= lo) & (values <= hi)]
    if len(values) == 0:
        return None, 0
    best_v, best_c = None, -1
    for c in np.arange(lo, hi, step):
        cnt = int(np.sum(np.abs(values - c) <= tol))
        if cnt > best_c:
            best_c, best_v = cnt, c
    return best_v, best_c


def _wrapped_dist(m, targets, period):
    return np.min([np.minimum(np.abs(m - t), period - np.abs(m - t)) for t in targets], axis=0)


def _best_phase(coords, period, suboffsets, tol=4.0, step=1.0):
    """Phase in [0, period) whose {phase + suboffsets} targets are closest
    to the most points (mod period) -- e.g. suboffsets=[0, dx] for the two
    dot-columns of a cell, or [0, dy, 2*dy] for the three dot-rows."""
    m = coords % period
    best_phase, best_score = 0.0, -1
    for phase in np.arange(0, period, step):
        targets = (phase + np.array(suboffsets)) % period
        score = int(np.sum(_wrapped_dist(m, targets, period) <= tol))
        if score > best_score:
            best_score, best_phase = score, phase
    return best_phase, best_score


def _snap_phase(phase, ref_phase, dx, Px):
    """Of {phase, phase-dx, phase+dx}, returns whichever lands closest to
    ref_phase modulo Px."""
    best_ph, best_dist = phase, None
    for cand in (phase, phase - dx, phase + dx):
        dist = abs((cand - ref_phase + Px / 2) % Px - Px / 2)
        if best_dist is None or dist < best_dist:
            best_dist, best_ph = dist, cand
    return best_ph


def _resolve_column_ambiguity(line_phase_x, dx, Px, weights):
    """Fixes a column-swap ambiguity in independently-fit per-line x-phases.

    A cell's two dot columns are dx apart, so fitting phase_x per line via
    _best_phase(..., [0, dx]) can lock onto "the right column is offset 0"
    instead of "the left column is offset 0" -- indistinguishable from x
    positions alone on a line whose own dot pattern happens to favor that
    reading. Confirmed on a real DBSI page: independently-fit per-line
    phases split into two clean clusters exactly dx apart (mod Px, residual
    <2.5px on either side) rather than genuine indentation differences,
    which cost the page ~65 accuracy points end-to-end (30% vs the 97%
    class-level accuracy confirmed on ground-truth crops from the exact
    same page) since half the lines' crops were centered a full dx off from
    every true cell.

    Resolves each line against its immediate NEIGHBOR in line order (not one
    single global reference) -- confirmed necessary on a real Angelina
    photo: unlike DBSI's flat scan, a handheld photo's per-line phase can
    genuinely drift smoothly line-to-line (perspective skew), and snapping
    every line to one fixed global reference fights that real drift instead
    of just fixing the binary dx ambiguity, measurably hurting accuracy
    there (20.7% vs 43.2%). Chaining from the most-supported line outward
    through physical line order tracks real gradual drift (each snap
    compares to an already-resolved *neighbor*, not a possibly-distant
    reference) while still correcting an isolated dx-flip, since a flip is a
    one-line outlier against its neighbors either way.
    """
    if len(line_phase_x) < 2:
        return line_phase_x
    lines_sorted = sorted(line_phase_x.keys())
    anchor = max(line_phase_x, key=lambda li: weights.get(li, 0))
    anchor_pos = lines_sorted.index(anchor)
    resolved = {anchor: line_phase_x[anchor]}
    for i in range(anchor_pos - 1, -1, -1):
        li, prev_li = lines_sorted[i], lines_sorted[i + 1]
        resolved[li] = _snap_phase(line_phase_x[li], resolved[prev_li], dx, Px)
    for i in range(anchor_pos + 1, len(lines_sorted)):
        li, prev_li = lines_sorted[i], lines_sorted[i - 1]
        resolved[li] = _snap_phase(line_phase_x[li], resolved[prev_li], dx, Px)
    return resolved


def _refine_y_fit(points, dx, dy, Px, Py, phase_y, tol):
    """Least-squares polish of (phase_y, Py, dy) using ALL points at once
    (not the coarse per-value tolerance-voting search), given each point's
    row/line assignment under the coarse fit. y_i = phase_y + Py*line_i +
    dy*sub_row_i is a plain linear model in (line_i, sub_row_i) -- solving it
    over every confidently-assigned point uses far more information than the
    coarse fit's single best-bin search, and removes that search's 0.5px/1px
    step-size quantization.
    """
    line_idx = np.round((points[:, 1] - phase_y) / Py)
    row_pos = points[:, 1] - phase_y - line_idx * Py
    sub_row = np.clip(np.round(row_pos / dy), 0, 2)
    resid = np.abs(row_pos - sub_row * dy)
    keep = resid <= tol
    if keep.sum() < 20:
        return phase_y, Py, dy
    design = np.stack([np.ones(keep.sum()), line_idx[keep], sub_row[keep]], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, points[keep, 1], rcond=None)
    new_phase_y, new_Py, new_dy = coeffs
    if new_Py <= 0 or new_dy <= 0:
        return phase_y, Py, dy
    return float(new_phase_y), float(new_Py), float(new_dy)


def _refine_x_fit(points, line_idx, dx, Px, line_phase_x, tol):
    """Same idea as _refine_y_fit but for x, with a separate phase per line
    (one-hot column per line) since indentation varies line to line while
    the pitch (Px, dx) is one shared physical property of the whole page.
    """
    lines = sorted(li for li in np.unique(line_idx) if li in line_phase_x)
    if len(lines) < 3:
        return Px, dx, line_phase_x
    line_pos_map = {li: i for i, li in enumerate(lines)}

    col_idx = np.empty(len(points))
    sub_col = np.empty(len(points))
    resid = np.empty(len(points))
    for li in lines:
        mask = line_idx == li
        phase_x = line_phase_x[li]
        pos = points[mask, 0] - phase_x
        ci = np.round(pos / Px)
        sc = np.clip(np.round((pos - ci * Px) / dx), 0, 1)
        col_idx[mask] = ci
        sub_col[mask] = sc
        resid[mask] = np.abs(pos - ci * Px - sc * dx)

    keep = np.array([resid[i] <= tol and line_idx[i] in line_phase_x for i in range(len(points))])
    if keep.sum() < 20:
        return Px, dx, line_phase_x

    n_lines = len(lines)
    design = np.zeros((keep.sum(), n_lines + 2))
    kept_lines = line_idx[keep]
    for row_i, li in enumerate(kept_lines):
        design[row_i, line_pos_map[li]] = 1.0
    design[:, n_lines] = col_idx[keep]
    design[:, n_lines + 1] = sub_col[keep]
    coeffs, *_ = np.linalg.lstsq(design, points[keep, 0], rcond=None)
    new_Px, new_dx = coeffs[n_lines], coeffs[n_lines + 1]
    if new_Px <= 0 or new_dx <= 0:
        return Px, dx, line_phase_x
    new_line_phase_x = {int(li): float(coeffs[line_pos_map[li]]) for li in lines}
    return float(new_Px), float(new_dx), new_line_phase_x


def fit_cell_grid(points, min_points=40, refine=True):
    """Fits the page's regular cell grid from a clean set of dot points.

    Returns a dict (dx, dy, Px, Py, phase_y, line_phase_x) or None if there
    aren't enough points to fit reliably. x-phase is fit PER LINE (not one
    global value) since where a line of text starts varies with indentation/
    content -- only the pitch (a fixed physical property of the embosser) is
    assumed constant across the whole page; y-phase is fit once globally
    since lines stack vertically in a regular, content-independent way.

    refine=True runs a least-squares polish pass on top of the coarse
    tolerance-voting fit (see _refine_x_fit/_refine_y_fit) -- confirmed on a
    held-out DBSI page to noticeably improve downstream classification
    accuracy over the coarse fit alone (the coarse search is quantized to
    its step size and only locally optimal one axis at a time).
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < min_points:
        return None

    # Scale anchor: overall 1st-nearest-neighbor distance, not restricted to
    # one axis. This is resolution/distance-agnostic (a photo taken closer
    # or scanned at a different DPI just scales this number), unlike a fixed
    # pixel range -- confirmed necessary empirically: search bounds tuned
    # for DBSI's scale (dx/dy ~20px) found completely wrong values (~2x too
    # large for dx, ~1.5x too small for dy) on Angelina's photos, which have
    # a different physical scale (cell width ~25px but a much bigger
    # relative gap between cells) -- so a photo-specific pitch has to be
    # re-derived from the photo's own points, not assumed from one dataset.
    tree = cKDTree(points)
    nn_dist, _ = tree.query(points, k=2)
    scale_anchor = float(np.median(nn_dist[:, 1]))
    if scale_anchor <= 0:
        return None

    search_radius = max(100.0, scale_anchor * 10)
    pairs = tree.query_pairs(r=search_radius, output_type="ndarray")
    if len(pairs) == 0:
        return None
    disp = points[pairs[:, 1]] - points[pairs[:, 0]]

    horiz = np.abs(disp[np.abs(disp[:, 1]) < 6][:, 0])
    vert = np.abs(disp[np.abs(disp[:, 0]) < 6][:, 1])

    # Vertical periodicity is the cleaner signal to estimate the fundamental
    # dot pitch from: horizontal has an extra confound vertical doesn't --
    # real text has inter-WORD gaps (a third, larger structural spacing) on
    # top of intra-cell and inter-cell spacing, and that third peak can
    # outvote the true (smaller) intra-cell pitch in an unconstrained global
    # search. Confirmed on a real photo: the single strongest horizontal
    # peak (143 votes) was ~2.5x the true dot pitch found cleanly on the
    # vertical axis (only two peaks there, cleanly harmonically related).
    # Standard Braille dot spacing is physically ~isotropic (nearly the same
    # horizontal and vertical pitch by spec), so dy anchors the dx search
    # range instead of searching dx independently and risking exactly that
    # word-gap confound. Falls back to an unconstrained search if nothing
    # is found near dy (e.g. a genuinely different layout).
    #
    # The anchor window has to be kept tight (~+-35%), not the physically
    # "safe"-looking 0.6-1.7x: a real photo has another confound at almost
    # exactly Px-dx (the gap from one cell's rightmost dot column to the
    # next cell's leftmost one) which a wide window happily includes.
    # Confirmed on an Angelina photo: true dx=14 (verified directly from
    # points matched to their ground-truth cell), but Px-dx=33-14=19 sat
    # inside the old [dy*0.6, dy*1.7] window and had denser local support
    # there than the true, wider-spread 12-16 cluster, so the tolerance
    # search locked onto ~19 instead -- which then pushed the Px search's
    # lower bound (dx*1.8) past the true Px peak entirely and onto its 2x
    # harmonic. A tighter window keeps that Px-dx confound (~1.4-1.9x dy
    # once dx is even a little too high) out of range from the start.
    dy, _ = _robust_peak(vert, scale_anchor * 0.4, scale_anchor * 2.2)
    if dy is None:
        return None
    dx, dx_votes = _robust_peak(horiz, dy * 0.75, dy * 1.35)
    if dx is None:
        dx, _ = _robust_peak(horiz, scale_anchor * 0.4, scale_anchor * 2.2)
    Px, _ = _robust_peak(horiz, max(scale_anchor * 2.2, dx * 1.8 if dx else 0), scale_anchor * 8)
    Py, _ = _robust_peak(vert, scale_anchor * 2.2, scale_anchor * 8)
    if None in (dx, dy, Px, Py):
        return None

    phase_y, y_score = _best_phase(points[:, 1], Py, [0, dy, 2 * dy])
    if y_score < 0.35 * len(points):
        return None  # y-periodicity too weak to trust -- e.g. too few points, or real skew

    if refine:
        # One pass reuses its own coarse (possibly biased) point-to-row
        # assignment, which tends to reinforce rather than correct that
        # bias -- confirmed empirically (a single pass left a ~4px
        # systematic y-offset essentially unchanged). A few iterations,
        # reassigning points from each pass's improved parameters before
        # refitting, lets it actually converge instead of getting stuck.
        for _ in range(4):
            phase_y, Py, dy = _refine_y_fit(points, dx, dy, Px, Py, phase_y, tol=dy * 0.4)

    line_idx = np.round((points[:, 1] - phase_y) / Py).astype(int)
    line_phase_x = {}
    line_weights = {}
    for li in np.unique(line_idx):
        mask = line_idx == li
        if mask.sum() < 4:
            continue
        ph, _ = _best_phase(points[mask, 0], Px, [0, dx])
        line_phase_x[int(li)] = ph
        line_weights[int(li)] = int(mask.sum())

    line_phase_x = _resolve_column_ambiguity(line_phase_x, dx, Px, line_weights)

    if refine:
        for _ in range(4):
            Px, dx, line_phase_x = _refine_x_fit(points, line_idx, dx, Px, line_phase_x, tol=dx * 0.4)

    return {"dx": dx, "dy": dy, "Px": Px, "Py": Py, "phase_y": phase_y, "line_phase_x": line_phase_x}


def grid_cell_center(anchor_point, grid):
    """Given a rough (x, y) anchor inside a cell (e.g. a cluster's point
    centroid) and a grid from fit_cell_grid, returns that cell's true
    center -- robust to asymmetric dot patterns that bias the anchor itself,
    since it's derived from the page-wide fitted grid, not the cell's own
    (possibly incomplete/lopsided) points. Returns None if this cell's line
    wasn't in the fitted grid (e.g. too few points on that line).
    """
    ax, ay = anchor_point
    dx, dy, Px, Py = grid["dx"], grid["dy"], grid["Px"], grid["Py"]
    line_idx = int(round((ay - grid["phase_y"]) / Py))
    phase_x = grid["line_phase_x"].get(line_idx)
    if phase_x is None:
        return None
    col_idx = round((ax - phase_x) / Px)
    return (phase_x + col_idx * Px + dx / 2.0, grid["phase_y"] + line_idx * Py + dy)


def cluster_by_grid(points, grid):
    """Groups points into cells by which fitted grid slot they fall in,
    instead of cluster_into_cells' geometric distance threshold.

    Why this exists: distance-based clustering can quietly fuse two
    different real cells into one "valid" (not merged-flagged) cluster
    whenever their combined point count stays <= MAX_DOTS_PER_CELL -- e.g. a
    1-dot cell next to a 2-dot cell (3 points total, well under the 6-dot
    merge-detection threshold) still spans two true cells. Confirmed on a
    held-out DBSI page: distance-based clustering produced 473 "valid"
    clusters for 618 true cells -- ~150 true cells' dots were getting
    silently absorbed into a neighboring cell's cluster. Grid-based
    assignment sidesteps this entirely: a point is assigned to whichever
    grid slot it's actually closest to, regardless of geometric distance to
    other points, so two real cells can never merge as long as the grid fit
    itself is accurate.

    Only usable where fit_cell_grid succeeded; points whose line isn't
    covered by the grid are returned as their own single-point clusters
    (same as before -- better to keep them than silently drop them).

    Returns the same cluster dict shape as cluster_into_cells.
    """
    dx, dy, Px, Py = grid["dx"], grid["dy"], grid["Px"], grid["Py"]
    phase_y, line_phase_x = grid["phase_y"], grid["line_phase_x"]

    slots = {}
    for x, y in points:
        line_idx = int(round((y - phase_y) / Py))
        phase_x = line_phase_x.get(line_idx)
        if phase_x is None:
            slots.setdefault(("_unassigned", x, y), []).append((x, y))
            continue
        col_idx = round((x - phase_x) / Px)
        slots.setdefault((line_idx, col_idx), []).append((x, y))

    clusters = []
    for pts in slots.values():
        pts = np.array(pts)
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
