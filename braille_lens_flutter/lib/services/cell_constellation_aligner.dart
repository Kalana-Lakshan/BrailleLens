import 'dart:math';
import 'dart:ui';

import 'package:flutter/foundation.dart';

import '../utils/homography.dart';

class CellAlignment {
  final Homography homography;
  final int inliers;

  const CellAlignment({required this.homography, required this.inliers});
}

/// Aligns the stage-2 finger frame to the stage-1 prescan using the *cell
/// boxes themselves* as the common landmarks.
///
/// Pixel feature matching (ORB and friends) is unreliable here: a Braille
/// page is a near-periodic field of identical dots, so every local patch
/// looks like every other one and the ratio test throws almost everything
/// away. Cell centres, by contrast, form a distinctive global constellation.
/// No correspondences are known up front, so instead of matching points this
/// votes: for each candidate scale/rotation, every (live, reference) pair
/// casts a vote for the translation that would align it, and the winning
/// bin is the transform that the largest number of cells agree on.
///
/// Centres under the fingertip ([excludeLive]) are dropped so the hand does
/// not cast votes. After similarity ICP, an optional affine refine absorbs
/// mild non-uniform scale without falling back to a full projective model.
class CellConstellationAligner {
  /// Minimum cells that must agree before the transform is trusted.
  static const int minInliers = 10;

  /// How far outside [excludeLive] to also drop centres (hand pad > tip box).
  static const double excludeExpand = 1.5;

  static Future<CellAlignment?> align({
    required List<Offset> liveCenters,
    required List<Offset> referenceCenters,
    required double cellWidth,
    Rect? excludeLive,
  }) async {
    final args = _args(liveCenters, referenceCenters, cellWidth, excludeLive);
    if (args == null) return null;
    return _wrap(await compute(_alignIsolate, args));
  }

  /// Same maths on the calling thread — for tests and for callers that are
  /// already off the UI thread.
  static CellAlignment? alignSync({
    required List<Offset> liveCenters,
    required List<Offset> referenceCenters,
    required double cellWidth,
    Rect? excludeLive,
  }) {
    final args = _args(liveCenters, referenceCenters, cellWidth, excludeLive);
    if (args == null) return null;
    return _wrap(_alignIsolate(args));
  }

  /// Expanded fingertip rect used to mask live centres under the hand.
  static Rect? expandedExclude(Rect? box, {double expand = excludeExpand}) {
    if (box == null || box.isEmpty) return null;
    final cx = box.center.dx;
    final cy = box.center.dy;
    final hw = box.width * expand / 2;
    final hh = box.height * expand / 2;
    return Rect.fromCenter(center: Offset(cx, cy), width: hw * 2, height: hh * 2);
  }

  static Map<String, dynamic>? _args(
    List<Offset> live,
    List<Offset> ref,
    double cellWidth,
    Rect? excludeLive,
  ) {
    final masked = _maskLive(live, expandedExclude(excludeLive));
    if (masked.length < minInliers || ref.length < minInliers) return null;
    return <String, dynamic>{
      'live': _flatten(masked),
      'ref': _flatten(ref),
      'cell': cellWidth,
    };
  }

  static List<Offset> _maskLive(List<Offset> live, Rect? exclude) {
    if (exclude == null) return live;
    return live.where((p) => !exclude.contains(p)).toList();
  }

  static CellAlignment? _wrap(Map<String, dynamic>? raw) {
    if (raw == null) return null;
    return CellAlignment(
      homography: Homography((raw['h'] as List).cast<double>()),
      inliers: raw['inliers'] as int,
    );
  }

  static List<double> _flatten(List<Offset> pts) {
    final out = List<double>.filled(pts.length * 2, 0);
    for (var i = 0; i < pts.length; i++) {
      out[i * 2] = pts[i].dx;
      out[i * 2 + 1] = pts[i].dy;
    }
    return out;
  }
}

/// Keep the vote loop O(n^2) on a bounded n — a full page is ~200 cells and
/// every extra cell multiplies the pair count.
const int _maxVotePoints = 120;
const List<double> _scales = [0.85, 0.925, 1.0, 1.08, 1.18];
const List<double> _rotationsDeg = [-6, -3, 0, 3, 6];

Map<String, dynamic>? _alignIsolate(Map<String, dynamic> args) {
  final live = _unflatten((args['live'] as List).cast<double>());
  final ref = _unflatten((args['ref'] as List).cast<double>());
  final cellWidth = (args['cell'] as num).toDouble();
  if (cellWidth <= 0) return null;

  final liveVote = _sample(live, _maxVotePoints);
  final refVote = _sample(ref, _maxVotePoints);
  final binSize = max(2.0, cellWidth * 0.5);

  // Best translation bin per scale/rotation. Text is close to periodic, so a
  // transform that is off by one cell also scores well; keeping several
  // candidates and judging them after refinement settles which is real.
  final candidates = <_Candidate>[];
  for (final s in _scales) {
    for (final deg in _rotationsDeg) {
      final rad = deg * pi / 180.0;
      final cosA = s * cos(rad);
      final sinA = s * sin(rad);

      final votes = <int, int>{};
      var topKey = 0;
      var topVotes = 0;
      for (final p in liveVote) {
        final px = cosA * p.dx - sinA * p.dy;
        final py = sinA * p.dx + cosA * p.dy;
        for (final q in refVote) {
          final bx = ((q.dx - px) / binSize).floor();
          final by = ((q.dy - py) / binSize).floor();
          final key = (bx + 4096) * 100000 + (by + 4096);
          final next = (votes[key] ?? 0) + 1;
          votes[key] = next;
          if (next > topVotes) {
            topVotes = next;
            topKey = key;
          }
        }
      }
      if (topVotes < CellConstellationAligner.minInliers) continue;
      final bx = topKey ~/ 100000 - 4096;
      final by = topKey % 100000 - 4096;
      candidates.add(_Candidate(
        votes: topVotes,
        h: _similarity(s, rad, (bx + 0.5) * binSize, (by + 0.5) * binSize),
      ));
    }
  }
  if (candidates.isEmpty) return null;

  candidates.sort((a, b) => b.votes.compareTo(a.votes));
  final tol = cellWidth * 0.6;
  final grid = _PointGrid(ref, max(tol, binSize));

  List<double>? best;
  var bestInliers = 0;
  var bestResidual = double.infinity;
  for (final candidate in candidates.take(6)) {
    final fit = _refine(candidate.h, live, grid, tol, cellWidth);
    if (fit == null) continue;
    if (fit.inliers > bestInliers ||
        (fit.inliers == bestInliers && fit.residual < bestResidual)) {
      best = fit.h;
      bestInliers = fit.inliers;
      bestResidual = fit.residual;
    }
  }

  if (best == null || bestInliers < CellConstellationAligner.minInliers) {
    return null;
  }

  // A fit that zooms or flips the page is a wrong answer dressed up as a
  // confident one; let the caller fall back instead.
  final fittedScale = sqrt(best[0] * best[0] + best[3] * best[3]);
  if (fittedScale < 0.6 || fittedScale > 1.7) return null;

  return <String, dynamic>{'h': best, 'inliers': bestInliers};
}

class _Candidate {
  final int votes;
  final List<double> h;

  const _Candidate({required this.votes, required this.h});
}

class _Fit {
  final List<double> h;
  final int inliers;
  final double residual;

  const _Fit({required this.h, required this.inliers, required this.residual});
}

/// The winning bin is only accurate to half a cell. Re-derive the transform
/// from every cell it brings into agreement, so the mapping is set by the
/// whole page rather than by the coarse vote grid. If similarity residual
/// stays high, try an affine refine on the same inliers.
_Fit? _refine(
  List<double> seed,
  List<Offset> live,
  _PointGrid grid,
  double tol,
  double cellWidth,
) {
  var h = seed;
  _Fit? fit;
  List<Offset>? lastSrc;
  List<Offset>? lastDst;
  for (var pass = 0; pass < 3; pass++) {
    final srcIn = <Offset>[];
    final dstIn = <Offset>[];
    var residual = 0.0;
    for (final p in live) {
      final mapped = Homography.transformPoint(h, p);
      final nearest = grid.nearest(mapped, tol);
      if (nearest != null) {
        srcIn.add(p);
        dstIn.add(nearest);
        residual += (nearest - mapped).distance;
      }
    }
    if (srcIn.length < CellConstellationAligner.minInliers) break;
    fit = _Fit(
      h: h,
      inliers: srcIn.length,
      residual: residual / srcIn.length,
    );
    lastSrc = srcIn;
    lastDst = dstIn;
    final next = _fitSimilarity(srcIn, dstIn);
    if (next == null) break;
    h = next;
  }
  if (fit == null || lastSrc == null || lastDst == null) return fit;

  // Mild perspective / non-uniform scale: affine on the same inliers when
  // similarity residual is still a noticeable fraction of a cell.
  if (fit.residual > cellWidth * 0.25 &&
      lastSrc.length >= CellConstellationAligner.minInliers) {
    var affine = fitAffineLeastSquares(lastSrc, lastDst);
    if (affine != null && _affineSane(affine)) {
      // Re-associate with the affine seed so stretched axes can reclaim
      // points that similarity left just outside the tolerance.
      final srcIn = <Offset>[];
      final dstIn = <Offset>[];
      var residual = 0.0;
      for (final p in live) {
        final mapped = Homography.transformPoint(affine, p);
        final nearest = grid.nearest(mapped, tol);
        if (nearest != null) {
          srcIn.add(p);
          dstIn.add(nearest);
          residual += (nearest - mapped).distance;
        }
      }
      if (srcIn.length >= CellConstellationAligner.minInliers) {
        final refined = fitAffineLeastSquares(srcIn, dstIn);
        if (refined != null && _affineSane(refined)) {
          affine = refined;
          residual = 0.0;
          var inliers = 0;
          for (var i = 0; i < srcIn.length; i++) {
            final mapped = Homography.transformPoint(affine, srcIn[i]);
            final d = (mapped - dstIn[i]).distance;
            if (d <= tol) {
              inliers++;
              residual += d;
            }
          }
          if (inliers >= CellConstellationAligner.minInliers) {
            final mean = residual / inliers;
            if (mean < fit.residual) {
              return _Fit(h: affine, inliers: inliers, residual: mean);
            }
          }
        } else {
          final mean = residual / srcIn.length;
          if (mean < fit.residual) {
            return _Fit(h: affine, inliers: srcIn.length, residual: mean);
          }
        }
      }
    }
  }
  return fit;
}

bool _affineSane(List<double> h) {
  // 2x2 linear part determinant must stay positive and away from zero.
  final det = h[0] * h[4] - h[1] * h[3];
  if (det < 0.25 || det > 4.0) return false;
  final sx = sqrt(h[0] * h[0] + h[3] * h[3]);
  final sy = sqrt(h[1] * h[1] + h[4] * h[4]);
  if (sx < 0.6 || sx > 1.7 || sy < 0.6 || sy > 1.7) return false;
  return true;
}

List<Offset> _unflatten(List<double> flat) {
  final out = <Offset>[];
  for (var i = 0; i + 1 < flat.length; i += 2) {
    out.add(Offset(flat[i], flat[i + 1]));
  }
  return out;
}

List<Offset> _sample(List<Offset> pts, int limit) {
  if (pts.length <= limit) return pts;
  final step = pts.length / limit;
  return List<Offset>.generate(limit, (i) => pts[(i * step).floor()]);
}

List<double> _similarity(double scale, double rad, double tx, double ty) {
  final cosA = scale * cos(rad);
  final sinA = scale * sin(rad);
  return [cosA, -sinA, tx, sinA, cosA, ty, 0, 0, 1];
}

/// Least-squares similarity (rotation + uniform scale + translation) fit,
/// solved as a complex linear regression q = a*p + b.
List<double>? _fitSimilarity(List<Offset> src, List<Offset> dst) {
  final n = src.length;
  if (n < 2) return null;
  var pmx = 0.0, pmy = 0.0, qmx = 0.0, qmy = 0.0;
  for (var i = 0; i < n; i++) {
    pmx += src[i].dx;
    pmy += src[i].dy;
    qmx += dst[i].dx;
    qmy += dst[i].dy;
  }
  pmx /= n;
  pmy /= n;
  qmx /= n;
  qmy /= n;

  var num1 = 0.0, num2 = 0.0, den = 0.0;
  for (var i = 0; i < n; i++) {
    final px = src[i].dx - pmx;
    final py = src[i].dy - pmy;
    final qx = dst[i].dx - qmx;
    final qy = dst[i].dy - qmy;
    num1 += px * qx + py * qy;
    num2 += px * qy - py * qx;
    den += px * px + py * py;
  }
  if (den < 1e-9) return null;
  final ar = num1 / den;
  final ai = num2 / den;
  final bx = qmx - (ar * pmx - ai * pmy);
  final by = qmy - (ai * pmx + ar * pmy);
  return [ar, -ai, bx, ai, ar, by, 0, 0, 1];
}

/// Uniform bucket grid so nearest-neighbour lookups stay O(1) per query.
class _PointGrid {
  final double cell;
  final Map<int, List<Offset>> _buckets = {};

  _PointGrid(List<Offset> pts, this.cell) {
    for (final p in pts) {
      _buckets.putIfAbsent(_key(p.dx, p.dy), () => []).add(p);
    }
  }

  int _key(double x, double y) =>
      ((x / cell).floor() + 4096) * 100000 + ((y / cell).floor() + 4096);

  Offset? nearest(Offset p, double maxDistance) {
    Offset? best;
    var bestD = maxDistance;
    final gx = (p.dx / cell).floor();
    final gy = (p.dy / cell).floor();
    for (var dx = -1; dx <= 1; dx++) {
      for (var dy = -1; dy <= 1; dy++) {
        final bucket = _buckets[((gx + dx) + 4096) * 100000 + ((gy + dy) + 4096)];
        if (bucket == null) continue;
        for (final q in bucket) {
          final d = (q - p).distance;
          if (d <= bestD) {
            bestD = d;
            best = q;
          }
        }
      }
    }
    return best;
  }
}
