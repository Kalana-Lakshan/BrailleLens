import 'dart:math';
import 'dart:ui';

/// 3x3 homography (row-major). Maps source pixels onto destination pixels.
class Homography {
  final List<double> m;

  const Homography(this.m);

  static const Homography identity = Homography([
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
  ]);

  Offset transform(Offset p) => transformPoint(m, p);

  static Offset transformPoint(List<double> h, Offset p) {
    if (h.length != 9) return p;
    final x = p.dx;
    final y = p.dy;
    final w = h[6] * x + h[7] * y + h[8];
    if (w.abs() < 1e-12) return p;
    return Offset(
      (h[0] * x + h[1] * y + h[2]) / w,
      (h[3] * x + h[4] * y + h[5]) / w,
    );
  }
}

/// DLT from four source→destination pairs. Returns null if degenerate.
List<double>? homographyFrom4(List<Offset> src, List<Offset> dst) {
  if (src.length != 4 || dst.length != 4) return null;
  final a = List.generate(8, (_) => List<double>.filled(8, 0));
  final b = List<double>.filled(8, 0);
  for (var i = 0; i < 4; i++) {
    final x = src[i].dx;
    final y = src[i].dy;
    final u = dst[i].dx;
    final v = dst[i].dy;
    final r0 = i * 2;
    final r1 = r0 + 1;
    a[r0][0] = x;
    a[r0][1] = y;
    a[r0][2] = 1;
    a[r0][6] = -u * x;
    a[r0][7] = -u * y;
    b[r0] = u;
    a[r1][3] = x;
    a[r1][4] = y;
    a[r1][5] = 1;
    a[r1][6] = -v * x;
    a[r1][7] = -v * y;
    b[r1] = v;
  }
  final h8 = _solve8(a, b);
  if (h8 == null) return null;
  return [...h8, 1.0];
}

/// RANSAC homography. [src] and [dst] are paired correspondences.
Homography? ransacHomography(
  List<Offset> src,
  List<Offset> dst, {
  int minInliers = 12,
  double reprojThresh = 5.0,
  int iterations = 250,
  int? seed,
}) {
  if (src.length != dst.length || src.length < 4) return null;
  final rng = Random(seed ?? 0);
  var bestCount = 0;
  List<double>? bestH;
  final n = src.length;
  final idx = List<int>.generate(n, (i) => i);

  for (var t = 0; t < iterations; t++) {
    idx.shuffle(rng);
    final s = [src[idx[0]], src[idx[1]], src[idx[2]], src[idx[3]]];
    final d = [dst[idx[0]], dst[idx[1]], dst[idx[2]], dst[idx[3]]];
    final h = homographyFrom4(s, d);
    if (h == null) continue;
    var count = 0;
    for (var i = 0; i < n; i++) {
      final mapped = Homography.transformPoint(h, src[i]);
      if ((mapped - dst[i]).distance <= reprojThresh) count++;
    }
    if (count > bestCount) {
      bestCount = count;
      bestH = h;
    }
  }
  if (bestH == null || bestCount < minInliers) return null;
  return Homography(bestH);
}

List<double>? _solve8(List<List<double>> a, List<double> b) {
  const n = 8;
  final m = List.generate(n, (i) => [...a[i], b[i]]);
  for (var col = 0; col < n; col++) {
    var pivot = col;
    var best = m[col][col].abs();
    for (var r = col + 1; r < n; r++) {
      final v = m[r][col].abs();
      if (v > best) {
        best = v;
        pivot = r;
      }
    }
    if (best < 1e-12) return null;
    if (pivot != col) {
      final tmp = m[col];
      m[col] = m[pivot];
      m[pivot] = tmp;
    }
    final div = m[col][col];
    for (var c = col; c <= n; c++) {
      m[col][c] /= div;
    }
    for (var r = 0; r < n; r++) {
      if (r == col) continue;
      final f = m[r][col];
      if (f == 0) continue;
      for (var c = col; c <= n; c++) {
        m[r][c] -= f * m[col][c];
      }
    }
  }
  return List<double>.generate(n, (i) => m[i][n]);
}
