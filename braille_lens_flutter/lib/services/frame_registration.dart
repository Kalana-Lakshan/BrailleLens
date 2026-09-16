import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

import '../utils/homography.dart';
import '../utils/image_decode.dart';

class FrameRegistrationResult {
  final Homography? homography;
  final int inliers;
  final String alignMode;

  const FrameRegistrationResult({
    required this.homography,
    required this.inliers,
    required this.alignMode,
  });

  bool get usedHomography => homography != null && alignMode == 'homography';
}

/// ORB-style whole-page alignment: gradient keypoints, patch descriptors,
/// Lowe-ratio match, RANSAC homography mapping live pixels → reference pixels.
class FrameRegistration {
  static const int _maxSide = 640;

  static Future<FrameRegistrationResult> estimate({
    required Uint8List referenceJpeg,
    required Uint8List liveJpeg,
    Rect? liveMask,
  }) async {
    final raw = await compute(_estimateIsolate, <String, dynamic>{
      'ref': referenceJpeg,
      'live': liveJpeg,
      'mask': liveMask == null
          ? null
          : <double>[liveMask.left, liveMask.top, liveMask.right, liveMask.bottom],
    });
    final h = raw['h'] as List<dynamic>?;
    return FrameRegistrationResult(
      homography: h == null ? null : Homography(h.map((e) => (e as num).toDouble()).toList()),
      inliers: raw['inliers'] as int,
      alignMode: raw['mode'] as String,
    );
  }
}

const _minMatches = 12;

Map<String, dynamic> _failScale([int inliers = 0]) =>
    <String, dynamic>{'h': null, 'inliers': inliers, 'mode': 'scale'};

Map<String, dynamic> _estimateIsolate(Map<String, dynamic> args) {
  final refJpeg = args['ref'] as Uint8List;
  final liveJpeg = args['live'] as Uint8List;
  final maskList = args['mask'] as List<dynamic>?;

  final refImg = decodeUpright(refJpeg);
  final liveImg = decodeUpright(liveJpeg);
  if (refImg == null || liveImg == null) return _failScale();

  final refG = _toGrayDown(refImg);
  final liveG = _toGrayDown(liveImg);
  Rect? mask;
  if (maskList != null && maskList.length == 4) {
    final sx = liveG.w / liveImg.width;
    final sy = liveG.h / liveImg.height;
    var l = (maskList[0] as num).toDouble() * sx;
    var t = (maskList[1] as num).toDouble() * sy;
    var r = (maskList[2] as num).toDouble() * sx;
    var b = (maskList[3] as num).toDouble() * sy;
    final padX = (r - l) * 0.2;
    final padY = (b - t) * 0.2;
    mask = Rect.fromLTRB(l - padX, t - padY, r + padX, b + padY);
  }

  final refKp = _keypoints(refG, null);
  final liveKp = _keypoints(liveG, mask);
  if (refKp.length < _minMatches || liveKp.length < _minMatches) return _failScale();

  final pairs = _match(liveKp, refKp);
  if (pairs.length < _minMatches) return _failScale();

  final src = <Offset>[];
  final dst = <Offset>[];
  final liveScaleX = liveImg.width / liveG.w;
  final liveScaleY = liveImg.height / liveG.h;
  final refScaleX = refImg.width / refG.w;
  final refScaleY = refImg.height / refG.h;
  for (final p in pairs) {
    src.add(Offset(p.$1.dx * liveScaleX, p.$1.dy * liveScaleY));
    dst.add(Offset(p.$2.dx * refScaleX, p.$2.dy * refScaleY));
  }

  final h = ransacHomography(src, dst, minInliers: _minMatches, seed: 7);
  if (h == null) return _failScale();

  var inliers = 0;
  for (var i = 0; i < src.length; i++) {
    if ((h.transform(src[i]) - dst[i]).distance <= 5.0) inliers++;
  }
  if (inliers < _minMatches) return _failScale(inliers);
  return <String, dynamic>{'h': h.m, 'inliers': inliers, 'mode': 'homography'};
}

class _Gray {
  final int w;
  final int h;
  final Float32List lum;
  _Gray(this.w, this.h, this.lum);
}

class _Kp {
  final Offset pt;
  final List<double> desc;
  _Kp(this.pt, this.desc);
}

_Gray _toGrayDown(img.Image src) {
  final long = max(src.width, src.height);
  final scale = long > FrameRegistration._maxSide ? FrameRegistration._maxSide / long : 1.0;
  final w = max(8, (src.width * scale).round());
  final h = max(8, (src.height * scale).round());
  final small = img.copyResize(src, width: w, height: h);
  final gray = img.grayscale(small);
  final lum = Float32List(w * h);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      lum[y * w + x] = gray.getPixel(x, y).r.toDouble();
    }
  }
  return _Gray(w, h, lum);
}

List<_Kp> _keypoints(_Gray g, Rect? mask) {
  final w = g.w;
  final h = g.h;
  final score = Float32List(w * h);
  for (var y = 1; y < h - 1; y++) {
    for (var x = 1; x < w - 1; x++) {
      final gx = g.lum[y * w + x + 1] - g.lum[y * w + x - 1];
      final gy = g.lum[(y + 1) * w + x] - g.lum[(y - 1) * w + x];
      score[y * w + x] = gx * gx + gy * gy;
    }
  }

  const nms = 6;
  final candidates = <(double, int, int)>[];
  for (var y = 8; y < h - 8; y++) {
    for (var x = 8; x < w - 8; x++) {
      if (mask != null && mask.contains(Offset(x.toDouble(), y.toDouble()))) {
        continue;
      }
      final s = score[y * w + x];
      if (s < 80) continue;
      var peak = true;
      for (var dy = -1; dy <= 1 && peak; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
          if (dx == 0 && dy == 0) continue;
          if (score[(y + dy) * w + x + dx] > s) peak = false;
        }
      }
      if (peak) candidates.add((s, x, y));
    }
  }
  candidates.sort((a, b) => b.$1.compareTo(a.$1));

  final kept = <(int, int)>[];
  for (final c in candidates) {
    final x = c.$2;
    final y = c.$3;
    var far = true;
    for (final k in kept) {
      final dx = k.$1 - x;
      final dy = k.$2 - y;
      if (dx * dx + dy * dy < nms * nms) {
        far = false;
        break;
      }
    }
    if (far) kept.add((x, y));
    if (kept.length >= 600) break;
  }

  return kept.map((p) => _Kp(Offset(p.$1.toDouble(), p.$2.toDouble()), _desc(g, p.$1, p.$2))).toList();
}

List<double> _desc(_Gray g, int cx, int cy) {
  const r = 4;
  final vals = <double>[];
  var sum = 0.0;
  for (var y = cy - r; y <= cy + r; y++) {
    for (var x = cx - r; x <= cx + r; x++) {
      final v = g.lum[y * g.w + x];
      vals.add(v);
      sum += v;
    }
  }
  final mean = sum / vals.length;
  var varSum = 0.0;
  for (final v in vals) {
    final d = v - mean;
    varSum += d * d;
  }
  final std = sqrt(varSum / vals.length) + 1e-6;
  return vals.map((v) => (v - mean) / std).toList();
}

List<(Offset, Offset)> _match(List<_Kp> live, List<_Kp> ref) {
  const ratio = 0.75;
  final out = <(Offset, Offset)>[];
  for (final a in live) {
    var best = 1e18;
    var second = 1e18;
    _Kp? bestKp;
    for (final b in ref) {
      var d = 0.0;
      for (var i = 0; i < a.desc.length; i++) {
        final e = a.desc[i] - b.desc[i];
        d += e * e;
      }
      if (d < best) {
        second = best;
        best = d;
        bestKp = b;
      } else if (d < second) {
        second = d;
      }
    }
    if (bestKp != null && best < ratio * ratio * second) {
      out.add((a.pt, bestKp.pt));
    }
  }
  return out;
}
