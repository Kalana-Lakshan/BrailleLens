import 'dart:math';
import 'dart:typed_data';

import 'package:image/image.dart' as img;

import '../models/braille_cell.dart';

/// Finds Braille dot blobs on a page photo and groups them into cell boxes.
/// Pure Dart — no YOLO. Used with [ClassifierService] for on-device prescan.
class DotCellDetector {
  /// Max width for detection (faster on phone); boxes are scaled back to full res.
  static const int maxDetectWidth = 1280;

  static List<_Dot> findDots(img.Image gray) {
    final w = gray.width;
    final h = gray.height;
    final pixels = gray.getBytes(order: img.ChannelOrder.rgb);
    final lum = List<double>.generate(w * h, (i) {
      final o = i * 3;
      return (pixels[o] + pixels[o + 1] + pixels[o + 2]) / 3.0;
    });

    lum.sort();
    final p25 = lum[(lum.length * 0.25).floor()];
    final p75 = lum[(lum.length * 0.75).floor()];

    final darkMask = _componentsFromMask(
      w,
      h,
      pixels,
      (l) => l < p25 + (p75 - p25) * 0.15,
      true,
    );
    final brightMask = _componentsFromMask(
      w,
      h,
      pixels,
      (l) => l > p75 - (p75 - p25) * 0.15,
      false,
    );

    final area = w * h;
    final minA = area * 0.000008;
    final maxA = area * 0.008;

    var dots = _filterComponents(darkMask, w, h, minA, maxA);
    if (dots.length < 20) {
      dots = _filterComponents(brightMask, w, h, minA, maxA);
    }
    return dots;
  }

  static List<_Dot> _componentsFromMask(
    int w,
    int h,
    Uint8List rgb,
    bool Function(double l) keep,
    bool dark,
  ) {
    final mask = List<bool>.filled(w * h, false);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final o = (y * w + x) * 3;
        final l = (rgb[o] + rgb[o + 1] + rgb[o + 2]) / 3.0;
        mask[y * w + x] = keep(l);
      }
    }
    return _labelComponents(mask, w, h);
  }

  static List<_Dot> _labelComponents(List<bool> mask, int w, int h) {
    final labels = List<int>.filled(w * h, 0);
    var next = 1;
    final sums = <int, List<double>>{};

    void flood(int sx, int sy, int id) {
      final stack = <(int, int)>[(sx, sy)];
      var count = 0;
      var sxSum = 0.0;
      var sySum = 0.0;
      while (stack.isNotEmpty) {
        final (x, y) = stack.removeLast();
        if (x < 0 || y < 0 || x >= w || y >= h) continue;
        final i = y * w + x;
        if (!mask[i] || labels[i] != 0) continue;
        labels[i] = id;
        count++;
        sxSum += x;
        sySum += y;
        stack.add((x + 1, y));
        stack.add((x - 1, y));
        stack.add((x, y + 1));
        stack.add((x, y - 1));
      }
      if (count > 0) {
        sums[id] = [sxSum / count, sySum / count, count.toDouble()];
      }
    }

    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final i = y * w + x;
        if (mask[i] && labels[i] == 0) {
          flood(x, y, next++);
        }
      }
    }

    return sums.entries
        .map((e) => _Dot(e.value[0], e.value[1], sqrt(e.value[2] / pi)))
        .toList();
  }

  static List<_Dot> _filterComponents(
    List<_Dot> raw,
    int w,
    int h,
    double minA,
    double maxA,
  ) {
    return raw.where((d) {
      final a = pi * d.radius * d.radius;
      return a >= minA &&
          a <= maxA &&
          d.x > w * 0.02 &&
          d.x < w * 0.98 &&
          d.y > h * 0.02 &&
          d.y < h * 0.98;
    }).toList();
  }

  /// Group dots into cell bounding boxes (reading order).
  static List<RectL> dotsToCellBoxes(List<_Dot> dots) {
    if (dots.length < 4) return [];

    final sorted = List<_Dot>.from(dots)..sort((a, b) => a.y.compareTo(b.y));
    final radii = dots.map((d) => d.radius).toList()..sort();
    final medR = radii[radii.length ~/ 2];
    final lineGap = medR * 4.5;

    final lines = <List<_Dot>>[];
    for (final d in sorted) {
      if (lines.isEmpty || d.y - lines.last.last.y > lineGap) {
        lines.add([d]);
      } else {
        lines.last.add(d);
      }
    }

    final boxes = <RectL>[];
    final cellGap = medR * 3.2;

    for (final line in lines) {
      line.sort((a, b) => a.x.compareTo(b.x));
      var group = <_Dot>[line.first];
      for (var i = 1; i < line.length; i++) {
        if (line[i].x - group.last.x > cellGap) {
          boxes.add(_boxFromDots(group, medR));
          group = [line[i]];
        } else {
          group.add(line[i]);
        }
      }
      if (group.isNotEmpty) boxes.add(_boxFromDots(group, medR));
    }
    return boxes;
  }

  static RectL _boxFromDots(List<_Dot> g, double medR) {
    var x0 = g.map((d) => d.x).reduce(min);
    var x1 = g.map((d) => d.x).reduce(max);
    var y0 = g.map((d) => d.y).reduce(min);
    var y1 = g.map((d) => d.y).reduce(max);
    final pad = medR * 2.8;
    return RectL(x0 - pad, y0 - pad, x1 + pad, y1 + pad);
  }

  /// Detect cell boxes on a JPEG; coordinates are in original image space.
  static ({List<RectL> boxes, int width, int height}) detectCellBoxes(
    Uint8List jpegBytes,
  ) {
    final decoded = img.decodeImage(jpegBytes);
    if (decoded == null) {
      return (boxes: <RectL>[], width: 0, height: 0);
    }

    var work = decoded;
    var scale = 1.0;
    if (work.width > maxDetectWidth) {
      scale = maxDetectWidth / work.width;
      work = img.copyResize(
        work,
        width: maxDetectWidth,
        height: (work.height * scale).round(),
      );
    }

    final gray = img.grayscale(work);
    final dots = findDots(gray);
    var boxes = dotsToCellBoxes(dots);

    if (scale != 1.0) {
      final inv = 1 / scale;
      boxes = boxes
          .map(
            (b) => RectL(
              b.x0 * inv,
              b.y0 * inv,
              b.x1 * inv,
              b.y1 * inv,
            ),
          )
          .toList();
    }

    return (boxes: boxes, width: decoded.width, height: decoded.height);
  }
}

class _Dot {
  final double x;
  final double y;
  final double radius;
  _Dot(this.x, this.y, this.radius);
}

class RectL {
  final double x0;
  final double y0;
  final double x1;
  final double y1;
  const RectL(this.x0, this.y0, this.x1, this.y1);

  int get width => (x1 - x0).round().clamp(1, 10000);
  int get height => (y1 - y0).round().clamp(1, 10000);
}
