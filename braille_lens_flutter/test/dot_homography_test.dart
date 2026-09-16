import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';
import 'package:braille_lens_flutter/utils/dot_sequence.dart';
import 'package:braille_lens_flutter/utils/homography.dart';

void main() {
  test('compactDotSequence from label prose', () {
    expect(compactDotSequence('dots 1, 2, 5'), '125');
    expect(compactDotSequence('dot 3'), '3');
    expect(compactDotSequence('34'), '34');
    expect(compactDotSequence('dots 3, 4'), '34');
    expect(compactDotSequence('no dots'), '—');
    expect(compactDotSequence(''), '—');
    expect(compactDotSequence('dot 8'), '—');
  });

  test('identity homography leaves a point unchanged', () {
    expect(Homography.identity.transform(const Offset(12, 34)), const Offset(12, 34));
  });

  test('translation homography from four corners', () {
    final src = [
      const Offset(0, 0),
      const Offset(10, 0),
      const Offset(0, 10),
      const Offset(10, 10),
    ];
    final dst = [
      const Offset(5, 8),
      const Offset(15, 8),
      const Offset(5, 18),
      const Offset(15, 18),
    ];
    final h = homographyFrom4(src, dst);
    expect(h, isNotNull);
    final p = Homography.transformPoint(h!, const Offset(3, 4));
    expect(p.dx, closeTo(8, 1e-6));
    expect(p.dy, closeTo(12, 1e-6));
  });

  test('RANSAC homography recovers a known translation', () {
    final src = <Offset>[];
    final dst = <Offset>[];
    for (var i = 0; i < 20; i++) {
      final p = Offset(i * 3.0, i * 2.0);
      src.add(p);
      dst.add(Offset(p.dx + 7, p.dy - 4));
    }
    final h = ransacHomography(src, dst, minInliers: 12, seed: 1);
    expect(h, isNotNull);
    final mapped = h!.transform(const Offset(10, 10));
    expect(mapped.dx, closeTo(17, 0.5));
    expect(mapped.dy, closeTo(6, 0.5));
  });
}
