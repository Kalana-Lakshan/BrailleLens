import 'package:flutter_test/flutter_test.dart';

import 'package:braille_lens_flutter/models/braille_cell.dart';
import 'package:braille_lens_flutter/services/cell_hit_test.dart';
import 'package:braille_lens_flutter/services/coordinate_mapper.dart';

/// A 3x2 grid of cells on a 300x200 reference page, cell (col,row) centred
/// at (50 + col*80, 50 + row*100), each cell 40x40px, 40px gaps.
/// code = 10*row + col + 1, so results are unambiguous.
CellMap _gridPage() {
  final cells = <BrailleCell>[];
  var id = 0;
  for (var row = 0; row < 2; row++) {
    for (var col = 0; col < 3; col++) {
      final cx = 50.0 + col * 80.0;
      final cy = 50.0 + row * 100.0;
      cells.add(BrailleCell(
        id: id,
        x0: cx - 20,
        y0: cy - 20,
        x1: cx + 20,
        y1: cy + 20,
        char: String.fromCharCode(65 + id), // A, B, C, ...
        code: 10 * row + col + 1, // never 0 -- real prescan output never is either
      ));
      id++;
    }
  }
  return CellMap(cells: cells, imageWidth: 300, imageHeight: 200);
}

void main() {
  group('CoordinateMapper.mapFingerTipToPrescan', () {
    test('identity when both captures are the same resolution', () {
      final mapped = CoordinateMapper.mapFingerTipToPrescan(
        tipInFingerImage: const Offset(123, 45),
        prescanWidth: 300,
        prescanHeight: 200,
        fingerImageWidth: 300,
        fingerImageHeight: 200,
      );
      expect(mapped.dx, closeTo(123, 1e-9));
      expect(mapped.dy, closeTo(45, 1e-9));
    });

    test('scales proportionally when the finger capture is a different resolution', () {
      // Finger photo taken at half the prescan's resolution -- same physical
      // page, same framing, just a smaller capture. A point at the finger
      // image's exact centre must map to the prescan's exact centre.
      final mapped = CoordinateMapper.mapFingerTipToPrescan(
        tipInFingerImage: const Offset(75, 50), // centre of 150x100
        prescanWidth: 300,
        prescanHeight: 200,
        fingerImageWidth: 150,
        fingerImageHeight: 100,
      );
      expect(mapped.dx, closeTo(150, 1e-9)); // centre of 300
      expect(mapped.dy, closeTo(100, 1e-9)); // centre of 200
    });

    test('falls back to the raw point when a dimension is invalid (no divide-by-zero)', () {
      final mapped = CoordinateMapper.mapFingerTipToPrescan(
        tipInFingerImage: const Offset(10, 20),
        prescanWidth: 0,
        prescanHeight: 200,
        fingerImageWidth: 300,
        fingerImageHeight: 200,
      );
      expect(mapped, const Offset(10, 20));
    });
  });

  group('CellHitTest.hitTest end-to-end through CoordinateMapper', () {
    test('finger dead-centre on a cell resolves to that exact cell/character/code', () {
      final page = _gridPage();
      // Same-resolution finger capture; tip exactly on cell (col=1,row=0) -> 'B', code 2.
      final tipInFinger = const Offset(130, 50); // cx for col=1
      final tipInPrescan = CoordinateMapper.mapFingerTipToPrescan(
        tipInFingerImage: tipInFinger,
        prescanWidth: page.imageWidth,
        prescanHeight: page.imageHeight,
        fingerImageWidth: 300,
        fingerImageHeight: 200,
      );
      final hit = CellHitTest.hitTest(tipInPrescan, page);
      expect(hit, isNotNull);
      expect(hit!.char, 'B');
      expect(hit.code, 2);
    });

    test('resolves the correct cell even when the finger photo is a different resolution',
        () {
      final page = _gridPage();
      // Finger photo at 2x prescan resolution (600x400). Physical point over
      // cell (col=2,row=1) ('F', code = 10*1+2+1 = 13) sits at prescan
      // (210,150) -> finger-image (420,300).
      final tipInPrescan = CoordinateMapper.mapFingerTipToPrescan(
        tipInFingerImage: const Offset(420, 300),
        prescanWidth: page.imageWidth,
        prescanHeight: page.imageHeight,
        fingerImageWidth: 600,
        fingerImageHeight: 400,
      );
      final hit = CellHitTest.hitTest(tipInPrescan, page);
      expect(hit, isNotNull);
      expect(hit!.char, 'F');
      expect(hit.code, 13);
    });

    test('finger in the gap between two cells (outside margin) resolves to nothing', () {
      final page = _gridPage();
      // Midway between col=0 (cx=50) and col=1 (cx=130) on row 0: x=90, well
      // outside both cells' 40px box + 12% margin (cells end at x=70/x=110).
      final hit = CellHitTest.hitTest(const Offset(90, 50), page);
      expect(hit, isNull);
    });

    test('finger just inside the margin resolves to the nearer cell', () {
      final page = _gridPage();
      // Cell col=0 spans x=[30,70]; margin = 40*0.12 = 4.8, so x=74 is just
      // inside col=0's expanded box but outside col=1's (x=[110-4.8,...]).
      final hit = CellHitTest.hitTest(const Offset(74, 50), page);
      expect(hit, isNotNull);
      expect(hit!.char, 'A');
    });

    test('empty page returns null, not a crash', () {
      final empty = CellMap(cells: const [], imageWidth: 300, imageHeight: 200);
      expect(CellHitTest.hitTest(const Offset(50, 50), empty), isNull);
    });

    test('skipEmpty prefers a real character over an overlapping space cell', () {
      // Defensive case: even if a code-0 cell ever reaches this layer (it
      // shouldn't -- PrescanOnnxService already drops code-0 before building
      // the CellMap), hitTest must still prefer the real letter.
      final space = BrailleCell(id: 0, x0: 0, y0: 0, x1: 40, y1: 40, code: 0, char: ' ');
      final letter = BrailleCell(id: 1, x0: 5, y0: 5, x1: 35, y1: 35, code: 5, char: 'X');
      final page = CellMap(cells: [space, letter], imageWidth: 40, imageHeight: 40);
      final hit = CellHitTest.hitTest(const Offset(20, 20), page);
      expect(hit!.code, 5);
      expect(hit.char, 'X');
    });
  });
}
