import 'package:flutter_test/flutter_test.dart';
import 'package:braille_lens_flutter/models/braille_cell.dart';
import 'package:braille_lens_flutter/services/cell_hit_test.dart';
import 'package:braille_lens_flutter/services/coordinate_mapper.dart';
import 'package:braille_lens_flutter/services/covered_cell_service.dart';

void main() {
  test('hit_test finds cell under tip', () {
    final map = CellMap(
      imageWidth: 720,
      imageHeight: 1280,
      cells: [
        const BrailleCell(id: 0, x0: 0, y0: 0, x1: 10, y1: 10, char: ' ', code: 0),
        const BrailleCell(id: 1, x0: 100, y0: 200, x1: 140, y1: 240, char: 'ක', code: 19, pattern: '125'),
      ],
    );
    final hit = CellHitTest.hitTest(const Offset(120, 220), map);
    expect(hit?.char, 'ක');
    expect(hit?.id, 1);
  });

  test('coordinate mapper scales finger tip to prescan frame', () {
    final tip = CoordinateMapper.mapFingerTipToPrescan(
      tipInFingerImage: const Offset(360, 640),
      prescanWidth: 720,
      prescanHeight: 1280,
      fingerImageWidth: 720,
      fingerImageHeight: 1280,
    );
    expect(tip.dx, 360);
    expect(tip.dy, 640);
  });

  test('covered cell service end-to-end', () {
    final map = CellMap(
      imageWidth: 720,
      imageHeight: 1280,
      cells: [
        const BrailleCell(id: 5, x0: 300, y0: 500, x1: 340, y1: 540, char: 'ත', code: 6, pattern: '23'),
      ],
    );
    final result = CoveredCellService().resolve(
      tipInFingerImage: const Offset(320, 530),
      cellMap: map,
      fingerImageWidth: 720,
      fingerImageHeight: 1280,
    );
    expect(result.hasHit, isTrue);
    expect(result.headline, 'ත');
  });
}
