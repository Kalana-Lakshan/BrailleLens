import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:braille_lens_flutter/models/braille_cell.dart';
import 'package:braille_lens_flutter/services/cell_hit_test.dart';
import 'package:braille_lens_flutter/services/coordinate_mapper.dart';
import 'package:braille_lens_flutter/services/covered_cell_service.dart';
import 'package:braille_lens_flutter/theme/app_theme.dart';

/// One space cell and one ක cell on a 720x1280 prescan frame.
CellMap _sampleMap() {
  return const CellMap(
    imageWidth: 720,
    imageHeight: 1280,
    cells: [
      BrailleCell(id: 0, x0: 0, y0: 0, x1: 10, y1: 10, char: ' ', code: 0),
      BrailleCell(
        id: 1,
        x0: 100,
        y0: 200,
        x1: 140,
        y1: 240,
        char: 'ක',
        code: 19,
        pattern: '125',
      ),
    ],
  );
}

void main() {
  // Tip at (120,220) sits inside the ක box; hit id must be 1.
  test('FT-FL-01 hit_test finds cell under tip', () {
    final hit = CellHitTest.hitTest(const Offset(120, 220), _sampleMap());
    expect(hit?.char, 'ක');
    expect(hit?.id, 1);
  });

  // A far tip, and a tip on an empty CellMap, must not invent a character.
  test('FT-FL-02 hit_test miss and empty map', () {
    expect(CellHitTest.hitTest(const Offset(500, 500), _sampleMap()), isNull);
    expect(
      CellHitTest.hitTest(
        const Offset(1, 1),
        const CellMap(cells: [], imageWidth: 10, imageHeight: 10),
      ),
      isNull,
    );
  });

  // When the tip sits on both a synthetic space and a letter, prefer ත.
  test('FT-FL-03 skipEmpty prefers letter over synthetic space', () {
    final map = const CellMap(
      imageWidth: 100,
      imageHeight: 100,
      cells: [
        BrailleCell(id: 0, x0: 0, y0: 0, x1: 40, y1: 40, char: ' ', code: 0),
        BrailleCell(id: 1, x0: 10, y0: 10, x1: 50, y1: 50, char: 'ත', code: 6),
      ],
    );
    final hit = CellHitTest.hitTest(const Offset(20, 20), map, skipEmpty: true);
    expect(hit?.char, 'ත');
  });

  // Overlapping A and B still return one of those letters (nearest centre).
  test('FT-FL-04 overlap chooses nearest centre', () {
    final map = const CellMap(
      imageWidth: 200,
      imageHeight: 200,
      cells: [
        BrailleCell(id: 0, x0: 0, y0: 0, x1: 40, y1: 40, char: 'A', code: 1),
        BrailleCell(id: 1, x0: 20, y0: 20, x1: 60, y1: 60, char: 'B', code: 2),
      ],
    );
    final hit = CellHitTest.hitTest(const Offset(30, 30), map);
    expect(hit, isNotNull);
    expect(['A', 'B'], contains(hit!.char));
  });

  // Half-size finger photo tip (360,640) scales 2x into the 720x1280 prescan.
  test('FT-FL-05 coordinate mapper scales to prescan frame', () {
    final tip = CoordinateMapper.mapFingerTipToPrescan(
      tipInFingerImage: const Offset(360, 640),
      prescanWidth: 720,
      prescanHeight: 1280,
      fingerImageWidth: 360,
      fingerImageHeight: 640,
    );
    expect(tip.dx, 720);
    expect(tip.dy, 1280);
  });

  // Prescan width 0 must not divide by zero; return the original tip.
  test('FT-FL-06 invalid mapper sizes return original tip', () {
    final tip = CoordinateMapper.mapFingerTipToPrescan(
      tipInFingerImage: const Offset(10, 20),
      prescanWidth: 0,
      prescanHeight: 1280,
      fingerImageWidth: 720,
      fingerImageHeight: 1280,
    );
    expect(tip, const Offset(10, 20));
  });

  // Full lookup: finger-photo tip -> prescan -> headline ත.
  test('FT-FL-07 covered cell service end-to-end', () {
    final map = const CellMap(
      imageWidth: 720,
      imageHeight: 1280,
      cells: [
        BrailleCell(
          id: 5,
          x0: 300,
          y0: 500,
          x1: 340,
          y1: 540,
          char: 'ත',
          code: 6,
          pattern: '23',
        ),
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

  // JSON round-trip must keep cell count, ක on id 1, and image width.
  test('DI-FL-01 CellMap JSON round-trip does not drop cells', () {
    final original = _sampleMap();
    final copy = CellMap.fromJson(original.toJson());
    expect(copy.length, original.length);
    expect(copy.byId(1)?.char, 'ක');
    expect(copy.imageWidth, 720);
  });

  // Theme yellow on black must meet WCAG 7:1 contrast (SRS UI).
  test('UI-FL-01 yellow on black meets WCAG 7:1', () {
    final l1 = AppTheme.backgroundBlack.computeLuminance();
    final l2 = AppTheme.primaryYellow.computeLuminance();
    final lighter = math.max(l1, l2);
    final darker = math.min(l1, l2);
    final contrast = (lighter + 0.05) / (darker + 0.05);
    expect(contrast, greaterThanOrEqualTo(7.0));
  });
}
