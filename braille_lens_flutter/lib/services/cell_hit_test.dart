import 'dart:ui';

import '../models/braille_cell.dart';

/// Pure geometry — maps a fingertip point to the prescan cell underneath.
/// No ML model runs at this step; labels come from stage-1 prescan only.
class CellHitTest {
  /// Return the cell under [tip] in prescan pixel coordinates.
  ///
  /// [skipEmpty] — prefer real Braille cells over synthetic code-0 gaps.
  static BrailleCell? hitTest(
    Offset tip,
    CellMap cellMap, {
    double marginFrac = 0.12,
    bool skipEmpty = true,
  }) {
    if (cellMap.cells.isEmpty) return null;

    var hits = cellMap.cells.where((c) => _contains(c, tip, marginFrac)).toList();
    if (hits.isEmpty) return null;

    if (skipEmpty) {
      final letters = hits.where((c) => c.code != 0).toList();
      if (letters.isNotEmpty) hits = letters;
    }

    if (hits.length == 1) return hits.first;

    hits.sort((a, b) {
      final da = (a.center - tip).distanceSquared;
      final db = (b.center - tip).distanceSquared;
      return da.compareTo(db);
    });
    return hits.first;
  }

  static bool _contains(BrailleCell cell, Offset tip, double marginFrac) {
    final w = cell.x1 - cell.x0;
    final h = cell.y1 - cell.y0;
    final mx = w * marginFrac;
    final my = h * marginFrac;
    return tip.dx >= cell.x0 - mx &&
        tip.dx <= cell.x1 + mx &&
        tip.dy >= cell.y0 - my &&
        tip.dy <= cell.y1 + my;
  }
}
