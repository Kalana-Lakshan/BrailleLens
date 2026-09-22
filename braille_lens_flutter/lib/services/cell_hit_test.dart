import 'dart:math' as math;
import 'dart:ui';

import '../models/braille_cell.dart';

/// Pure geometry — maps a fingertip point to the prescan cell underneath.
/// No ML model runs at this step; labels come from stage-1 prescan only.
class CellHitTest {
  /// Return the cell under [tip] in prescan pixel coordinates.
  ///
  /// [skipEmpty] — prefer real Braille cells over synthetic code-0 gaps.
  ///
  /// [nearestWithinCells] — how far outside every box the tip may still land
  /// and be resolved, in multiples of a cell width. A finger pad covers a cell
  /// far more generously than the single contact point the detector reports,
  /// so demanding the point fall inside a box rejects plenty of genuine
  /// touches. Set to 0 to require a strict containment hit.
  ///
  /// [marginPx] — absolute slack in image pixels, applied alongside
  /// [marginFrac]; the larger of the two wins. A cell photographed from arm's
  /// length can be ~20 px wide, where 12% is barely two pixels, while a finger
  /// pad covers far more than that.
  ///
  /// Defaults to 0 so that `nearestWithinCells: 0` still means strict
  /// containment for callers that ask for it. [CoveredCellService] passes the
  /// live-capture tolerance.
  static BrailleCell? hitTest(
    Offset tip,
    CellMap cellMap, {
    double marginFrac = 0.12,
    double marginPx = 0.0,
    bool skipEmpty = true,
    double nearestWithinCells = 1.0,
  }) {
    if (cellMap.cells.isEmpty) return null;

    var hits =
        cellMap.cells.where((c) => _contains(c, tip, marginFrac, marginPx)).toList();

    if (hits.isEmpty && nearestWithinCells > 0) {
      final radius = _medianCellWidth(cellMap.cells) * nearestWithinCells;
      if (radius > 0) {
        hits = cellMap.cells
            .where((c) => (c.center - tip).distance <= radius)
            .toList();
      }
    }
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

  /// Cell whose *centre* is closest to [point], within [withinCells] cell
  /// widths, or null if nothing is that close.
  ///
  /// For a [point] that is itself a mapped cell centre, centre-to-centre is
  /// the truer test: box containment would hand the answer to a neighbour as
  /// soon as the mapping drifts past the box edge, while the nearest centre
  /// stays correct until the drift approaches half a cell.
  static BrailleCell? nearestCell(
    Offset point,
    CellMap cellMap, {
    double withinCells = 0.75,
    bool skipEmpty = true,
  }) {
    final radius = _medianCellWidth(cellMap.cells) * withinCells;
    if (radius <= 0) return null;

    var candidates =
        cellMap.cells.where((c) => (c.center - point).distance <= radius).toList();
    if (candidates.isEmpty) return null;
    if (skipEmpty) {
      final letters = candidates.where((c) => c.code != 0).toList();
      if (letters.isNotEmpty) candidates = letters;
    }
    candidates.sort((a, b) => (a.center - point)
        .distanceSquared
        .compareTo((b.center - point).distanceSquared));
    return candidates.first;
  }

  /// Typical cell width, used as the unit for distance tolerances.
  static double medianCellWidth(CellMap cellMap) => _medianCellWidth(cellMap.cells);

  static double _medianCellWidth(List<BrailleCell> cells) {
    if (cells.isEmpty) return 0;
    final widths = cells.map((c) => c.x1 - c.x0).where((w) => w > 0).toList()..sort();
    if (widths.isEmpty) return 0;
    return widths[widths.length ~/ 2];
  }

  static bool _contains(
    BrailleCell cell,
    Offset tip,
    double marginFrac,
    double marginPx,
  ) {
    final w = cell.x1 - cell.x0;
    final h = cell.y1 - cell.y0;
    final mx = math.max(w * marginFrac, marginPx);
    final my = math.max(h * marginFrac, marginPx);
    return tip.dx >= cell.x0 - mx &&
        tip.dx <= cell.x1 + mx &&
        tip.dy >= cell.y0 - my &&
        tip.dy <= cell.y1 + my;
  }
}
