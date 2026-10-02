import 'dart:math' as math;
import 'dart:ui' show Offset;

import '../models/braille_cell.dart';
import 'cell_hit_test.dart';

/// Dwell identity for the hands-free session: the same id while the
/// fingertip holds still, a new one when it jumps by more than a cell.
///
/// The session's dwell counts consecutive samples with the same cell id.
/// Getting a real cell id meant running the full-page cell detector on every
/// sample — 4-5 s each on a mid-range phone (Galaxy M14), so two matching
/// samples in a row almost never happened and the 3 s lock never started.
/// "The finger is holding still" is all the dwell needs; which cell it is
/// comes from the full aligned lookup at lock time.
///
/// Each sample is compared with the previous one, not with where the dwell
/// began, so a slow drift keeps the id while a move to another letter does
/// not.
class TipDwellTracker {
  TipDwellTracker({this.stillCells = 1.0, this.minRadiusPx = 16.0});

  /// Movement between samples, in cell widths, that still counts as still.
  final double stillCells;

  /// Floor for the radius, for a page map with implausibly small cells.
  final double minRadiusPx;

  Offset? _last;
  int _id = 0;

  /// Dwell id for a tip at [tip] in an image [imageWidth] pixels wide, with
  /// cell sizes taken from the scanned [map].
  int track(Offset tip, int imageWidth, CellMap map) {
    // Prescan and finger frames come from the same camera, so a cell is
    // about as wide in both once scaled by the image widths.
    final scale = map.imageWidth == 0 ? 1.0 : imageWidth / map.imageWidth;
    final cellWidth = CellHitTest.medianCellWidth(map) * scale;
    final radius = math.max(cellWidth * stillCells, minRadiusPx);

    final last = _last;
    if (last == null || (tip - last).distance > radius) _id++;
    _last = tip;
    return _id;
  }

  /// The tip left the frame: the next sighting starts a new dwell.
  void lost() => _last = null;
}
