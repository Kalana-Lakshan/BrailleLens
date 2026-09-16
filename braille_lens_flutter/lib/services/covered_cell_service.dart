import 'dart:ui';

import 'package:flutter/foundation.dart';

import '../models/braille_cell.dart';
import '../utils/dot_sequence.dart';
import '../utils/homography.dart';
import 'cell_constellation_aligner.dart';
import 'cell_hit_test.dart';
import 'coordinate_mapper.dart';
import 'frame_registration.dart';

/// Result of covered-character lookup (geometry only — no CNN on finger photo).
class CoveredCellResult {
  final BrailleCell? cell;
  final Offset tipInPrescan;
  final Offset tipInFingerImage;
  final Rect? fingertipBox;
  final String alignMode;

  const CoveredCellResult({
    required this.cell,
    required this.tipInPrescan,
    required this.tipInFingerImage,
    this.fingertipBox,
    this.alignMode = 'scale',
  });

  bool get hasHit => cell != null;

  String get headline {
    final c = cell;
    if (c == null) return '—';
    if (c.char.trim().isNotEmpty && c.char != ' ') return c.char;
    return c.displayLabel;
  }

  /// Raised-dot sequence for the bottom terminal (e.g. 124, 34).
  String get compactDots => compactDotSequence(cell?.pattern);

  String get subtitle {
    final c = cell;
    if (c == null) {
      // 'scale' means alignment fell through to raw image ratios, which is
      // the usual reason a covered cell reads as a miss — say so instead of
      // leaving the learner to guess at their finger placement.
      return alignMode == 'scale'
          ? 'no cell under your finger · page not aligned — frame it as you scanned it'
          : 'no cell under your finger · align: $alignMode';
    }
    return '$headline · align: $alignMode';
  }
}

/// Identifies which prescan cell is covered by the fingertip.
class CoveredCellService {
  /// Full stage-2 lookup: work out how the finger frame sits on the prescan,
  /// then read the label off the prescan map.
  ///
  /// Alignment is attempted in order of how well it survives a Braille page:
  /// 1. `cells` — the detected cell constellation in both frames. Best, since
  ///    the cells are exactly the landmarks the hit-test cares about.
  /// 2. `homography` — pixel feature registration, for when the finger frame
  ///    detector came up short.
  /// 3. `scale` — plain width/height ratios, correct only if the phone barely
  ///    moved between the two shots.
  Future<CoveredCellResult> resolveAligned({
    required Uint8List fingerJpeg,
    required Offset tipInFingerImage,
    required CellMap cellMap,
    required int fingerImageWidth,
    required int fingerImageHeight,
    required List<Rect> fingerFrameCells,
    Rect? fingertipBox,
    Uint8List? prescanJpeg,
  }) async {
    Homography? h;
    var mode = 'scale';

    if (fingerFrameCells.length >= CellConstellationAligner.minInliers) {
      final aligned = await CellConstellationAligner.align(
        liveCenters: fingerFrameCells.map((b) => b.center).toList(),
        referenceCenters: cellMap.cells.map((c) => c.center).toList(),
        cellWidth: CellHitTest.medianCellWidth(cellMap),
      );
      if (aligned != null) {
        h = aligned.homography;
        mode = 'cells';
        debugPrint('[CoveredCell] cell alignment on ${aligned.inliers} cells');
      }
    }

    if (h == null && prescanJpeg != null) {
      final reg = await FrameRegistration.estimate(
        referenceJpeg: prescanJpeg,
        liveJpeg: fingerJpeg,
        liveMask: fingertipBox,
      );
      h = reg.homography;
      if (h != null) mode = 'homography';
    }

    return resolve(
      tipInFingerImage: tipInFingerImage,
      cellMap: cellMap,
      fingerImageWidth: fingerImageWidth,
      fingerImageHeight: fingerImageHeight,
      fingertipBox: fingertipBox,
      homography: h,
      alignMode: mode,
    );
  }

  /// [tipInFingerImage] — contact point in stage-2 JPEG pixel coordinates.
  /// [cellMap] — stage-1 prescan with labels already assigned.
  /// [homography] — optional live→prescan 3x3; otherwise independent X/Y scale.
  CoveredCellResult resolve({
    required Offset tipInFingerImage,
    required CellMap cellMap,
    required int fingerImageWidth,
    required int fingerImageHeight,
    Rect? fingertipBox,
    Homography? homography,
    String alignMode = 'scale',
    double marginFrac = 0.12,
    double nearestWithinCells = 1.0,
  }) {
    final tipInPrescan = homography != null
        ? homography.transform(tipInFingerImage)
        : CoordinateMapper.mapFingerTipToPrescan(
            tipInFingerImage: tipInFingerImage,
            prescanWidth: cellMap.imageWidth,
            prescanHeight: cellMap.imageHeight,
            fingerImageWidth: fingerImageWidth,
            fingerImageHeight: fingerImageHeight,
          );

    final hit = CellHitTest.hitTest(
      tipInPrescan,
      cellMap,
      marginFrac: marginFrac,
      skipEmpty: true,
      nearestWithinCells: nearestWithinCells,
    );

    return CoveredCellResult(
      cell: hit,
      tipInPrescan: tipInPrescan,
      tipInFingerImage: tipInFingerImage,
      fingertipBox: fingertipBox,
      alignMode: alignMode,
    );
  }
}
