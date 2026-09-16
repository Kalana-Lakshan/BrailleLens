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
      return alignMode.startsWith('scale')
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
  ///
  /// The finger frame's own cell boxes then do most of the work. Asking which
  /// of *those* boxes the tip sits in is exact — tip and boxes were measured
  /// in the same picture — so the transform is only ever asked to say which
  /// prescan cell a whole box corresponds to. That question tolerates far
  /// more error than mapping a bare point: the boxes are a cell apart, so the
  /// answer stays right until the mapping drifts by half a cell, whereas a
  /// mapped point lands wrong as soon as it leaves its box.
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

    final touched = cellUnderTipInFingerFrame(
      tipInFingerImage: tipInFingerImage,
      fingerFrameCells: fingerFrameCells,
      fingerImageWidth: fingerImageWidth,
      fingerImageHeight: fingerImageHeight,
    );
    if (touched != null) {
      mode = '$mode+box';
      debugPrint('[CoveredCell] tip is in finger-frame box '
          '${touched.rect} of ${fingerFrameCells.length}');
    }

    return resolve(
      tipInFingerImage: tipInFingerImage,
      cellMap: cellMap,
      fingerImageWidth: fingerImageWidth,
      fingerImageHeight: fingerImageHeight,
      fingertipBox: fingertipBox,
      homography: h,
      alignMode: mode,
      // Map the touched box's centre rather than the tip: a box centre has a
      // matching box centre to land on in the prescan, so the lookup becomes
      // nearest-cell-to-a-cell instead of point-in-a-box.
      probeInFingerImage: touched?.center,
      matchNearestCentre: touched != null,
    );
  }

  /// Which of the finger frame's own detected boxes the tip is touching.
  /// Pure geometry within that one frame — no mapping involved.
  BrailleCell? cellUnderTipInFingerFrame({
    required Offset tipInFingerImage,
    required List<Rect> fingerFrameCells,
    required int fingerImageWidth,
    required int fingerImageHeight,
  }) {
    if (fingerFrameCells.isEmpty) return null;
    final frameMap = CellMap(
      cells: [
        for (var i = 0; i < fingerFrameCells.length; i++)
          BrailleCell(
            id: i,
            x0: fingerFrameCells[i].left,
            y0: fingerFrameCells[i].top,
            x1: fingerFrameCells[i].right,
            y1: fingerFrameCells[i].bottom,
          ),
      ],
      imageWidth: fingerImageWidth,
      imageHeight: fingerImageHeight,
    );
    // These boxes carry no labels, so code 0 means "unclassified" here rather
    // than "space" — skipping them would reject every one of them.
    return CellHitTest.hitTest(tipInFingerImage, frameMap, skipEmpty: false);
  }

  /// [tipInFingerImage] — contact point in stage-2 JPEG pixel coordinates.
  /// [cellMap] — stage-1 prescan with labels already assigned.
  /// [homography] — optional live→prescan 3x3; otherwise independent X/Y scale.
  /// [probeInFingerImage] — point to map instead of the tip, when something
  /// more mappable is known (see [resolveAligned]). The tip is still what
  /// gets reported.
  /// [matchNearestCentre] — the probe is a cell centre, so match it against
  /// prescan cell centres rather than testing box containment.
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
    Offset? probeInFingerImage,
    bool matchNearestCentre = false,
  }) {
    final probe = probeInFingerImage ?? tipInFingerImage;
    final tipInPrescan = homography != null
        ? homography.transform(probe)
        : CoordinateMapper.mapFingerTipToPrescan(
            tipInFingerImage: probe,
            prescanWidth: cellMap.imageWidth,
            prescanHeight: cellMap.imageHeight,
            fingerImageWidth: fingerImageWidth,
            fingerImageHeight: fingerImageHeight,
          );

    var hit = matchNearestCentre ? CellHitTest.nearestCell(tipInPrescan, cellMap) : null;
    // Centre matching comes up empty where the prescan dropped a cell as
    // background, so the box test still gets its say.
    hit ??= CellHitTest.hitTest(
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
