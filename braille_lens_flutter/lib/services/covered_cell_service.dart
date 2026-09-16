import 'dart:ui';

import '../models/braille_cell.dart';
import '../utils/dot_sequence.dart';
import '../utils/homography.dart';
import 'cell_hit_test.dart';
import 'coordinate_mapper.dart';

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
    if (c == null) return 'no cell under tip · align: $alignMode';
    return '${headline} · align: $alignMode';
  }
}

/// Identifies which prescan cell is covered by the fingertip.
class CoveredCellService {
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
    );

    return CoveredCellResult(
      cell: hit,
      tipInPrescan: tipInPrescan,
      tipInFingerImage: tipInFingerImage,
      fingertipBox: fingertipBox,
      alignMode: homography != null ? 'homography' : alignMode,
    );
  }
}
