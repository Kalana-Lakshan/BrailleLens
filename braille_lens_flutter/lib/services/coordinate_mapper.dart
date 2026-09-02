import 'dart:ui';

import '../models/braille_cell.dart';

/// Maps fingertip coordinates from the stage-2 finger photo into the
/// stage-1 prescan reference frame.
class CoordinateMapper {
  /// Scale tip from finger-image pixels to prescan-image pixels.
  ///
  /// Both photos should be taken with the phone held still over the page.
  /// Uses independent X/Y scale when aspect ratios differ slightly.
  static Offset mapFingerTipToPrescan({
    required Offset tipInFingerImage,
    required int prescanWidth,
    required int prescanHeight,
    required int fingerImageWidth,
    required int fingerImageHeight,
  }) {
    if (prescanWidth <= 0 ||
        prescanHeight <= 0 ||
        fingerImageWidth <= 0 ||
        fingerImageHeight <= 0) {
      return tipInFingerImage;
    }
    return Offset(
      tipInFingerImage.dx * prescanWidth / fingerImageWidth,
      tipInFingerImage.dy * prescanHeight / fingerImageHeight,
    );
  }

  /// Fingertip contact point: bottom-centre of the detection box (pad on page).
  static Offset contactPointFromBox(Rect box) {
    return Offset(
      box.center.dx,
      box.bottom - (box.height * 0.08).clamp(1.0, 12.0),
    );
  }

  /// Map a prescan cell rect into finger-image space (for drawing overlay).
  static Rect mapCellToFingerImage({
    required BrailleCell cell,
    required int prescanWidth,
    required int prescanHeight,
    required int fingerImageWidth,
    required int fingerImageHeight,
  }) {
    final sx = fingerImageWidth / prescanWidth;
    final sy = fingerImageHeight / prescanHeight;
    return Rect.fromLTRB(
      cell.x0 * sx,
      cell.y0 * sy,
      cell.x1 * sx,
      cell.y1 * sy,
    );
  }
}
