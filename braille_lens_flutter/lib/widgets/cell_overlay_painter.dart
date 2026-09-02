import 'package:flutter/material.dart';

import '../models/braille_cell.dart';
import '../utils/image_fit.dart';

/// Draws prescan cell boxes aligned with box-fit image display.
class CellOverlayPainter extends CustomPainter {
  final List<BrailleCell> cells;
  final BrailleCell? highlighted;
  final int imageWidth;
  final int imageHeight;

  CellOverlayPainter({
    required this.cells,
    this.highlighted,
    required this.imageWidth,
    required this.imageHeight,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (imageWidth <= 0 || imageHeight <= 0) return;

    final fit = ImageFit.fittedRect(size, imageWidth / imageHeight);

    final normalPaint = Paint()
      ..color = const Color(0xFFFFD700).withValues(alpha: 0.85)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2;

    final hitPaint = Paint()
      ..color = const Color(0xFF00E5FF)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5;

    for (final cell in cells) {
      final isHit = highlighted != null && highlighted!.id == cell.id;
      final paint = isHit ? hitPaint : normalPaint;
      final rect = Rect.fromLTRB(
        fit.left + cell.x0 * fit.width / imageWidth,
        fit.top + cell.y0 * fit.height / imageHeight,
        fit.left + cell.x1 * fit.width / imageWidth,
        fit.top + cell.y1 * fit.height / imageHeight,
      );
      canvas.drawRect(rect, paint);
    }
  }

  @override
  bool shouldRepaint(CellOverlayPainter oldDelegate) {
    return oldDelegate.cells != cells ||
        oldDelegate.highlighted?.id != highlighted?.id;
  }
}

/// Fingertip box + contact dot on stage-2 image.
class FingertipOverlayPainter extends CustomPainter {
  final Rect? tipBox;
  final Offset? contactPoint;
  final int imageWidth;
  final int imageHeight;

  FingertipOverlayPainter({
    this.tipBox,
    this.contactPoint,
    required this.imageWidth,
    required this.imageHeight,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (imageWidth <= 0 || imageHeight <= 0) return;
    final fit = ImageFit.fittedRect(size, imageWidth / imageHeight);

    if (tipBox != null) {
      final r = Rect.fromLTRB(
        fit.left + tipBox!.left * fit.width / imageWidth,
        fit.top + tipBox!.top * fit.height / imageHeight,
        fit.left + tipBox!.right * fit.width / imageWidth,
        fit.top + tipBox!.bottom * fit.height / imageHeight,
      );
      canvas.drawRect(
        r,
        Paint()
          ..color = const Color(0xFFFFD700)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2,
      );
      canvas.drawRect(
        r,
        Paint()
          ..color = const Color(0xFF00E5FF).withValues(alpha: 0.35)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.5,
      );
    }

    if (contactPoint != null) {
      final p = Offset(
        fit.left + contactPoint!.dx * fit.width / imageWidth,
        fit.top + contactPoint!.dy * fit.height / imageHeight,
      );
      canvas.drawCircle(p, 6, Paint()..color = const Color(0xFFFF1744));
    }
  }

  @override
  bool shouldRepaint(FingertipOverlayPainter old) =>
      old.tipBox != tipBox || old.contactPoint != contactPoint;
}
