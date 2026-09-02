import 'dart:ui';

/// Box-fit contain math shared by image + cell overlays.
class ImageFit {
  static Rect fittedRect(Size viewSize, double imageAspect) {
    final viewAspect = viewSize.width / viewSize.height;
    if (viewAspect > imageAspect) {
      final h = viewSize.height;
      final w = h * imageAspect;
      return Rect.fromLTWH((viewSize.width - w) / 2, 0, w, h);
    }
    final w = viewSize.width;
    final h = w / imageAspect;
    return Rect.fromLTWH(0, (viewSize.height - h) / 2, w, h);
  }

  /// Map image pixel coordinate → view coordinate (box-fit contain).
  static Offset imageToView(Offset imagePoint, Size viewSize, int imageW, int imageH) {
    final rect = fittedRect(viewSize, imageW / imageH);
    return Offset(
      rect.left + imagePoint.dx * rect.width / imageW,
      rect.top + imagePoint.dy * rect.height / imageH,
    );
  }

  /// Map view tap → image pixel coordinate.
  static Offset? viewToImage(Offset viewPoint, Size viewSize, int imageW, int imageH) {
    final rect = fittedRect(viewSize, imageW / imageH);
    if (!rect.contains(viewPoint)) return null;
    return Offset(
      (viewPoint.dx - rect.left) * imageW / rect.width,
      (viewPoint.dy - rect.top) * imageH / rect.height,
    );
  }
}
