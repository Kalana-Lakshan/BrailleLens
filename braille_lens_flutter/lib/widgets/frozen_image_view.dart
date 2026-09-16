import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../models/braille_cell.dart';
import '../services/fingertip_onnx_service.dart';
import '../utils/image_fit.dart';
import 'cell_overlay_painter.dart';

/// Renders a frozen (already-captured) JPEG with the cell-map boxes and/or
/// fingertip marker overlaid — shared between Learning and Testing Mode's
/// "here's what Stage 1 / Stage 2 saw" view.
class FrozenImageView extends StatelessWidget {
  final Uint8List jpeg;
  final CellMap? cellMap;
  final BrailleCell? highlighted;
  final FingertipDetection? fingertip;

  const FrozenImageView({
    super.key,
    required this.jpeg,
    this.cellMap,
    this.highlighted,
    this.fingertip,
  });

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<ui.Image>(
      future: _decode(jpeg),
      builder: (context, snap) {
        if (!snap.hasData) {
          return const Center(child: CircularProgressIndicator(color: Color(0xFFFFD700)));
        }
        final image = snap.data!;
        return LayoutBuilder(
          builder: (context, constraints) {
            final size = Size(constraints.maxWidth, constraints.maxHeight);
            final imgW = cellMap?.imageWidth ?? image.width;
            final imgH = cellMap?.imageHeight ?? image.height;

            return Stack(
              fit: StackFit.expand,
              children: [
                CustomPaint(
                  painter: ImagePainter(image),
                  size: size,
                ),
                if (cellMap != null)
                  CustomPaint(
                    painter: CellOverlayPainter(
                      cells: cellMap!.cells,
                      highlighted: highlighted,
                      imageWidth: imgW,
                      imageHeight: imgH,
                    ),
                    size: size,
                  ),
                if (fingertip != null)
                  CustomPaint(
                    painter: FingertipOverlayPainter(
                      tipBox: fingertip!.box,
                      contactPoint: fingertip!.contactPoint,
                      imageWidth: fingertip!.imageWidth,
                      imageHeight: fingertip!.imageHeight,
                    ),
                    size: size,
                  ),
              ],
            );
          },
        );
      },
    );
  }

  Future<ui.Image> _decode(Uint8List bytes) async {
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    return frame.image;
  }
}

class ImagePainter extends CustomPainter {
  final ui.Image image;
  ImagePainter(this.image);

  @override
  void paint(Canvas canvas, Size size) {
    final src = Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble());
    final dst = ImageFit.fittedRect(size, image.width / image.height);
    canvas.drawImageRect(image, src, dst, Paint());
  }

  @override
  bool shouldRepaint(ImagePainter old) => old.image != image;
}
