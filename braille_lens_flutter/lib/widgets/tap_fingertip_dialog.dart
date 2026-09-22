import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../utils/image_decode.dart';
import '../utils/image_fit.dart';
import 'frozen_image_view.dart';

/// Size of the painted photo inside the dialog; taps are mapped through it.
const double _paintWidth = 280;
const double _paintHeight = 360;

/// Fallback when the ONNX fingertip model is missing/fails: shows the just-
/// captured photo and lets the user tap where their fingertip is, returning
/// that point in the photo's own pixel coordinates (or null if dismissed).
class TapFingertipDialog extends StatefulWidget {
  final Uint8List jpeg;
  const TapFingertipDialog({super.key, required this.jpeg});

  @override
  State<TapFingertipDialog> createState() => _TapFingertipDialogState();
}

class _TapFingertipDialogState extends State<TapFingertipDialog> {
  ui.Image? _image;

  @override
  void initState() {
    super.initState();
    _load();
  }

  /// Upright: the point this dialog returns is used as a fingertip position in
  /// `decodeUpright` space, so it has to be picked on that same raster.
  Future<void> _load() async {
    final image = await decodeUprightUi(widget.jpeg);
    if (mounted) setState(() => _image = image);
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      backgroundColor: Colors.grey[900],
      title: const Text('Tap fingertip contact point', style: TextStyle(color: Colors.white)),
      content: SizedBox(
        width: _paintWidth,
        height: _paintHeight,
        child: _image == null
            ? const Center(child: CircularProgressIndicator())
            // localPosition is relative to this GestureDetector, which is
            // exactly the painted area. Using the dialog's own RenderBox
            // instead (as this did) offsets every tap by the title and
            // padding above the image, so the returned "fingertip" sits on
            // the wrong cell.
            : GestureDetector(
                onTapDown: (d) {
                  final image = _image;
                  if (image == null) return;
                  final point = ImageFit.viewToImage(
                    d.localPosition,
                    const Size(_paintWidth, _paintHeight),
                    image.width,
                    image.height,
                  );
                  // Null means the tap landed on the letterbox bars, where
                  // there is no pixel to report.
                  if (point == null) return;
                  Navigator.pop(context, point);
                },
                child: CustomPaint(
                  painter: ImagePainter(_image!),
                  size: const Size(_paintWidth, _paintHeight),
                ),
              ),
      ),
    );
  }
}
