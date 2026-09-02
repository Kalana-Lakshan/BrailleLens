import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import 'frozen_image_view.dart';

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

  Future<void> _load() async {
    final codec = await ui.instantiateImageCodec(widget.jpeg);
    final frame = await codec.getNextFrame();
    if (mounted) setState(() => _image = frame.image);
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      backgroundColor: Colors.grey[900],
      title: const Text('Tap fingertip contact point', style: TextStyle(color: Colors.white)),
      content: SizedBox(
        width: 280,
        height: 360,
        child: _image == null
            ? const Center(child: CircularProgressIndicator())
            : GestureDetector(
                onTapDown: (d) {
                  final box = context.findRenderObject() as RenderBox?;
                  if (box == null || _image == null) return;
                  final local = box.globalToLocal(d.globalPosition);
                  final scale = 280 / _image!.width;
                  final scaleY = 360 / _image!.height;
                  final s = scale < scaleY ? scale : scaleY;
                  final dw = _image!.width * s;
                  final dh = _image!.height * s;
                  final ox = (280 - dw) / 2;
                  final oy = (360 - dh) / 2;
                  final ix = ((local.dx - ox) / s).clamp(0, _image!.width.toDouble());
                  final iy = ((local.dy - oy) / s).clamp(0, _image!.height.toDouble());
                  Navigator.pop(context, Offset(ix.toDouble(), iy.toDouble()));
                },
                child: CustomPaint(
                  painter: ImagePainter(_image!),
                  size: const Size(280, 360),
                ),
              ),
      ),
    );
  }
}
