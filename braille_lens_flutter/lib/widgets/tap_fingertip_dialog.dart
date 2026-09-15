import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../utils/image_fit.dart';

Future<Offset?> showTapFingertipDialog({
  required BuildContext context,
  required Uint8List jpeg,
}) {
  return showDialog<Offset>(
    context: context,
    barrierDismissible: false,
    builder: (ctx) => _TapFingertipDialog(jpeg: jpeg),
  );
}

class _TapFingertipDialog extends StatefulWidget {
  final Uint8List jpeg;
  const _TapFingertipDialog({required this.jpeg});

  @override
  State<_TapFingertipDialog> createState() => _TapFingertipDialogState();
}

class _TapFingertipDialogState extends State<_TapFingertipDialog> {
  ui.Image? _image;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final codec = await ui.instantiateImageCodec(widget.jpeg);
    final frame = await codec.getNextFrame();
    if (!mounted) return;
    setState(() => _image = frame.image);
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
                  if (_image == null) return;
                  final mapped = ImageFit.viewToImage(
                    d.localPosition,
                    const Size(280, 360),
                    _image!.width,
                    _image!.height,
                  );
                  if (mapped == null) return;
                  Navigator.pop(context, mapped);
                },
                child: CustomPaint(
                  painter: _StillPainter(_image!),
                  size: const Size(280, 360),
                ),
              ),
      ),
    );
  }
}

class _StillPainter extends CustomPainter {
  final ui.Image image;
  _StillPainter(this.image);

  @override
  void paint(Canvas canvas, Size size) {
    final src = Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble());
    final dst = ImageFit.fittedRect(size, image.width / image.height);
    canvas.drawImageRect(image, src, dst, Paint());
  }

  @override
  bool shouldRepaint(_StillPainter old) => old.image != image;
}
