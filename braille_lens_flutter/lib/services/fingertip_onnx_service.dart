import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

import '../config/app_config.dart';

/// YOLO26n fingertip detector — stage 2 only (where the finger is).
/// Covered character comes from prescan hit-test, not this model.
class FingertipOnnxService {
  OrtSession? _session;
  String? _loadedAsset;
  String? _lastError;
  bool _ready = false;

  bool get isReady => _ready;
  String? get loadedAsset => _loadedAsset;
  String? get lastError => _lastError;

  Future<bool> initialize() async {
    if (_ready) return true;
    for (final asset in [
      AppConfig.fingertipOnnxAsset,
      AppConfig.fingertipOnnxFallbackAsset,
    ]) {
      if (await _tryLoad(asset)) return true;
    }
    debugPrint('[FingertipOnnx] no model loaded — tap fingertip on photo');
    return false;
  }

  Future<bool> _tryLoad(String assetPath) async {
    try {
      OrtEnv.instance.init();
      final bytes = await rootBundle.load(assetPath);
      _session?.release();
      _session = OrtSession.fromBuffer(
        bytes.buffer.asUint8List(),
        OrtSessionOptions(),
      );
      _loadedAsset = assetPath;
      _lastError = null;
      _ready = true;
      debugPrint('[FingertipOnnx] loaded $assetPath');
      return true;
    } catch (e) {
      _lastError = e.toString();
      debugPrint('[FingertipOnnx] failed $assetPath: $e');
      return false;
    }
  }

  /// Returns contact point + box in original image pixel coordinates.
  Future<FingertipDetection?> detect(Uint8List jpegBytes) async {
    if (!_ready && !await initialize()) return null;
    final session = _session;
    if (session == null) return null;

    final decoded = img.decodeImage(jpegBytes);
    if (decoded == null) return null;

    final origW = decoded.width;
    final origH = decoded.height;
    const imgsz = 640;

    final letterbox = _letterbox(decoded, imgsz);
    final input = _toTensor(letterbox.image, imgsz);

    final inputTensor = OrtValueTensor.createTensorWithDataList(
      input,
      [1, 3, imgsz, imgsz],
    );
    final runOptions = OrtRunOptions();
    final outputs = await session.runAsync(runOptions, {'images': inputTensor});
    inputTensor.release();
    runOptions.release();

    if (outputs == null || outputs.isEmpty) {
      return null;
    }

    dynamic outValue;
    if (outputs is Map) {
      outValue = outputs['output0']?.value ?? outputs.values.first?.value;
      for (final o in outputs.values) {
        o?.release();
      }
    } else {
      outValue = outputs[0]?.value ?? outputs[0];
      for (final o in outputs) {
        o?.release();
      }
    }

    final rows = _parseOutputRows(outValue);
    const confThresh = 0.25;
    Map<String, dynamic>? best;
    for (final row in rows) {
      if (row.length < 5) continue;
      var x1 = row[0];
      var y1 = row[1];
      var x2 = row[2];
      var y2 = row[3];
      final conf = row.length >= 5 ? row[4] : 0.0;
      final cls = row.length >= 6 ? row[5].round() : 0;
      if (conf < confThresh || cls != 0) continue;
      if (x2 < x1 || y2 < y1) {
        final cx = row[0];
        final cy = row[1];
        final w = row[2];
        final h = row[3];
        x1 = cx - w / 2;
        y1 = cy - h / 2;
        x2 = cx + w / 2;
        y2 = cy + h / 2;
      }
      if (best == null || conf > (best['conf'] as double)) {
        best = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'conf': conf};
      }
    }
    if (best == null) return null;

    final x1 = _unmap((best['x1'] as num).toDouble(), letterbox);
    final y1 = _unmapY((best['y1'] as num).toDouble(), letterbox);
    final x2 = _unmap((best['x2'] as num).toDouble(), letterbox);
    final y2 = _unmapY((best['y2'] as num).toDouble(), letterbox);

    final box = Rect.fromLTRB(
      x1.clamp(0, origW.toDouble()),
      y1.clamp(0, origH.toDouble()),
      x2.clamp(0, origW.toDouble()),
      y2.clamp(0, origH.toDouble()),
    );
    // Pad contact (bottom-centre) — better for cell hit-test than box centre.
    final contact = Offset(
      box.center.dx,
      box.bottom - (box.height * 0.08).clamp(1.0, 12.0),
    );

    debugPrint(
      '[FingertipOnnx] $_loadedAsset conf=${best['conf']} '
      'contact=(${contact.dx.toStringAsFixed(0)}, ${contact.dy.toStringAsFixed(0)})',
    );

    return FingertipDetection(
      contactPoint: contact,
      box: box,
      confidence: best['conf'] as double,
      imageWidth: origW,
      imageHeight: origH,
    );
  }

  void dispose() {
    _session?.release();
    _session = null;
    _ready = false;
  }

    List<List<double>> _parseOutputRows(dynamic value) {
    if (value is! List || value.isEmpty) return [];
    dynamic node = value;
    if (node.length == 1 && node[0] is List) {
      node = node[0];
    }
    if (node is! List || node.isEmpty) return [];
    if (node[0] is! List) return [];

    final asRows = <List<double>>[];
    for (final r in node) {
      if (r is List && r.isNotEmpty && r[0] is num) {
        asRows.add(r.map((e) => (e as num).toDouble()).toList());
      }
    }
    if (asRows.isEmpty) return [];

    // Ultralytics ONNX is often [C, N] (e.g. 6 x 8400), not [N, C].
    if (asRows.length <= 16 && asRows.first.length > 32) {
      final c = asRows.length;
      final n = asRows.first.length;
      return List.generate(n, (i) => List.generate(c, (k) => asRows[k][i]));
    }
    return asRows;
  }

  Float32List _toTensor(img.Image rgb, int size) {
    final nchw = Float32List(3 * size * size);
    final plane = size * size;
    for (var y = 0; y < size; y++) {
      for (var x = 0; x < size; x++) {
        final p = rgb.getPixel(x, y);
        final idx = y * size + x;
        nchw[idx] = p.r / 255.0;
        nchw[plane + idx] = p.g / 255.0;
        nchw[2 * plane + idx] = p.b / 255.0;
      }
    }
    return nchw;
  }

  _Letterbox _letterbox(img.Image src, int size) {
    final scale = min(size / src.width, size / src.height);
    final nw = (src.width * scale).round();
    final nh = (src.height * scale).round();
    final resized = img.copyResize(src, width: nw, height: nh);
    final canvas = img.Image(width: size, height: size);
    img.fill(canvas, color: img.ColorRgb8(114, 114, 114));
    final padX = ((size - nw) / 2).round();
    final padY = ((size - nh) / 2).round();
    img.compositeImage(canvas, resized, dstX: padX, dstY: padY);
    return _Letterbox(
      image: canvas,
      scale: scale,
      padX: padX.toDouble(),
      padY: padY.toDouble(),
    );
  }

  double _unmap(double v, _Letterbox lb) => (v - lb.padX) / lb.scale;
  double _unmapY(double v, _Letterbox lb) => (v - lb.padY) / lb.scale;
}

class FingertipDetection {
  final Offset contactPoint;
  final Rect box;
  final double confidence;
  final int imageWidth;
  final int imageHeight;

  const FingertipDetection({
    required this.contactPoint,
    required this.box,
    required this.confidence,
    required this.imageWidth,
    required this.imageHeight,
  });
}

class _Letterbox {
  final img.Image image;
  final double scale;
  final double padX;
  final double padY;

  _Letterbox({
    required this.image,
    required this.scale,
    required this.padX,
    required this.padY,
  });
}
