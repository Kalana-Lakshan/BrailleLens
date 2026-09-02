import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

import '../config/app_config.dart';

/// One detected Braille cell box, before classification.
class CellDetection {
  final Rect box; // original image pixel coordinates
  final double confidence;
  const CellDetection({required this.box, required this.confidence});
}

/// Multi-cell page detector — Stage 1's "where are the cells" step.
///
/// YOLO26n (`braille_cell_yolo26n*.onnx`), single class (`braille_cell`),
/// 1280×1280 input. Same YOLO26 family and export pattern as
/// [FingertipOnnxService]'s model, but unlike it, this returns *every* box
/// above threshold — a page has many cells, not one fingertip.
class CellDetectorService {
  OrtSession? _session;
  String? _loadedAsset;
  String? _lastError;
  bool _ready = false;

  static const int _imgsz = 1280;
  static const double _confThreshold = 0.50;
  static const double _iouThreshold = 0.40;

  bool get isReady => _ready;
  String? get loadedAsset => _loadedAsset;
  String? get lastError => _lastError;

  Future<bool> initialize() async {
    if (_ready) return true;
    for (final asset in [
      AppConfig.cellDetectorOnnxAsset,
      AppConfig.cellDetectorOnnxFallbackAsset,
    ]) {
      if (await _tryLoad(asset)) return true;
    }
    debugPrint('[CellDetector] no model loaded');
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
      debugPrint('[CellDetector] loaded $assetPath');
      return true;
    } catch (e) {
      _lastError = e.toString();
      debugPrint('[CellDetector] failed $assetPath: $e');
      return false;
    }
  }

  /// Every Braille-cell box found in [jpegBytes], in original image pixel
  /// coordinates, confidence-filtered and NMS'd.
  Future<List<CellDetection>> detect(Uint8List jpegBytes) async {
    if (!_ready && !await initialize()) return const [];
    final session = _session;
    if (session == null) return const [];

    final decoded = img.decodeImage(jpegBytes);
    if (decoded == null) return const [];

    final origW = decoded.width;
    final origH = decoded.height;

    final letterbox = _letterbox(decoded, _imgsz);
    final input = _toTensor(letterbox.image, _imgsz);

    final inputTensor = OrtValueTensor.createTensorWithDataList(
      input,
      [1, 3, _imgsz, _imgsz],
    );
    final runOptions = OrtRunOptions();
    // outputs is a List<OrtValue?> (not a Map) -- index it, don't key it.
    final outputs = await session.runAsync(runOptions, {'images': inputTensor});
    inputTensor.release();
    runOptions.release();

    if (outputs == null || outputs.isEmpty) {
      return const [];
    }
    final dynamic outValue = outputs[0]?.value;
    for (final o in outputs) {
      o?.release();
    }

    // output0: [1, N, 6], rows = [x1, y1, x2, y2, conf, cls] in 1280x1280
    // letterbox space -- already de-duplicated by YOLO26's end-to-end head
    // (verified empirically: no overlapping duplicate boxes in practice),
    // but NMS below is kept as a safety net rather than assumed.
    final rows = _parseOutputRows(outValue);

    final candidates = <CellDetection>[];
    for (final row in rows) {
      if (row.length < 6) continue;
      final conf = (row[4] as num).toDouble();
      final cls = (row[5] as num).toInt();
      if (conf < _confThreshold || cls != 0) continue;

      final x1 = _unmap((row[0] as num).toDouble(), letterbox);
      final y1 = _unmapY((row[1] as num).toDouble(), letterbox);
      final x2 = _unmap((row[2] as num).toDouble(), letterbox);
      final y2 = _unmapY((row[3] as num).toDouble(), letterbox);

      final box = Rect.fromLTRB(
        x1.clamp(0, origW.toDouble()),
        y1.clamp(0, origH.toDouble()),
        x2.clamp(0, origW.toDouble()),
        y2.clamp(0, origH.toDouble()),
      );
      if (box.width <= 0 || box.height <= 0) continue;
      candidates.add(CellDetection(box: box, confidence: conf));
    }

    final kept = _nms(candidates, _iouThreshold);
    debugPrint('[CellDetector] $_loadedAsset  ${rows.length} raw rows -> '
        '${candidates.length} above conf $_confThreshold -> '
        '${kept.length} after NMS (iou $_iouThreshold)');
    return kept;
  }

  /// Greedy NMS, highest confidence first. Safety net for the end-to-end
  /// head's already-de-duplicated output — a no-op when it holds, cheap
  /// insurance when it doesn't.
  List<CellDetection> _nms(List<CellDetection> dets, double iouThreshold) {
    final sorted = List<CellDetection>.from(dets)
      ..sort((a, b) => b.confidence.compareTo(a.confidence));
    final kept = <CellDetection>[];
    for (final d in sorted) {
      var overlaps = false;
      for (final k in kept) {
        if (_iou(d.box, k.box) > iouThreshold) {
          overlaps = true;
          break;
        }
      }
      if (!overlaps) kept.add(d);
    }
    return kept;
  }

  double _iou(Rect a, Rect b) {
    final inter = a.intersect(b);
    final interArea = inter.width > 0 && inter.height > 0 ? inter.width * inter.height : 0.0;
    if (interArea <= 0) return 0.0;
    final union = a.width * a.height + b.width * b.height - interArea;
    return union <= 0 ? 0.0 : interArea / union;
  }

  void dispose() {
    _session?.release();
    _session = null;
    _ready = false;
  }

  List<List<dynamic>> _parseOutputRows(dynamic value) {
    if (value is! List || value.isEmpty) return [];
    final outer = value[0];
    if (outer is! List) return [];
    return outer.map((r) => r is List ? r : <dynamic>[]).toList();
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
