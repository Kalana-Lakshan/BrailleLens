import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

import '../models/braille_cell.dart';
import 'classifier_service.dart';
import 'dot_cell_detector.dart';

/// Stage-1 on-device prescan: dot grid + `braille_model.onnx` per cell.
class PrescanOnnxService {
  final ClassifierService _cnn;

  PrescanOnnxService({ClassifierService? classifier})
      : _cnn = classifier ?? ClassifierService();

  bool get isReady => _cnn.isInitialized;

  Future<bool> initialize() => _cnn.initialize();

  /// Build [CellMap] from a full-page JPEG using bundled CNN.
  Future<CellMap> prescanPage(
    Uint8List jpegBytes, {
    void Function(int done, int total)? onProgress,
  }) async {
    if (!_cnn.isInitialized) {
      final ok = await _cnn.initialize();
      if (!ok) {
        throw Exception(_cnn.lastError ?? 'braille_model.onnx failed to load');
      }
    }

    final decoded = img.decodeImage(jpegBytes);
    if (decoded == null) {
      throw Exception('Could not decode page image');
    }

    final det = DotCellDetector.detectCellBoxes(jpegBytes);
    if (det.boxes.isEmpty) {
      throw Exception(
        'No Braille cells found — use even lighting and fill the frame with the page',
      );
    }

    final cells = <BrailleCell>[];
    final total = det.boxes.length;

    for (var i = 0; i < total; i++) {
      final b = det.boxes[i];
      final x0 = b.x0.round().clamp(0, decoded.width - 1);
      final y0 = b.y0.round().clamp(0, decoded.height - 1);
      final x1 = b.x1.round().clamp(x0 + 1, decoded.width);
      final y1 = b.y1.round().clamp(y0 + 1, decoded.height);

      final crop = img.copyCrop(
        decoded,
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
      );

      try {
        final pred = await _cnn.predictCrop(crop);
        final ch = pred.character.toUpperCase();
        cells.add(
          BrailleCell(
            id: i,
            x0: x0.toDouble(),
            y0: y0.toDouble(),
            x1: x1.toDouble(),
            y1: y1.toDouble(),
            char: ch,
            pattern: '',
            code: pred.classIndex + 1,
            conf: pred.confidence,
            line: 0,
            col: i,
          ),
        );
      } catch (e) {
        debugPrint('[PrescanOnnx] cell $i classify error: $e');
      }
      onProgress?.call(i + 1, total);
    }

    if (cells.isEmpty) {
      throw Exception('CNN could not classify any cells');
    }

    debugPrint('[PrescanOnnx] ${cells.length} cells classified on-device');
    return CellMap(
      cells: cells,
      imageWidth: det.width,
      imageHeight: det.height,
    );
  }

  void dispose() => _cnn.dispose();
}
