import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

import '../models/braille_cell.dart';
import 'cell_detector_service.dart';
import 'classifier_service.dart';

/// Stage-1 on-device prescan: multi-cell YOLO26n detector
/// (`braille_cell_yolo26n*.onnx`) finds every cell box, then
/// `braille_cnn.onnx` classifies each crop.
class PrescanOnnxService {
  final ClassifierService _cnn;
  final CellDetectorService _detector;

  PrescanOnnxService({ClassifierService? classifier, CellDetectorService? detector})
      : _cnn = classifier ?? ClassifierService(),
        _detector = detector ?? CellDetectorService();

  bool get isReady => _cnn.isInitialized && _detector.isReady;

  /// Exposes the loaded classifier (e.g. its `.labels`) without loading a
  /// second copy of the model elsewhere.
  ClassifierService get classifier => _cnn;

  Future<bool> initialize() async {
    final results = await Future.wait([_cnn.initialize(), _detector.initialize()]);
    return results.every((ok) => ok);
  }

  /// Build [CellMap] from a full-page JPEG: detect every cell box, classify
  /// each one, and drop background/space tokens from the result.
  Future<CellMap> prescanPage(
    Uint8List jpegBytes, {
    void Function(int done, int total)? onProgress,
  }) async {
    if (!_cnn.isInitialized) {
      final ok = await _cnn.initialize();
      if (!ok) {
        throw Exception(_cnn.lastError ?? 'braille_cnn.onnx failed to load');
      }
    }
    if (!_detector.isReady) {
      final ok = await _detector.initialize();
      if (!ok) {
        throw Exception(_detector.lastError ?? 'braille_cell_yolo26n.onnx failed to load');
      }
    }

    final decoded = img.decodeImage(jpegBytes);
    if (decoded == null) {
      throw Exception('Could not decode page image');
    }

    final detections = await _detector.detect(jpegBytes);
    if (detections.isEmpty) {
      // The models loaded fine and this ran — it just found nothing to
      // classify. Log that distinctly from a load/init failure so it's
      // obvious this is a framing/lighting problem, not a broken model.
      debugPrint('Prescan returned 0 detections '
          '(page ${decoded.width}x${decoded.height}px, detector found no cell boxes)');
      throw Exception(
        'No Braille cells found — use even lighting and fill the frame with the page',
      );
    }

    final cells = <BrailleCell>[];
    final total = detections.length;
    var droppedBackground = 0;

    for (var i = 0; i < total; i++) {
      final b = detections[i].box;
      final x0 = b.left.round().clamp(0, decoded.width - 1);
      final y0 = b.top.round().clamp(0, decoded.height - 1);
      final x1 = b.right.round().clamp(x0 + 1, decoded.width);
      final y1 = b.bottom.round().clamp(y0 + 1, decoded.height);

      final crop = img.copyCrop(
        decoded,
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
      );

      try {
        // The CNN's class index IS the 6-dot cell code -- no label
        // round-tripping through an English letter (see AppConfig.brailleCnnAsset).
        final pred = await _cnn.predictCrop(crop);

        // Exclude background/space tokens from the primary detection list:
        // code 0 is the space/blank cell, '#<code>' is a real dot pattern
        // just outside the curated label chart -- neither is a nameable
        // character worth surfacing to the learner.
        final isBackground = pred.classIndex == 0 ||
            pred.character.trim().isEmpty ||
            pred.character.startsWith('#');
        if (isBackground) {
          droppedBackground++;
          onProgress?.call(i + 1, total);
          continue;
        }

        cells.add(
          BrailleCell(
            id: cells.length,
            x0: x0.toDouble(),
            y0: y0.toDouble(),
            x1: x1.toDouble(),
            y1: y1.toDouble(),
            char: pred.character,
            pattern: pred.dots,
            code: pred.classIndex,
            conf: pred.confidence,
            line: 0,
            col: cells.length,
          ),
        );
      } catch (e) {
        debugPrint('[PrescanOnnx] cell $i classify error: $e');
      }
      onProgress?.call(i + 1, total);
    }

    if (cells.isEmpty) {
      debugPrint('Prescan returned 0 detections '
          '($total box(es) found, $droppedBackground background/unmapped, '
          'rest failed to classify)');
      throw Exception('CNN could not classify any cells');
    }

    debugPrint('[PrescanOnnx] $total box(es) -> ${cells.length} cell(s) kept '
        '(dropped $droppedBackground background/space)');
    return CellMap(
      cells: cells,
      imageWidth: decoded.width,
      imageHeight: decoded.height,
    );
  }

  void dispose() {
    _cnn.dispose();
    _detector.dispose();
  }
}
