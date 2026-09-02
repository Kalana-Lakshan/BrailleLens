import 'dart:convert';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:image/image.dart' as img;

import '../config/app_config.dart';

/// One row from `assets/models/braille_labels.json`.
class BrailleLabel {
  final int code; // 6-dot cell code == CNN class index (0..63)
  final String si; // Sinhala grapheme / base form
  final String en; // English Grade-1 letter for the same dot pattern (reference only)
  final String dots; // e.g. "dots 1, 3"

  const BrailleLabel(this.code, this.si, this.en, this.dots);
}

class PredictionResult {
  final String character; // Sinhala grapheme (or '#<code>' if unmapped)
  final double confidence;
  final int classIndex; // == 6-dot cell code
  final BrailleLabel? label;

  PredictionResult({
    required this.character,
    required this.confidence,
    required this.classIndex,
    this.label,
  });

  String get dots => label?.dots ?? '(code $classIndex)';
}

/// 64-class Sinhala Braille cell CNN (`braille_cnn.onnx`, 64×64 grayscale,
/// class index == the 6-dot cell code). See [AppConfig.brailleCnnAsset].
class ClassifierService {
  static const int _imgSize = 64;

  OrtSession? _session;
  final Map<int, BrailleLabel> _labels = {};
  bool _isInitialized = false;
  String? _loadedAsset;
  String? _lastError;

  bool get isInitialized => _isInitialized;
  String? get loadedAsset => _loadedAsset;
  String? get lastError => _lastError;

  Future<bool> initialize() async {
    if (_isInitialized) return true;

    // Split into two stages, each logged separately: an asset-not-found
    // error (bad path / not listed in pubspec.yaml) and an ONNX session
    // creation error (corrupt file, unsupported IR version, etc.) look very
    // different in $e but were previously caught together — split them so
    // the debug console tells you which one actually happened.
    Uint8List modelBytes;
    try {
      final data = await rootBundle.load(AppConfig.brailleCnnAsset);
      modelBytes = data.buffer.asUint8List();
    } catch (e) {
      _lastError = e.toString();
      _isInitialized = false;
      debugPrint('Failed to load asset: $e');
      debugPrint(
          '[Classifier] asset not found at "${AppConfig.brailleCnnAsset}" — '
          'check the path matches assets/models/ and is listed under pubspec.yaml\'s flutter/assets:');
      return false;
    }

    try {
      OrtEnv.instance.init();
      final sessionOptions = OrtSessionOptions();
      _session = OrtSession.fromBuffer(modelBytes, sessionOptions);

      final labelsRaw = await rootBundle.loadString(AppConfig.brailleLabelsAsset);
      _labels.clear();
      for (final e in (jsonDecode(labelsRaw) as List)) {
        final m = e as Map<String, dynamic>;
        final code = m['code'] as int;
        _labels[code] = BrailleLabel(
          code,
          (m['si'] ?? '') as String,
          (m['en'] ?? '') as String,
          (m['dots'] ?? '') as String,
        );
      }

      _loadedAsset = AppConfig.brailleCnnAsset;
      _lastError = null;
      _isInitialized = true;
      debugPrint('[Classifier] loaded $_loadedAsset (${_labels.length} labels)');
      return true;
    } catch (e) {
      _lastError = e.toString();
      _isInitialized = false;
      debugPrint('Failed to load asset: $e');
      debugPrint('[Classifier] asset bytes loaded OK but ONNX session/tensor '
          'init failed — likely a corrupt file or unsupported IR version, '
          'not a missing-asset problem: $e');
      return false;
    }
  }

  Future<PredictionResult> predict(Uint8List imageBytes) async {
    if (!_isInitialized || _session == null) {
      final ok = await initialize();
      if (!ok || _session == null) {
        throw Exception(_lastError ?? 'CNN model not loaded');
      }
    }

    final originalImage = img.decodeImage(imageBytes);
    if (originalImage == null) {
      throw Exception('Failed to decode image');
    }
    return predictCrop(originalImage);
  }

  /// Classify a single cell crop (64×64 grayscale inside the model,
  /// per-crop brightness/contrast standardised — mirrors
  /// `braille_cnn/normalize.py::normalize_crop` exactly).
  Future<PredictionResult> predictCrop(img.Image crop) async {
    if (!_isInitialized || _session == null) {
      final ok = await initialize();
      if (!ok || _session == null) {
        throw Exception(_lastError ?? 'CNN model not loaded');
      }
    }

    final resized = img.copyResize(
      crop,
      width: _imgSize,
      height: _imgSize,
      interpolation: img.Interpolation.cubic,
    );
    final grayscale = img.grayscale(resized);

    final px = Float32List(_imgSize * _imgSize);
    var i = 0;
    for (var y = 0; y < _imgSize; y++) {
      for (var x = 0; x < _imgSize; x++) {
        px[i++] = grayscale.getPixel(x, y).r.toDouble();
      }
    }

    return _runInference(_normalizeCrop(px));
  }

  /// Subtract the crop's own mean, divide by its own std (floored so a
  /// blank cell's sensor noise isn't amplified), rescale, recentre at 0.5,
  /// clip to [0, 1] — same algorithm as `braille_cnn/normalize.py`.
  Float32List _normalizeCrop(
    Float32List px, {
    double stdFloor = 10.0,
    double spanStd = 4.0,
  }) {
    var sum = 0.0;
    for (final v in px) {
      sum += v;
    }
    final mean = sum / px.length;

    var sq = 0.0;
    for (final v in px) {
      final d = v - mean;
      sq += d * d;
    }
    final std = sqrt(sq / px.length);
    final scale = max(std, stdFloor) * spanStd;

    final out = Float32List(px.length);
    for (var i = 0; i < px.length; i++) {
      final v = (px[i] - mean) / scale + 0.5;
      out[i] = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
    }
    return out;
  }

  Future<PredictionResult> _runInference(Float32List inputFloatList) async {
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      inputFloatList,
      [1, 1, _imgSize, _imgSize],
    );

    final runOptions = OrtRunOptions();
    final outputs = await _session!.runAsync(
      runOptions,
      {'input': inputTensor},
    );

    inputTensor.release();
    runOptions.release();

    if (outputs == null || outputs.isEmpty) {
      throw Exception('CNN inference returned empty output');
    }

    final dynamic outValue = outputs[0]?.value;
    for (final o in outputs) {
      o?.release();
    }

    final logits = _flattenLogits(outValue);
    if (logits.isEmpty) {
      throw Exception('CNN output was empty');
    }

    final maxLogit = logits.reduce(max);
    final expValues = logits.map((l) => exp(l - maxLogit)).toList();
    final sumExp = expValues.reduce((a, b) => a + b);
    final probabilities = expValues.map((e) => e / sumExp).toList();

    var maxIndex = 0;
    var maxProb = probabilities[0];
    for (var i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        maxIndex = i;
      }
    }

    final label = _labels[maxIndex];
    final predictedChar =
        (label?.si.trim().isNotEmpty ?? false) ? label!.si : '#$maxIndex';

    return PredictionResult(
      character: predictedChar,
      confidence: maxProb,
      classIndex: maxIndex,
      label: label,
    );
  }

  List<double> _flattenLogits(dynamic value) {
    if (value is List && value.isNotEmpty) {
      final first = value[0];
      if (first is List) {
        return first.map((e) => (e as num).toDouble()).toList();
      }
      return value.map((e) => (e as num).toDouble()).toList();
    }
    return const [];
  }

  void dispose() {
    _session?.release();
    _session = null;
    _isInitialized = false;
    _labels.clear();
  }
}
