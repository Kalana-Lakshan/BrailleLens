import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:image/image.dart' as img;

import '../config/app_config.dart';

class PredictionResult {
  final String character;
  final double confidence;
  final int classIndex;

  PredictionResult({
    required this.character,
    required this.confidence,
    required this.classIndex,
  });
}

/// 26-class Braille letter CNN (`braille_model.onnx`, 28×28 grayscale).
class ClassifierService {
  OrtSession? _session;
  List<String> _labels = [];
  bool _isInitialized = false;
  String? _loadedAsset;
  String? _lastError;

  bool get isInitialized => _isInitialized;
  String? get loadedAsset => _loadedAsset;
  String? get lastError => _lastError;

  Future<bool> initialize() async {
    if (_isInitialized) return true;

    try {
      OrtEnv.instance.init();
      final modelBytes = await rootBundle.load(AppConfig.brailleCnnAsset);
      final sessionOptions = OrtSessionOptions();
      _session = OrtSession.fromBuffer(
        modelBytes.buffer.asUint8List(),
        sessionOptions,
      );

      final labelsRaw = await rootBundle.loadString('assets/labels.txt');
      _labels = labelsRaw
          .split('\n')
          .map((e) => e.trim())
          .where((e) => e.isNotEmpty)
          .toList();

      _loadedAsset = AppConfig.brailleCnnAsset;
      _lastError = null;
      _isInitialized = true;
      debugPrint('[Classifier] loaded $_loadedAsset (${_labels.length} labels)');
      return true;
    } catch (e) {
      _lastError = e.toString();
      _isInitialized = false;
      debugPrint('[Classifier] failed to load ${AppConfig.brailleCnnAsset}: $e');
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

  /// Classify a single cell crop (28×28 grayscale inside the model).
  Future<PredictionResult> predictCrop(img.Image crop) async {
    if (!_isInitialized || _session == null) {
      final ok = await initialize();
      if (!ok || _session == null) {
        throw Exception(_lastError ?? 'CNN model not loaded');
      }
    }

    final resized = img.copyResize(crop, width: 28, height: 28);
    final grayscale = img.grayscale(resized);

    final inputFloatList = Float32List(1 * 1 * 28 * 28);
    var index = 0;
    for (var y = 0; y < 28; y++) {
      for (var x = 0; x < 28; x++) {
        final pixel = grayscale.getPixel(x, y);
        final r = pixel.r / 255.0;
        inputFloatList[index++] = (r - 0.5) / 0.5;
      }
    }

    return _runInference(inputFloatList);
  }

  Future<PredictionResult> _runInference(Float32List inputFloatList) async {
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      inputFloatList,
      [1, 1, 28, 28],
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

    dynamic outValue;
    if (outputs is Map) {
      outValue = outputs['output']?.value ?? outputs.values.first?.value;
      for (final o in outputs.values) {
        o?.release();
      }
    } else {
      outValue = outputs[0]?.value ?? outputs[0];
      for (final o in outputs) {
        o?.release();
      }
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

    final predictedChar =
        (maxIndex < _labels.length) ? _labels[maxIndex] : '?';

    return PredictionResult(
      character: predictedChar,
      confidence: maxProb,
      classIndex: maxIndex,
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
  }
}
