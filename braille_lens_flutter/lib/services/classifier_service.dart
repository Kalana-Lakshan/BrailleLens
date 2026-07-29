import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:image/image.dart' as img;

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

class ClassifierService {
  OrtSession? _session;
  List<String> _labels = [];
  bool _isInitialized = false;

  bool get isInitialized => _isInitialized;

  Future<void> initialize() async {
    if (_isInitialized) return;

    // Initialize OrtEnv
    OrtEnv.instance.init();

    // Load model asset bytes
    final modelBytes = await rootBundle.load('assets/models/braille_model.onnx');
    final sessionOptions = OrtSessionOptions();
    _session = OrtSession.fromBuffer(
      modelBytes.buffer.asUint8List(),
      sessionOptions,
    );

    // Load labels
    final labelsRaw = await rootBundle.loadString('assets/labels.txt');
    _labels = labelsRaw
        .split('\n')
        .map((e) => e.trim())
        .where((e) => e.isNotEmpty)
        .toList();

    _isInitialized = true;
  }

  Future<PredictionResult> predict(Uint8List imageBytes) async {
    if (!_isInitialized || _session == null) {
      await initialize();
    }

    // 1. Decode Image
    final originalImage = img.decodeImage(imageBytes);
    if (originalImage == null) {
      throw Exception('Failed to decode image');
    }

    // 2. Convert to Grayscale & Resize to 28x28
    final resized = img.copyResize(originalImage, width: 28, height: 28);
    final grayscale = img.grayscale(resized);

    // 3. Prepare Float32 Tensor Data [1, 1, 28, 28]
    // Normalized to (pixel / 255.0 - 0.5) / 0.5
    final inputFloatList = Float32List(1 * 1 * 28 * 28);
    int index = 0;

    for (int y = 0; y < 28; y++) {
      for (int x = 0; x < 28; x++) {
        final pixel = grayscale.getPixel(x, y);
        // Extract red/luminance channel
        final r = pixel.r / 255.0;
        final normalized = (r - 0.5) / 0.5;
        inputFloatList[index++] = normalized;
      }
    }

    // 4. Create OrtValueTensor
    final shape = [1, 1, 28, 28];
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      inputFloatList,
      shape,
    );

    final runOptions = OrtRunOptions();
    final outputs = await _session!.runAsync(
      runOptions,
      {'input': inputTensor},
    );

    inputTensor.release();
    runOptions.release();

    if (outputs == null || outputs.isEmpty) {
      throw Exception('Model inference returned empty output');
    }

    // Extract output logits tensor
    final outputOrtValue = outputs[0];
    final outputValues = (outputOrtValue?.value as List);
    final rawLogits = (outputValues[0] as List);
    final logits = rawLogits.map((e) => (e as num).toDouble()).toList();

    outputOrtValue?.release();

    // 5. Softmax to calculate confidence scores
    double maxLogit = logits.reduce(max);
    List<double> expValues = logits.map((l) => exp(l - maxLogit)).toList();
    double sumExp = expValues.reduce((a, b) => a + b);
    List<double> probabilities = expValues.map((e) => e / sumExp).toList();

    int maxIndex = 0;
    double maxProb = probabilities[0];
    for (int i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        maxIndex = i;
      }
    }

    String predictedChar = (maxIndex < _labels.length) ? _labels[maxIndex] : 'a';

    return PredictionResult(
      character: predictedChar,
      confidence: maxProb,
      classIndex: maxIndex,
    );
  }

  void dispose() {
    _session?.release();
    _session = null;
    _isInitialized = false;
  }
}
