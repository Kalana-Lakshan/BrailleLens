import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

import '../utils/letter_ctc.dart';

/// The classifier's answer for one recording, with how long it took.
class SpokenLetter {
  final LetterGuess guess;
  final Duration elapsed;

  const SpokenLetter(this.guess, this.elapsed);

  String get letter => guess.best.letter;
  double get confidence => guess.best.confidence;
  bool get heard => guess.heard;
}

/// On-device recognition of a spoken Sinhala letter.
///
/// Model: `sinhala_mms_small_int8.onnx` (~51 MB) — MMS-300M cut down to its
/// first transformer layers, fine-tuned on OpenSLR 30 Sinhala and int8
/// quantised (STT_model_training_small.ipynb).
/// * input  `input_values` float32 `[1, samples]`, 16 kHz mono
/// * output `logits` float32 `[1, frames, 79]`, CTC over Sinhala tokens
///
/// It is used as a closed-set classifier, not a transcriber: `letters.json`
/// lists the 57 letters that may be answered and their token ids, and
/// [classify] returns the most likely of them (see `utils/letter_ctc.dart`).
/// The learner therefore says the letter itself ("ක"), not its name.
///
/// Every entry point fails soft: with the model or `letters.json` missing,
/// [isAvailable] stays false and [classify] returns null, so Testing Mode
/// can read the answer out instead of crashing.
class SttOnnxService {
  SttOnnxService._();

  /// One session for the app: loading tens of MB per screen would stall the
  /// UI and leak native memory.
  static final SttOnnxService instance = SttOnnxService._();

  /// Sample rate the model expects. The glasses mic and the phone recorder
  /// are both configured to match, so no resampling is needed.
  static const int sampleRate = 16000;

  static const String _modelAsset = 'assets/models/sinhala_mms_small_int8.onnx';
  static const String _lettersAsset = 'assets/models/letters.json';

  OrtSession? _session;
  List<String> _letters = const [];
  List<int> _tokenIds = const [];
  int _blankId = 0;
  String? _lastError;
  bool _ready = false;
  Future<bool>? _loading;

  bool get isAvailable => _ready;
  String? get loadedAsset => _ready ? _modelAsset : null;
  String? get lastError => _lastError;

  /// The letters the model can answer with, in `letters.json` order.
  List<String> get letters => _letters;

  /// Whether [letter] is one the model can recognise. Signs and indicators
  /// (ං, ඃ, vowel signs) are not, so they cannot be quizzed by voice.
  bool canRecognise(String letter) => _letters.contains(letter.trim());

  /// Loads the model and letter table once. Concurrent callers share a load.
  Future<bool> initialize() {
    if (_ready) return Future.value(true);
    return _loading ??= _load().whenComplete(() => _loading = null);
  }

  Future<bool> _load() async {
    try {
      if (!await _loadLetters()) return false;

      final ByteData data;
      try {
        data = await rootBundle.load(_modelAsset);
      } catch (_) {
        _lastError = 'No speech model bundled — add $_modelAsset';
        debugPrint('[SttOnnx] ${_lastError!}');
        return false;
      }
      OrtEnv.instance.init();
      _session?.release();
      _session = OrtSession.fromBuffer(
        data.buffer.asUint8List(),
        OrtSessionOptions(),
      );
      _lastError = null;
      _ready = true;
      debugPrint('[SttOnnx] loaded $_modelAsset · ${_letters.length} letters');
      return true;
    } catch (e) {
      _lastError = e.toString();
      debugPrint('[SttOnnx] load failed: $e');
      return false;
    }
  }

  /// `letters.json`: `{"letters": [...], "token_ids": [...], "blank_id": n}`,
  /// written by the training notebook alongside the model.
  Future<bool> _loadLetters() async {
    try {
      final decoded = jsonDecode(await rootBundle.loadString(_lettersAsset));
      final letters = (decoded['letters'] as List).cast<String>();
      final ids = (decoded['token_ids'] as List)
          .map((v) => (v as num).toInt())
          .toList();
      if (letters.isEmpty || letters.length != ids.length) {
        _lastError = 'letters.json: ${letters.length} letters but '
            '${ids.length} token ids';
        debugPrint('[SttOnnx] ${_lastError!}');
        return false;
      }
      _letters = List.unmodifiable(letters);
      _tokenIds = List.unmodifiable(ids);
      _blankId = (decoded['blank_id'] as num).toInt();
      return true;
    } catch (e) {
      _lastError = 'letters.json missing or unreadable: $e';
      debugPrint('[SttOnnx] ${_lastError!}');
      return false;
    }
  }

  /// Classifies 16 kHz mono [samples] in [-1, 1] as one of [letters].
  /// Null when the model is unavailable or the clip holds no usable sound.
  Future<SpokenLetter?> classify(Float32List samples) async {
    if (!_ready && !await initialize()) return null;
    final session = _session;
    if (session == null) return null;

    // Feed check: [-1, 1] floats with real speech peaking around 0.05-0.9. A
    // peak near 0 is silence; above 1 means the PCM was not scaled.
    var lo = 0.0, hi = 0.0;
    for (final s in samples) {
      if (s < lo) lo = s;
      if (s > hi) hi = s;
    }
    final clip = prepareLetterClip(samples);
    debugPrint('[SttOnnx] input ${samples.length} samples '
        'min=${lo.toStringAsFixed(3)} max=${hi.toStringAsFixed(3)} '
        '-> ${clip.length} after trim+pad');

    // Under ~50 ms of sound there is nothing for the CTC head to align to.
    if (clip.length < 2 * edgePaddingSamples + sampleRate ~/ 20) {
      debugPrint('[SttOnnx] clip too short to classify');
      return null;
    }

    OrtValueTensor? input;
    List<OrtValue?>? outputs;
    OrtRunOptions? runOptions;
    try {
      input = OrtValueTensor.createTensorWithDataList(clip, [1, clip.length]);
      runOptions = OrtRunOptions();
      final inputName = session.inputNames.isNotEmpty
          ? session.inputNames.first
          : 'input_values';
      final watch = Stopwatch()..start();
      // runAsync executes the model off the UI isolate.
      outputs = await session.runAsync(runOptions, {inputName: input});
      if (outputs == null || outputs.isEmpty) return null;

      final frames = _frames(outputs.first?.value);
      if (frames == null) {
        debugPrint('[SttOnnx] unexpected logits shape');
        return null;
      }
      final guess = classifyLetter(
        logits: frames,
        letters: _letters,
        tokenIds: _tokenIds,
        blankId: _blankId,
      );
      watch.stop();
      debugPrint('[SttOnnx] logits [1, ${frames.length}, '
          '${frames.isEmpty ? 0 : frames.first.length}] in '
          '${watch.elapsedMilliseconds} ms -> '
          '${guess.ranked.take(3).join(', ')} heard=${guess.heard}');
      return SpokenLetter(guess, watch.elapsed);
    } catch (e) {
      _lastError = e.toString();
      debugPrint('[SttOnnx] inference failed: $e');
      return null;
    } finally {
      input?.release();
      runOptions?.release();
      if (outputs != null) {
        for (final o in outputs) {
          o?.release();
        }
      }
    }
  }

  /// `[1, frames, vocab]` nested lists → `[frames][vocab]` doubles.
  static List<List<double>>? _frames(dynamic logits) {
    if (logits is! List || logits.isEmpty) return null;
    final batch = logits.first;
    if (batch is! List || batch.isEmpty || batch.first is! List) return null;
    return [
      for (final frame in batch)
        [for (final v in frame as List) (v as num).toDouble()],
    ];
  }

  void dispose() {
    _session?.release();
    _session = null;
    _ready = false;
  }
}
