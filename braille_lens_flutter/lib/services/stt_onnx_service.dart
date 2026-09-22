import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

/// On-device Sinhala speech recognition: a wav2vec2/MMS-style CTC model
/// exported to ONNX.
///
/// Contract (see `assets/models/`):
/// * input  `input_values`  float32 `[1, samples]` — 16 kHz mono waveform,
///   normalised to zero mean and unit variance, which is what the wav2vec2
///   feature extractor does and what the model was trained on.
/// * output `logits` float32 `[1, frames, vocab]` — vocab is 79 for the
///   Sinhala MMS head; the size is read from the model output rather than
///   hard-coded, so a re-export with a different vocabulary still works.
///
/// The model is not in the repo yet. Every entry point fails soft:
/// [isAvailable] stays false and [transcribe] returns null, so Testing mode
/// can say "speech model missing" instead of crashing.
class SttOnnxService {
  SttOnnxService._();

  /// One session for the app: the model is tens of MB and loading it per
  /// screen would both stall the UI and leak native memory.
  static final SttOnnxService instance = SttOnnxService._();

  /// Sample rate the model expects. The glasses mic and the phone recorder
  /// are both configured to match, so no resampling is needed.
  static const int sampleRate = 16000;

  static const List<String> _modelAssets = [
    'assets/models/sinhala_mms_ctc.onnx',
    'assets/models/sinhala_stt.onnx',
  ];
  static const String _vocabAsset = 'assets/models/vocab.json';

  /// Tokens that never appear in a transcript.
  static const Set<String> _specialTokens = {
    '<pad>',
    '<s>',
    '</s>',
    '<unk>',
  };

  /// wav2vec2 writes word boundaries as this token.
  static const String _wordDelimiter = '|';

  OrtSession? _session;
  List<String> _idToToken = const [];
  int _blankId = 0;
  String? _loadedAsset;
  String? _lastError;
  bool _ready = false;
  Future<bool>? _loading;

  bool get isAvailable => _ready;
  String? get loadedAsset => _loadedAsset;
  String? get lastError => _lastError;
  int get vocabSize => _idToToken.length;

  /// Loads model + vocabulary once. Concurrent callers share one load.
  Future<bool> initialize() {
    if (_ready) return Future.value(true);
    return _loading ??= _load().whenComplete(() => _loading = null);
  }

  Future<bool> _load() async {
    try {
      if (!await _loadVocab()) return false;

      for (final asset in _modelAssets) {
        final bytes = await _tryLoadAsset(asset);
        if (bytes == null) continue;
        OrtEnv.instance.init();
        _session?.release();
        _session = OrtSession.fromBuffer(bytes, OrtSessionOptions());
        _loadedAsset = asset;
        _lastError = null;
        _ready = true;
        debugPrint('[SttOnnx] loaded $asset · vocab ${_idToToken.length}');
        return true;
      }
      _lastError =
          'No speech model bundled — add ${_modelAssets.first} to assets/models/';
      debugPrint('[SttOnnx] ${_lastError!}');
      return false;
    } catch (e) {
      _lastError = e.toString();
      debugPrint('[SttOnnx] load failed: $e');
      return false;
    }
  }

  Future<Uint8List?> _tryLoadAsset(String asset) async {
    try {
      final data = await rootBundle.load(asset);
      return data.buffer.asUint8List();
    } catch (_) {
      return null; // not bundled
    }
  }

  /// `vocab.json` is HuggingFace's `{"token": id}` map; inverted here into an
  /// id-indexed table so decoding is a lookup rather than a search.
  Future<bool> _loadVocab() async {
    try {
      final raw = await rootBundle.loadString(_vocabAsset);
      final decoded = jsonDecode(raw);

      final Map<String, int> tokenToId;
      if (decoded is Map) {
        tokenToId = decoded.map(
          (k, v) => MapEntry(k.toString(), (v as num).toInt()),
        );
      } else if (decoded is List) {
        // Alternative export: a plain id-ordered list of tokens.
        tokenToId = {
          for (var i = 0; i < decoded.length; i++) decoded[i].toString(): i,
        };
      } else {
        _lastError = 'vocab.json is neither an object nor a list';
        return false;
      }
      if (tokenToId.isEmpty) {
        _lastError = 'vocab.json is empty';
        return false;
      }

      final size = tokenToId.values.reduce(math.max) + 1;
      final table = List<String>.filled(size, '');
      tokenToId.forEach((token, id) {
        if (id >= 0 && id < size) table[id] = token;
      });
      _idToToken = table;
      // CTC blank is <pad> in wav2vec2 exports; id 0 if the model names it
      // something else.
      _blankId = tokenToId['<pad>'] ?? tokenToId['<blank>'] ?? 0;
      return true;
    } catch (e) {
      _lastError = 'vocab.json missing or unreadable: $e';
      debugPrint('[SttOnnx] ${_lastError!}');
      return false;
    }
  }

  /// Transcribes 16 kHz mono [samples] in [-1, 1]. Returns null when the
  /// model is unavailable or the audio is too short to decode.
  Future<String?> transcribe(Float32List samples) async {
    if (!_ready && !await initialize()) return null;
    final session = _session;
    if (session == null) return null;

    // Under ~0.2 s there is nothing for the CTC head to align to.
    if (samples.length < sampleRate ~/ 5) {
      debugPrint('[SttOnnx] ${samples.length} samples is too short');
      return null;
    }

    final normalised = _zeroMeanUnitVariance(samples);
    OrtValueTensor? input;
    List<OrtValue?>? outputs;
    OrtRunOptions? runOptions;
    try {
      input = OrtValueTensor.createTensorWithDataList(
        normalised,
        [1, normalised.length],
      );
      runOptions = OrtRunOptions();
      final inputName =
          session.inputNames.isNotEmpty ? session.inputNames.first : 'input_values';
      outputs = await session.runAsync(runOptions, {inputName: input});
      if (outputs == null || outputs.isEmpty) return null;

      final logits = outputs.first?.value;
      final ids = _argmaxPerFrame(logits);
      if (ids == null) {
        debugPrint('[SttOnnx] unexpected logits shape: ${logits.runtimeType}');
        return null;
      }
      return decodeGreedy(ids);
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

  /// wav2vec2's feature extractor normalisation. A waveform left at raw
  /// amplitude decodes to noise, so this is not optional.
  static Float32List _zeroMeanUnitVariance(Float32List samples) {
    if (samples.isEmpty) return samples;
    var sum = 0.0;
    for (final s in samples) {
      sum += s;
    }
    final mean = sum / samples.length;

    var sqSum = 0.0;
    for (final s in samples) {
      final d = s - mean;
      sqSum += d * d;
    }
    // +1e-7 mirrors HuggingFace's epsilon and avoids dividing by zero on
    // digital silence.
    final std = math.sqrt(sqSum / samples.length) + 1e-7;

    final out = Float32List(samples.length);
    for (var i = 0; i < samples.length; i++) {
      out[i] = (samples[i] - mean) / std;
    }
    return out;
  }

  /// `[1, frames, vocab]` logits → the winning token id per frame.
  ///
  /// onnxruntime hands back nested lists, but a flat list shows up too
  /// depending on the export, so both are handled.
  List<int>? _argmaxPerFrame(dynamic logits) {
    if (logits is! List || logits.isEmpty) return null;

    // [1, frames, vocab]
    final batch = logits.first;
    if (batch is List && batch.isNotEmpty && batch.first is List) {
      final ids = <int>[];
      for (final frame in batch) {
        ids.add(_argmax((frame as List).cast<num>()));
      }
      return ids;
    }

    // [1, frames * vocab] flattened.
    if (batch is List && batch.isNotEmpty && batch.first is num) {
      final flat = batch.cast<num>();
      final vocab = _idToToken.length;
      if (vocab == 0 || flat.length % vocab != 0) return null;
      final ids = <int>[];
      for (var f = 0; f < flat.length; f += vocab) {
        ids.add(_argmax(flat.sublist(f, f + vocab)));
      }
      return ids;
    }
    return null;
  }

  static int _argmax(List<num> row) {
    var best = 0;
    var bestValue = double.negativeInfinity;
    for (var i = 0; i < row.length; i++) {
      final v = row[i].toDouble();
      if (v > bestValue) {
        bestValue = v;
        best = i;
      }
    }
    return best;
  }

  /// CTC greedy decode: collapse runs of the same id, drop the blank, map the
  /// rest through the vocabulary, turn `|` into a space and drop the special
  /// tokens. Exposed for unit tests, which need no model.
  @visibleForTesting
  String decodeGreedy(List<int> ids) {
    final buffer = StringBuffer();
    var previous = -1;

    for (final id in ids) {
      // Repeats of a token within one run are the same emission.
      if (id == previous) continue;
      previous = id;
      if (id == _blankId) continue;
      if (id < 0 || id >= _idToToken.length) continue;

      final token = _idToToken[id];
      if (token.isEmpty || _specialTokens.contains(token)) continue;
      buffer.write(token == _wordDelimiter ? ' ' : token);
    }

    return buffer.toString().replaceAll(RegExp(r'\s+'), ' ').trim();
  }

  /// Test seam: lets the decoder be exercised without an ONNX model.
  @visibleForTesting
  void loadVocabForTest(List<String> idToToken, {int blankId = 0}) {
    _idToToken = idToToken;
    _blankId = blankId;
  }

  void dispose() {
    _session?.release();
    _session = null;
    _ready = false;
  }
}
