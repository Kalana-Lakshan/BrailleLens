/// Spoken-letter classification on top of a CTC speech model: the Dart port
/// of Cell 11 in `STT_model_training_small.ipynb`.
///
/// The model was not trained as a classifier — it is a cut-down MMS that
/// emits per-frame scores over 79 Sinhala tokens. A letter is picked by
/// asking, for each allowed letter, "how likely is it that this clip is
/// exactly that one token?" (the CTC likelihood of a one-token label) and
/// taking the best. The clip preparation below must match the notebook step
/// for step, or the scores stop meaning what they meant in training.
library;

import 'dart:math' as math;
import 'dart:typed_data';

const int _sampleRate = 16000;

/// `librosa.effects.trim` defaults.
const int _trimFrameLength = 2048;
const int _trimHopLength = 512;

/// Silence threshold, in dB below the loudest frame.
const double trimTopDb = 30;

/// Silence added to each end after trimming (0.2 s).
const int edgePaddingSamples = _sampleRate ~/ 5;

/// [samples] with leading and trailing silence removed, as
/// `librosa.effects.trim(y, top_db=30)` does: frame RMS (centred, zero
/// padded), in dB relative to the loudest frame, kept from the first frame
/// above the threshold to the end of the last.
Float32List trimSilence(Float32List samples, {double topDb = trimTopDb}) {
  final n = samples.length;
  if (n == 0) return samples;

  // Prefix sums of squares over the zero-padded signal, so each frame's
  // energy is one subtraction.
  const half = _trimFrameLength ~/ 2;
  final prefix = Float64List(n + 1);
  for (var i = 0; i < n; i++) {
    prefix[i + 1] = prefix[i] + samples[i] * samples[i];
  }
  double energy(int from, int to) {
    // [from, to) in padded coordinates → clamp to the real signal.
    final a = (from - half).clamp(0, n);
    final b = (to - half).clamp(0, n);
    return b > a ? prefix[b] - prefix[a] : 0.0;
  }

  final frames = 1 + n ~/ _trimHopLength;
  final rms = Float64List(frames);
  var peak = 0.0;
  for (var f = 0; f < frames; f++) {
    final start = f * _trimHopLength;
    rms[f] = math.sqrt(energy(start, start + _trimFrameLength) /
        _trimFrameLength);
    if (rms[f] > peak) peak = rms[f];
  }

  // amplitude_to_db(ref=max, amin=1e-5): 20·log10 of the clamped ratio.
  const amin = 1e-5;
  final refDb = 20 * _log10(math.max(amin, peak));
  int? first;
  var last = -1;
  for (var f = 0; f < frames; f++) {
    final db = 20 * _log10(math.max(amin, rms[f])) - refDb;
    if (db > -topDb) {
      first ??= f;
      last = f;
    }
  }
  if (first == null) return Float32List(0);

  final start = first * _trimHopLength;
  final end = math.min(n, (last + 1) * _trimHopLength);
  return Float32List.sublistView(samples, start, end);
}

double _log10(double x) => math.log(x) / math.ln10;

/// The waveform the model is fed: trimmed, 0.2 s of silence on each side,
/// then zero mean and unit variance. Empty when [samples] holds no sound.
Float32List prepareLetterClip(Float32List samples) {
  final trimmed = trimSilence(samples);
  if (trimmed.isEmpty) return trimmed;

  final out = Float32List(trimmed.length + 2 * edgePaddingSamples);
  out.setAll(edgePaddingSamples, trimmed);

  var sum = 0.0;
  for (final s in out) {
    sum += s;
  }
  final mean = sum / out.length;
  var sq = 0.0;
  for (final s in out) {
    final d = s - mean;
    sq += d * d;
  }
  final std = math.sqrt(sq / out.length + 1e-7);
  for (var i = 0; i < out.length; i++) {
    out[i] = (out[i] - mean) / std;
  }
  return out;
}

double _logAddExp(double a, double b) {
  if (a == double.negativeInfinity) return b;
  if (b == double.negativeInfinity) return a;
  final m = math.max(a, b);
  return m + math.log(math.exp(a - m) + math.exp(b - m));
}

/// CTC log-likelihood that the clip is exactly one token: blanks, then the
/// token for one or more frames, then blanks.
///
/// [blank] and [token] are that column's per-frame log-probabilities.
double singleTokenLogLikelihood(List<double> blank, List<double> token) {
  final frames = blank.length;
  if (frames == 0) return double.negativeInfinity;

  // suffix[t]: log P(frames t.. are all blank).
  final suffix = Float64List(frames + 1);
  for (var t = frames - 1; t >= 0; t--) {
    suffix[t] = suffix[t + 1] + blank[t];
  }

  var prefixBlank = 0.0; // log P(frames 0..t-1 all blank)
  var alpha = double.negativeInfinity; // paths whose frame t is the token
  var total = double.negativeInfinity;
  for (var t = 0; t < frames; t++) {
    alpha = token[t] + _logAddExp(alpha, prefixBlank);
    total = _logAddExp(total, alpha + suffix[t + 1]);
    prefixBlank += blank[t];
  }
  return total;
}

/// One letter and how likely the classifier thinks it is.
class LetterScore {
  final String letter;

  /// Share of the probability among the allowed letters, 0–1.
  final double confidence;

  const LetterScore(this.letter, this.confidence);

  @override
  String toString() => '$letter ${(confidence * 100).toStringAsFixed(0)}%';
}

/// The classifier's reading of one clip.
class LetterGuess {
  /// Every allowed letter, most likely first.
  final List<LetterScore> ranked;

  /// False when "nothing was said" explains the clip better than any letter
  /// does — the all-blank path outscores the best letter. Such a clip is not
  /// an answer, and must not be marked wrong.
  final bool heard;

  const LetterGuess({required this.ranked, required this.heard});

  LetterScore get best => ranked.first;
}

/// Ranks [letters] for a clip from the model's raw `[frames][vocab]`
/// [logits]. [tokenIds] gives each letter's column, [blankId] the CTC blank.
LetterGuess classifyLetter({
  required List<List<double>> logits,
  required List<String> letters,
  required List<int> tokenIds,
  required int blankId,
}) {
  final frames = logits.length;
  final blank = Float64List(frames);
  final columns = [for (final _ in tokenIds) Float64List(frames)];

  for (var t = 0; t < frames; t++) {
    final row = logits[t];
    // log-softmax needs only the row's log-sum-exp.
    var max = double.negativeInfinity;
    for (final v in row) {
      if (v > max) max = v;
    }
    var sum = 0.0;
    for (final v in row) {
      sum += math.exp(v - max);
    }
    final lse = max + math.log(sum);
    blank[t] = row[blankId] - lse;
    for (var i = 0; i < tokenIds.length; i++) {
      columns[i][t] = row[tokenIds[i]] - lse;
    }
  }

  final scores = [
    for (final column in columns) singleTokenLogLikelihood(blank, column),
  ];
  var top = double.negativeInfinity;
  for (final s in scores) {
    if (s > top) top = s;
  }
  var norm = 0.0;
  for (final s in scores) {
    norm += math.exp(s - top);
  }

  final ranked = [
    for (var i = 0; i < letters.length; i++)
      LetterScore(letters[i], math.exp(scores[i] - top) / norm),
  ]..sort((a, b) => b.confidence.compareTo(a.confidence));

  var allBlank = 0.0;
  for (final b in blank) {
    allBlank += b;
  }
  return LetterGuess(ranked: ranked, heard: frames > 0 && top > allBlank);
}
