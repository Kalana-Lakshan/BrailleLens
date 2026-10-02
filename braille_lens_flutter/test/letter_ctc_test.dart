import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:braille_lens_flutter/utils/letter_ctc.dart';

/// Expected values come from the notebook's own Python (librosa 0.10
/// `effects.trim`, and Cell 11's `single_letter_score`) on the same inputs,
/// so these pin the Dart port to what the model was evaluated with.
void main() {
  Float32List burst() {
    // Quiet hum with a 220 Hz burst from sample 4000 to 9000.
    final y = Float32List(16000);
    for (var i = 0; i < y.length; i++) {
      y[i] = (i >= 4000 && i < 9000)
          ? 0.3 * math.sin(2 * math.pi * 220 * i / 16000)
          : 0.001 * math.sin(2 * math.pi * 50 * i / 16000);
    }
    return y;
  }

  group('trimSilence matches librosa.effects.trim', () {
    test('cuts to the frames around the burst', () {
      // librosa: index [3072, 10240]
      final trimmed = trimSilence(burst());
      expect(trimmed.length, 7168);
      expect(trimmed.first, closeTo(burst()[3072], 1e-9));
    });

    test('sound from the first sample keeps the start', () {
      final y = Float32List(16000);
      for (var i = 0; i < 3000; i++) {
        y[i] = 0.2 * math.sin(2 * math.pi * 300 * i / 16000);
      }
      expect(trimSilence(y).length, 4096); // librosa: [0, 4096]
    });

    test('digital silence is left whole, as librosa leaves it', () {
      expect(trimSilence(Float32List(8000)).length, 8000);
    });
  });

  test('prepareLetterClip pads 0.2 s each side and normalises', () {
    final clip = prepareLetterClip(burst());
    expect(clip.length, 13568);
    expect(clip[0], closeTo(-0.0020692977122962475, 1e-6));
    expect(clip[3200 + 10], closeTo(-0.007772194221615791, 1e-6));
    expect(clip[3200 + 700], closeTo(-0.00962091889232397, 1e-6));
  });

  group('classifyLetter matches the notebook', () {
    final logits = [
      for (var t = 0; t < 6; t++)
        [for (var v = 0; v < 5; v++) math.sin(t * 1.3 + v * 0.7) * 2],
    ];
    final guess = classifyLetter(
      logits: logits,
      letters: const ['අ', 'ක', 'ම'],
      tokenIds: const [1, 2, 3],
      blankId: 0,
    );

    test('ranks letters by single-token CTC likelihood', () {
      expect(guess.best.letter, 'ම');
      expect(guess.best.confidence, closeTo(0.7055807025789284, 1e-9));
      expect(guess.ranked[1].letter, 'ක');
      expect(guess.ranked[1].confidence, closeTo(0.22946218333960902, 1e-9));
      expect(guess.ranked[2].confidence, closeTo(0.06495711408146276, 1e-9));
    });

    test('a letter that beats the all-blank path counts as heard', () {
      // best letter -7.76 vs all blank -13.83
      expect(guess.heard, isTrue);
    });
  });

  test('single-token likelihood equals brute force over all paths', () {
    // 3 frames: valid paths for token k are b*k+b* → enumerate by hand.
    const blank = [-0.2, -1.5, -0.7];
    const token = [-2.0, -0.4, -1.1];
    var total = 0.0;
    for (var start = 0; start < 3; start++) {
      for (var end = start; end < 3; end++) {
        var lp = 0.0;
        for (var t = 0; t < 3; t++) {
          lp += (t >= start && t <= end) ? token[t] : blank[t];
        }
        total += math.exp(lp);
      }
    }
    expect(singleTokenLogLikelihood(blank, token),
        closeTo(math.log(total), 1e-12));
  });

  test('silence-like output is not counted as an answer', () {
    // Blank dominates every frame.
    final logits = [
      for (var t = 0; t < 20; t++) [8.0, -3.0, -3.0],
    ];
    final guess = classifyLetter(
      logits: logits,
      letters: const ['අ', 'ක'],
      tokenIds: const [1, 2],
      blankId: 0,
    );
    expect(guess.heard, isFalse);
  });
}
