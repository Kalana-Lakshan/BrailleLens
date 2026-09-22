import 'dart:typed_data';

import 'package:braille_lens_flutter/services/stt_onnx_service.dart';
import 'package:braille_lens_flutter/services/voice_capture_service.dart';
import 'package:braille_lens_flutter/utils/sinhala_phonetics.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  // Index == token id, mirroring an inverted vocab.json.
  const vocab = ['<pad>', '<s>', '</s>', '<unk>', '|', 'ක', 'ය', 'න', '්', 'ම'];
  const blank = 0;
  const pad = 0, s = 1, es = 2, unk = 3, bar = 4;
  const ka = 5, ya = 6, na = 7, hal = 8, ma = 9;

  group('CTC greedy decode', () {
    final stt = SttOnnxService.instance..loadVocabForTest(vocab, blankId: blank);

    test('collapses repeated frames into one emission', () {
      expect(stt.decodeGreedy([ka, ka, ka, ya, ya, na]), 'කයන');
    });

    test('a blank between repeats keeps both letters', () {
      // Without the blank this would collapse to a single "ක".
      expect(stt.decodeGreedy([ka, pad, ka]), 'කක');
    });

    test('drops blanks and special tokens', () {
      expect(
        stt.decodeGreedy([pad, s, ka, pad, unk, ya, es, pad]),
        'කය',
      );
    });

    test('word delimiter becomes a space, edges trimmed', () {
      expect(stt.decodeGreedy([bar, ka, bar, ma, bar]), 'ක ම');
    });

    test('empty and all-blank input decode to an empty string', () {
      expect(stt.decodeGreedy(const []), '');
      expect(stt.decodeGreedy([pad, pad, pad]), '');
    });

    test('out-of-range ids are ignored rather than throwing', () {
      expect(stt.decodeGreedy([ka, 99, -1, ya]), 'කය');
    });

    test('decodes a full letter name', () {
      // ක ය න ් න — the blank separates the two න emissions.
      expect(
        stt.decodeGreedy([ka, ka, ya, ya, na, hal, pad, na, na]),
        'කයන්න',
      );
    });
  });

  group('pcm16ToFloat32', () {
    test('maps signed 16-bit samples into [-1, 1]', () {
      final pcm = Uint8List.fromList([
        0x00, 0x00, // 0
        0xFF, 0x7F, // 32767 → ~ +1
        0x00, 0x80, // -32768 → -1
      ]);
      final out = VoiceCaptureService.pcm16ToFloat32(pcm);
      expect(out.length, 3);
      expect(out[0], 0.0);
      expect(out[1], closeTo(1.0, 0.001));
      expect(out[2], -1.0);
    });

    test('a trailing odd byte cannot produce a partial sample', () {
      expect(VoiceCaptureService.pcm16ToFloat32(Uint8List(3)).length, 1);
    });
  });

  group('Sinhala letter names', () {
    test('consonants take the යන්න ending', () {
      expect(sinhalaLetterName('ක'), 'කයන්න');
      expect(sinhalaLetterName('ම'), 'මයන්න');
    });

    test('irregular names are used verbatim', () {
      expect(sinhalaLetterName('ං'), 'බින්දුව');
    });

    test('blank input has no name to speak', () {
      expect(sinhalaLetterName(' '), '');
      expect(sinhalaLetterName(''), '');
    });
  });

  group('spoken answer matching', () {
    test('accepts the bare letter and its name', () {
      expect(sinhalaAnswerMatches('ක', 'ක'), isTrue);
      expect(sinhalaAnswerMatches('කයන්න', 'ක'), isTrue);
    });

    test('accepts the answer inside a sentence', () {
      expect(sinhalaAnswerMatches('මේක කයන්න', 'ක'), isTrue);
    });

    test('accepts a clipped ending, which CTC output often has', () {
      expect(sinhalaAnswerMatches('කයන', 'ක'), isTrue);
    });

    test('rejects a different letter and empty input', () {
      expect(sinhalaAnswerMatches('මයන්න', 'ක'), isFalse);
      expect(sinhalaAnswerMatches('', 'ක'), isFalse);
      expect(sinhalaAnswerMatches(null, 'ක'), isFalse);
    });
  });
}
