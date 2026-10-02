import 'dart:typed_data';

import 'package:braille_lens_flutter/services/voice_capture_service.dart';
import 'package:braille_lens_flutter/utils/sinhala_phonetics.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
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

    test('ignores whitespace, punctuation and zero-width characters', () {
      expect(sinhalaAnswerMatches('  කයන්න.  ', 'ක'), isTrue);
      expect(sinhalaAnswerMatches('ක​යන්න', 'ක'), isTrue);
      expect(sinhalaAnswerMatches('ශ්‍ෂයන්න', 'ශ්‍ෂ'), isTrue);
    });

    test('joins a name split into syllables', () {
      expect(sinhalaAnswerMatches('ක යන්න', 'ක'), isTrue);
    });

    test('composes split vowel signs before comparing', () {
      // ඔ's name spelt with a decomposed vs precomposed o-sign must agree.
      expect(sinhalaAnswerMatches('කො', 'කො'), isTrue);
      expect(sinhalaAnswerMatches('කෝ', 'කෝ'), isTrue);
    });

    test('accepts homophones the microphone cannot tell apart', () {
      expect(sinhalaAnswerMatches('නයන්න', 'ණ'), isTrue);
      expect(sinhalaAnswerMatches('කයන්න', 'ඛ'), isTrue);
      expect(sinhalaAnswerMatches('සයන්න', 'ෂ'), isTrue);
    });

    // Real output of sinhala_mms_ctc_quantized.onnx for spoken letter names.
    test('accepts the MMS model\'s vowel-sign drift on letter names', () {
      expect(sinhalaAnswerMatches('කියන්නෙක්', 'ක'), isTrue);
      expect(sinhalaAnswerMatches('හොයන්නෙක්', 'හ'), isTrue);
      expect(sinhalaAnswerMatches('ඊයෙන්න', 'ඊ'), isTrue);
      expect(sinhalaAnswerMatches('අයෙන්නෙන්', 'අ'), isTrue);
      expect(sinhalaAnswerMatches('බ යැන්නෙක්', 'බ'), isTrue);
      expect(sinhalaAnswerMatches('රොයන්න', 'ර'), isTrue);
      expect(sinhalaAnswerMatches('තියෙන්ව', 'ත'), isTrue);
    });

    test('skeleton matching still needs the right letter', () {
      expect(sinhalaAnswerMatches('කියන්නෙක්', 'ග'), isFalse);
      expect(sinhalaAnswerMatches('ආයෙන්න', 'අ'), isFalse);
      // A word merely starting with the letter is not its name.
      expect(sinhalaAnswerMatches('කතාව', 'ක'), isFalse);
    });

    test('still rejects letters that sound different', () {
      expect(sinhalaAnswerMatches('ගයන්න', 'ක'), isFalse);
      expect(sinhalaAnswerMatches('ගයන්න', 'ඟ'), isFalse);
    });
  });

  group('quizzable letters', () {
    test('Sinhala letters can be named, indicators and blanks cannot', () {
      expect(isNameableSinhala('ක'), isTrue);
      expect(isNameableSinhala('ශ්‍ෂ'), isTrue);
      expect(isNameableSinhala('[IND-A]'), isFalse);
      expect(isNameableSinhala(' '), isFalse);
      expect(isNameableSinhala('space'), isFalse);
    });
  });

  group('letters that sound alike', () {
    test('the same letter matches', () {
      expect(sinhalaLettersSoundAlike('ක', 'ක'), isTrue);
    });

    test('homophones match, in either direction', () {
      expect(sinhalaLettersSoundAlike('න', 'ණ'), isTrue);
      expect(sinhalaLettersSoundAlike('ඛ', 'ක'), isTrue);
      expect(sinhalaLettersSoundAlike('ෂ', 'ශ'), isTrue);
    });

    test('different sounds do not', () {
      expect(sinhalaLettersSoundAlike('ක', 'ග'), isFalse);
      expect(sinhalaLettersSoundAlike('අ', 'ආ'), isFalse);
      expect(sinhalaLettersSoundAlike('', 'ක'), isFalse);
    });
  });
}
