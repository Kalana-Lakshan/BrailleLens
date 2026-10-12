import 'package:braille_lens_flutter/models/braille_cell.dart';
import 'package:braille_lens_flutter/utils/answer_match.dart';
import 'package:braille_lens_flutter/utils/app_prompts.dart';
import 'package:braille_lens_flutter/utils/english_braille.dart';
import 'package:braille_lens_flutter/utils/sinhala_prompts.dart';
import 'package:flutter_test/flutter_test.dart';

BrailleCell _cell(int code, String char) => BrailleCell(
    id: 0, x0: 0, y0: 0, x1: 1, y1: 1, char: char, code: code);

void main() {
  group('English Grade-1 letters by cell code', () {
    test('a-z map from their dot patterns', () {
      expect(englishBrailleLetter(1), 'a'); // dot 1
      expect(englishBrailleLetter(3), 'b'); // dots 1, 2
      expect(englishBrailleLetter(25), 'd'); // dots 1, 4, 5
      expect(englishBrailleLetter(58), 'w'); // dots 2, 4, 5, 6
      expect(englishBrailleLetter(53), 'z'); // dots 1, 3, 5, 6
    });

    test('blank, capital and number signs are not letters', () {
      expect(englishBrailleLetter(0), isNull);
      expect(englishBrailleLetter(32), isNull);
      expect(englishBrailleLetter(60), isNull);
    });
  });

  group('per-language letter under a cell', () {
    // Code 3 is "අ" on the Sinhala chart and "b" in English.
    final cell = _cell(3, 'අ');

    test('Sinhala reads the Sinhala glyph', () {
      expect(AppLanguage.sinhala.prompts.letterOf(cell), 'අ');
    });

    test('English reads the same code as an English letter', () {
      final p = AppLanguage.english.prompts;
      expect(p.letterOf(cell), 'b');
      expect(p.letterName('b'), 'B');
      expect(p.cellLabel(cell), 'B');
      expect(p.letterOf(_cell(60, 'x')), isNull);
    });
  });

  test('English prompts', () {
    final p = AppLanguage.english.prompts;
    expect(p.dots('124'), 'Dots 1, 2, 4.');
    expect(p.dots('1'), 'Dot 1.');
    expect(p.dots(''), '');
    expect(p.direction(Offset.zero, const Offset(40, 5), 20),
        EnglishAppPrompts.moveRight);
    expect(p.direction(Offset.zero, const Offset(4, 4), 20), isNull);
  });

  test('Sinhala prompts are unchanged', () {
    final p = AppLanguage.sinhala.prompts;
    expect(p.prescan, SinhalaPrompts.prescan);
    expect(p.homeInstructions, startsWith(SinhalaPrompts.homeInstructions));
  });

  test('language voice commands', () {
    expect(parseVoiceLanguageCommand('english please'), 'english');
    expect(parseVoiceLanguageCommand('Sinhala'), 'sinhala');
    expect(parseVoiceLanguageCommand('sinhalese'), 'sinhala');
    expect(parseVoiceLanguageCommand('learning'), isNull);
    expect(parseVoiceLanguageCommand(null), isNull);
  });

  test('English spoken answers accept the letter or its name', () {
    expect(spokenAnswerMatches('bee', 'b'), isTrue);
    expect(spokenAnswerMatches('b', 'b'), isTrue);
    expect(spokenAnswerMatches('see', 'b'), isFalse);
    expect(spokenAnswerMatches('el', 'l'), isTrue);
    expect(spokenAnswerMatches('letter ell', 'l'), isTrue);
    expect(spokenAnswerMatches('double you', 'w'), isTrue);
    expect(spokenAnswerMatches('lala lala', 'l'), isFalse);
  });
}
