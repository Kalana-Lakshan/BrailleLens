import 'package:flutter_test/flutter_test.dart';
import 'package:braille_lens_flutter/services/hands_free_learning_session.dart';
import 'package:braille_lens_flutter/utils/answer_match.dart';
import 'package:braille_lens_flutter/utils/sinhala_prompts.dart';

import 'support/session_walk.dart';

List<String> _speech(List<HandsFreeAction> actions) =>
    actions.whereType<HandsFreeSpeak>().map((a) => a.text).toList();

void main() {
  test('the session\'s page-ready line becomes the Sinhala Stage 2 prompt', () {
    final s = HandsFreeLearningSession()..start();
    final said = _speech(s.onPrescanFinished(success: true));
    expect(said.map(SinhalaPrompts.forSessionLine),
        contains(SinhalaPrompts.pageReady));
  });

  test('the session\'s scan-failed line becomes the Sinhala retry prompt', () {
    final s = HandsFreeLearningSession()..start();
    final said = _speech(s.onPrescanFinished(success: false));
    expect(said.map(SinhalaPrompts.forSessionLine),
        contains(SinhalaPrompts.retryError));
  });

  test('answers carry the letter name and what was heard', () {
    expect(SinhalaPrompts.correct('කයන්න'), contains('කයන්න'));
    final wrong = SinhalaPrompts.incorrect('මයන්න', 'කයන්න');
    expect(wrong, contains('මයන්න'));
    expect(wrong, contains('කයන්න'));
  });

  test('every command a prompt asks for is one the app understands', () {
    expect(parseScreenCommand('Help'), ScreenCommand.help);
    expect(parseScreenCommand('Back'), ScreenCommand.back);
    expect(parseScreenCommand('Retry'), ScreenCommand.retry);
    expect(parseScreenCommand('Capture'), ScreenCommand.capture);
    expect(parseVoiceModeCommand('Learning'), 'learning');
    expect(parseVoiceModeCommand('Testing'), 'testing');
  });

  group('Learning and Testing speak only Sinhala', () {
    test('every spoken session line has Sinhala', () {
      final said = spokenSessionLines();
      expect(said.length, greaterThanOrEqualTo(12));
      for (final line in said) {
        final sinhala = SinhalaPrompts.sessionLine(line);
        expect(sinhala, isNotNull, reason: 'no Sinhala for "$line"');
        // Quoted command words ('Retry') are English on purpose: they are
        // what the English recogniser listens for.
        final withoutCommands = sinhala!.replaceAll(RegExp(r"'[A-Za-z]+'"), '');
        expect(RegExp('[a-zA-Z]').hasMatch(withoutCommands), isFalse,
            reason: 'English left in "$sinhala"');
      }
    });

    test('beep counts become Sinhala number words', () {
      expect(
        SinhalaPrompts.sessionLine(
            'Keep your hands off the page. Scanning after 8 beeps.'),
        contains('අටකට'),
      );
    });

    test('dots are read as Sinhala numbers', () {
      expect(SinhalaPrompts.dots('124'), 'තිත් එක, දෙක, හතර.');
      expect(SinhalaPrompts.dots(''), '');
    });
  });

  group('direction hints', () {
    const w = 20.0;

    test('point toward the nearest cell on the larger axis', () {
      expect(SinhalaPrompts.direction(Offset.zero, const Offset(40, 5), w),
          SinhalaPrompts.moveRight);
      expect(SinhalaPrompts.direction(Offset.zero, const Offset(-40, 5), w),
          SinhalaPrompts.moveLeft);
      expect(SinhalaPrompts.direction(Offset.zero, const Offset(5, 40), w),
          SinhalaPrompts.moveDown);
      expect(SinhalaPrompts.direction(Offset.zero, const Offset(5, -40), w),
          SinhalaPrompts.moveUp);
    });

    test('say nothing when already on the cell', () {
      expect(SinhalaPrompts.direction(Offset.zero, const Offset(4, 4), w),
          isNull);
    });
  });
}
