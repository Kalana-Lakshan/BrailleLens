import 'package:flutter_test/flutter_test.dart';
import 'package:braille_lens_flutter/utils/answer_match.dart';

void main() {
  test('exact letter token matches', () {
    expect(spokenAnswerMatches('a', 'A'), isTrue);
    expect(spokenAnswerMatches('B', 'b'), isTrue);
  });

  test('letter name matches', () {
    expect(spokenAnswerMatches('see', 'C'), isTrue);
    expect(spokenAnswerMatches('kay', 'K'), isTrue);
  });

  test('substring in another word does not match', () {
    expect(spokenAnswerMatches('okay', 'A'), isFalse);
    expect(spokenAnswerMatches('please', 'E'), isFalse);
    expect(spokenAnswerMatches('hey', 'E'), isFalse);
  });

  test('empty or null is incorrect', () {
    expect(spokenAnswerMatches(null, 'A'), isFalse);
    expect(spokenAnswerMatches('', 'A'), isFalse);
    expect(spokenAnswerMatches('  ', 'A'), isFalse);
  });

  test('voice mode uses whole words not substrings', () {
    expect(parseVoiceModeCommand('learning'), 'learning');
    expect(parseVoiceModeCommand('open testing please'), 'testing');
    expect(parseVoiceModeCommand('latest'), isNull);
    expect(parseVoiceModeCommand('contest'), isNull);
  });
}
