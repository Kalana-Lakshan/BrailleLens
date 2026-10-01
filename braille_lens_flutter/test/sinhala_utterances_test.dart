import 'package:flutter_test/flutter_test.dart';
import 'package:braille_lens_flutter/utils/sinhala_prompts.dart';

/// The approved Sinhala utterances, verbatim. If a prompt in
/// [SinhalaPrompts] is edited, this fails until the wording is re-approved.
void main() {
  test('Welcome', () {
    expect(
      SinhalaPrompts.welcome,
      'ආයුබෝවන්! බ්‍රේල් ලෙන්ස් වෙත සාදරයෙන් පිළිගනිමු.',
    );
  });

  test('Home instructions', () {
    expect(
      SinhalaPrompts.homeInstructions,
      "ඉගෙනුම් මාදිලියට පිවිසීමට 'Learning' ලෙසත්, පරීක්ෂණ මාදිලියට පිවිසීමට 'Testing' ලෙසත් පවසන්න. උපදෙස් ඇසීමට 'Help' ලෙස පවසන්න.",
    );
  });

  test('Entering Learning Mode', () {
    expect(
      SinhalaPrompts.enterLearning,
      "ඉගෙනුම් මාදිලියට පිවිසුණා. ආපසු ප්‍රධාන මෙනුවට යාමට 'Back' ලෙස පවසන්න.",
    );
  });

  test('Entering Testing Mode', () {
    expect(
      SinhalaPrompts.enterTesting,
      "පරීක්ෂණ මාදිලියට පිවිසුණා. ආපසු ප්‍රධාන මෙනුවට යාමට 'Back' ලෙස පවසන්න.",
    );
  });

  test('Stage 1 (Prescan)', () {
    expect(
      SinhalaPrompts.prescan,
      'කරුණාකර සම්පූර්ණ පිටුව හොඳින් පෙනෙන සේ කැමරාවට අල්ලාගෙන සිටින්න. නාද අටකට පසුව පිටුව ස්වයංක්‍රීයව ස්කෑන් වනු ඇත.',
    );
  });

  test('Stage 1 success / Stage 2', () {
    expect(
      SinhalaPrompts.pageReady,
      'පිටුව සූදානම්. දැන් ඔබේ ඇඟිල්ල අකුරක් මත තත්පර තුනක් නොසෙල්වී තබාගෙන සිටින්න.',
    );
  });

  test('Testing question (Stage 3)', () {
    expect(
      SinhalaPrompts.askLetter,
      'ඔබේ ඇඟිල්ල යට ඇති අකුර කුමක්ද? කරුණාකර එය ශබ්ද නඟා කියන්න.',
    );
  });

  test('Testing correct answer', () {
    expect(
      SinhalaPrompts.correct('{phonetic}'),
      'ඉතා නිවැරදියි! එම අක්ෂරය {phonetic}.',
    );
  });

  test('Testing incorrect answer', () {
    expect(
      SinhalaPrompts.incorrect('{transcript}', '{phonetic}'),
      'වැරදියි. ඔබ කීවේ {transcript}, නමුත් නිවැරදි අක්ෂරය {phonetic}.',
    );
  });

  test('Page not found / error', () {
    expect(
      SinhalaPrompts.retryError,
      "මට එය පැහැදිලි මදි. කරුණාකර නැවත උත්සාහ කරන්න, නැතහොත් 'Retry' ලෙස පවසන්න.",
    );
  });
}
