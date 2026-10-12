/// The language the learner picked on Home, and everything Learning and
/// Testing say or check that depends on it.
///
/// Sinhala reads each cell with the Sri Lankan Sinhala Braille chart and
/// speaks through the Sinhala voice; English reads the same 6-dot code as an
/// English Grade-1 letter and speaks through the English voice. The command
/// words ('Learning', 'Back', 'Retry', …) are English in both, since the
/// command recogniser is English.
library;

import 'dart:ui' show Offset;

import 'package:flutter/foundation.dart' show debugPrint;

import '../models/braille_cell.dart';
import '../services/audio_service.dart';
import 'english_braille.dart';
import 'sinhala_phonetics.dart';
import 'sinhala_prompts.dart';

enum AppLanguage {
  sinhala('සිංහල', 'SI'),
  english('English', 'EN');

  /// The language's own name, shown on the Home toggle.
  final String label;
  final String short;

  const AppLanguage(this.label, this.short);

  AppLanguage get other =>
      this == AppLanguage.sinhala ? AppLanguage.english : AppLanguage.sinhala;

  AppPrompts get prompts => this == AppLanguage.sinhala
      ? const SinhalaAppPrompts()
      : const EnglishAppPrompts();
}

abstract class AppPrompts {
  const AppPrompts();

  AppLanguage get language;

  /// Speaks [text] with the voice for this language.
  Future<void> speak(AudioService audio, String text);

  String get welcome;
  String get homeInstructions;
  String get languageChosen;
  String get enterLearning;
  String get enterTesting;
  String get prescan;
  String get pageReady;
  String get askLetter;
  String correct(String name);
  String incorrect(String said, String name);
  String get retryError;
  String cellsFound(int n);
  String get fingerStillOnPage;
  String get notALetter;
  String dots(String dotDigits);
  String get learningHelp;
  String get testingHelp;
  String get returningHome;
  String get noCamera;
  String get readerFailed;
  String get fingerDetectorFailed;
  String get sttFailed;
  String get didNotHear;
  String get gaveUpListening;
  String reveal(String name);
  String? direction(Offset tip, Offset target, double cellWidth);

  /// Speaks a session line ([HandsFreeSpeak] / status speech), translated
  /// when this language has a version of it.
  Future<void> speakLine(AudioService audio, String text);

  /// Learning's announcement for a letter.
  String announce(String name);

  /// The letter a learner names for [cell] in this language, or null for an
  /// indicator, a blank or a pattern this language has no letter for.
  String? letterOf(BrailleCell cell);

  /// What a reader says aloud for [letter] (Sinhala: ක → කයන්න).
  String letterName(String letter);

  /// On-screen text for [cell]: its letter, or a code for other cells.
  String cellLabel(BrailleCell cell);
}

class SinhalaAppPrompts extends AppPrompts {
  const SinhalaAppPrompts();

  @override
  AppLanguage get language => AppLanguage.sinhala;

  @override
  Future<void> speak(AudioService audio, String text) =>
      audio.speakSinhala(text);

  @override
  String get welcome => SinhalaPrompts.welcome;
  @override
  String get homeInstructions => '${SinhalaPrompts.homeInstructions} '
      "ඉංග්‍රීසි භාෂාවට මාරු වීමට 'English' ලෙස පවසන්න.";
  @override
  String get languageChosen => 'සිංහල භාෂාව තෝරා ගත්තා.';
  @override
  String get enterLearning => SinhalaPrompts.enterLearning;
  @override
  String get enterTesting => SinhalaPrompts.enterTesting;
  @override
  String get prescan => SinhalaPrompts.prescan;
  @override
  String get pageReady => SinhalaPrompts.pageReady;
  @override
  String get askLetter => SinhalaPrompts.askLetter;
  @override
  String correct(String name) => SinhalaPrompts.correct(name);
  @override
  String incorrect(String said, String name) =>
      SinhalaPrompts.incorrect(said, name);
  @override
  String get retryError => SinhalaPrompts.retryError;
  @override
  String cellsFound(int n) => SinhalaPrompts.cellsFound(n);
  @override
  String get fingerStillOnPage => SinhalaPrompts.fingerStillOnPage;
  @override
  String get notALetter => SinhalaPrompts.notALetter;
  @override
  String dots(String dotDigits) => SinhalaPrompts.dots(dotDigits);
  @override
  String get learningHelp => SinhalaPrompts.learningHelp;
  @override
  String get testingHelp => SinhalaPrompts.testingHelp;
  @override
  String get returningHome => SinhalaPrompts.returningHome;
  @override
  String get noCamera => SinhalaPrompts.noCamera;
  @override
  String get readerFailed => SinhalaPrompts.readerFailed;
  @override
  String get fingerDetectorFailed => SinhalaPrompts.fingerDetectorFailed;
  @override
  String get sttFailed => SinhalaPrompts.sttFailed;
  @override
  String get didNotHear => SinhalaPrompts.didNotHear;
  @override
  String get gaveUpListening => SinhalaPrompts.gaveUpListening;
  @override
  String reveal(String name) => SinhalaPrompts.reveal(name);
  @override
  String? direction(Offset tip, Offset target, double cellWidth) =>
      SinhalaPrompts.direction(tip, target, cellWidth);

  /// A line with no Sinhala yet stays on the English voice rather than
  /// being mangled by the Sinhala one.
  @override
  Future<void> speakLine(AudioService audio, String text) {
    final sinhala = SinhalaPrompts.sessionLine(text) ??
        (RegExp('[඀-෿]').hasMatch(text) ? text : null);
    if (sinhala != null) return audio.speakSinhala(sinhala);
    debugPrint('[Prompts] no Sinhala for "$text"');
    return audio.speak(text);
  }

  @override
  String announce(String name) => 'අක්ෂරය $name';

  @override
  String? letterOf(BrailleCell cell) {
    final ch = cell.char.trim();
    return isNameableSinhala(ch) ? ch : null;
  }

  @override
  String letterName(String letter) {
    final name = sinhalaLetterName(letter);
    return name.isEmpty ? letter : name;
  }

  @override
  String cellLabel(BrailleCell cell) => cell.displayLabel;
}

class EnglishAppPrompts extends AppPrompts {
  const EnglishAppPrompts();

  @override
  AppLanguage get language => AppLanguage.english;

  @override
  Future<void> speak(AudioService audio, String text) => audio.speak(text);

  @override
  String get welcome => 'Welcome to BrailleLens.';
  @override
  String get homeInstructions =>
      "Say 'Learning' to enter Learning Mode, or 'Testing' to enter Testing "
      "Mode. Say 'Sinhala' to switch language, or 'Help' to hear this again.";
  @override
  String get languageChosen => 'English selected.';
  @override
  String get enterLearning =>
      "Learning Mode. Say 'Back' to return to the main menu.";
  @override
  String get enterTesting =>
      "Testing Mode. Say 'Back' to return to the main menu.";
  @override
  String get prescan => 'Hold the whole page clearly in front of the camera. '
      'It will be scanned automatically after eight beeps.';
  @override
  String get pageReady =>
      'Page ready. Now hold your finger still on a letter for three seconds.';
  @override
  String get askLetter =>
      'What is the letter under your finger? Please say it out loud.';
  @override
  String correct(String name) => 'Correct! The letter is $name.';
  @override
  String incorrect(String said, String name) =>
      'Not quite. You said $said, but the letter is $name.';
  @override
  String get retryError =>
      "I could not read that clearly. Please try again, or say 'Retry'.";
  @override
  String cellsFound(int n) =>
      n == 1 ? 'Found 1 Braille cell.' : 'Found $n Braille cells.';
  @override
  String get fingerStillOnPage =>
      'Your finger is still on the page, so the old page map is kept.';
  @override
  String get notALetter =>
      'This is not a letter. It is a sign or indicator cell.';
  @override
  String dots(String dotDigits) {
    final digits = dotDigits.split('').where('123456'.contains).toList();
    if (digits.isEmpty) return '';
    return '${digits.length == 1 ? 'Dot' : 'Dots'} ${digits.join(', ')}.';
  }

  @override
  String get learningHelp =>
      'Place your finger on a letter and hold still until the beeps end. '
      "Say 'Back' for the main menu, 'Retry' to scan the page again, or "
      "'Help' to hear this again.";
  @override
  String get testingHelp =>
      'Place your finger on a letter and hold still until the beeps end, '
      'then say the letter out loud. '
      "Say 'Back' for the main menu, 'Retry' to scan the page again, or "
      "'Help' to hear this again.";
  @override
  String get returningHome => 'Returning to the main menu.';
  @override
  String get noCamera => 'No camera available.';
  @override
  String get readerFailed => 'Warning: the Braille reader did not load.';
  @override
  String get fingerDetectorFailed => 'The finger detector did not load. '
      'Tap your fingertip on the screen instead.';
  @override
  String get sttFailed =>
      'Speech recognition is not available, so I will tell you the letter.';
  @override
  String get didNotHear => 'I did not hear you. Please say it again.';
  @override
  String get gaveUpListening => 'I did not hear you.';
  @override
  String reveal(String name) => 'The letter is $name.';

  static const moveRight = 'Move your finger a little to the right.';
  static const moveLeft = 'Move your finger a little to the left.';
  static const moveUp = 'Move your finger a little up.';
  static const moveDown = 'Move your finger a little down.';

  /// Same rule as [SinhalaPrompts.direction].
  @override
  String? direction(Offset tip, Offset target, double cellWidth) {
    if (cellWidth <= 0) return null;
    final d = target - tip;
    if (d.distance < cellWidth * 0.5) return null;
    if (d.dx.abs() >= d.dy.abs()) {
      return d.dx > 0 ? moveRight : moveLeft;
    }
    return d.dy > 0 ? moveDown : moveUp;
  }

  /// The session speaks English already; only its one Sinhala line needs
  /// replacing.
  @override
  Future<void> speakLine(AudioService audio, String text) => audio.speak(
      SinhalaPrompts.forSessionLine(text) == SinhalaPrompts.pageReady
          ? pageReady
          : text);

  @override
  String announce(String name) => 'Letter $name.';

  @override
  String? letterOf(BrailleCell cell) => englishBrailleLetter(cell.code);

  /// Upper case, so the voice says "B" as a letter name rather than a word.
  @override
  String letterName(String letter) => letter.toUpperCase();

  @override
  String cellLabel(BrailleCell cell) =>
      englishBrailleLetter(cell.code)?.toUpperCase() ?? '#${cell.code}';
}
