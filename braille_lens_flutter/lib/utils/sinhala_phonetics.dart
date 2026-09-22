/// Sinhala letter names and spoken-answer matching for Learning and Testing.
///
/// A learner asked "what letter is this?" says the letter's *name*, not its
/// sound: ක is "කයන්න", න is "නයන්න". Sinhala names most consonants by
/// appending "යන්න", so that is the rule here, with the exceptions listed
/// explicitly. Both forms count as correct — the recogniser may return either,
/// and a beginner may well say just the letter.
library;

/// Letters whose spoken name is not simply `<letter>යන්න`.
const Map<String, String> _irregularNames = {
  // Vowels are named by their own sound.
  'අ': 'අයන්න',
  'ආ': 'ආයන්න',
  'ඇ': 'ඇයන්න',
  'ඈ': 'ඈයන්න',
  'ඉ': 'ඉයන්න',
  'ඊ': 'ඊයන්න',
  'උ': 'උයන්න',
  'ඌ': 'ඌයන්න',
  'එ': 'එයන්න',
  'ඒ': 'ඒයන්න',
  'ඔ': 'ඔයන්න',
  'ඕ': 'ඕයන්න',
  // Named after their shape or role rather than the "යන්න" pattern.
  'ං': 'බින්දුව',
  'ඃ': 'විසර්ගය',
  '්': 'හල් කිරීම',
};

/// The name a reader would say aloud for [character].
///
/// Returns an empty string for a space or an empty input, so callers can skip
/// narration rather than announce nothing.
String sinhalaLetterName(String character) {
  final ch = character.trim();
  if (ch.isEmpty || ch == ' ') return '';
  final irregular = _irregularNames[ch];
  if (irregular != null) return irregular;
  // Combining marks (vowel signs) have no standalone name; say the glyph.
  if (_isCombiningMark(ch)) return ch;
  return '$chයන්න';
}

bool _isCombiningMark(String ch) {
  if (ch.length != 1) return false;
  final code = ch.codeUnitAt(0);
  // Sinhala dependent vowel signs and virama.
  return code >= 0x0DCA && code <= 0x0DDF;
}

/// True when [spoken] is an acceptable answer for [character].
///
/// Accepts the bare letter, its name, and the name with the recogniser's
/// spacing or missing final consonant — CTC output routinely drops or splits
/// a trailing syllable, and failing a learner for that would be wrong.
bool sinhalaAnswerMatches(String? spoken, String character) {
  final said = _normalise(spoken);
  if (said.isEmpty) return false;

  final ch = _normalise(character);
  if (ch.isEmpty) return false;
  final name = _normalise(sinhalaLetterName(character));

  for (final candidate in {ch, name}) {
    if (candidate.isEmpty) continue;
    if (said == candidate) return true;
    // "කයන්න කයි" or "අකුර කයන්න" — the answer is in there as a word.
    if (said.split(' ').contains(candidate)) return true;
  }

  // Name said without its ending, e.g. "කයන" or "කය" for "කයන්න".
  if (name.length > 2 && said.length >= ch.length) {
    final stem = name.substring(0, name.length - 2);
    if (said == stem || said.split(' ').contains(stem)) return true;
  }
  return false;
}

/// Lowercases, strips punctuation and collapses whitespace. Zero-width joiners
/// matter in Sinhala rendering but not in comparison, so they go too.
String _normalise(String? value) {
  if (value == null) return '';
  return value
      .replaceAll(RegExp(r'[‌‍]'), '')
      .replaceAll(RegExp(r'[^඀-෿\sa-zA-Z0-9]'), ' ')
      .replaceAll(RegExp(r'\s+'), ' ')
      .trim()
      .toLowerCase();
}
