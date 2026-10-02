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
/// a trailing syllable, and failing a learner for that would be wrong. Both
/// sides go through [_normalise] first (Unicode composition, zero-width
/// characters, homophones).
bool sinhalaAnswerMatches(String? spoken, String character) {
  final said = _normalise(spoken);
  if (said.isEmpty) return false;

  final ch = _normalise(character);
  if (ch.isEmpty) return false;
  final name = _normalise(sinhalaLetterName(character));

  // CTC output sometimes splits a name into syllables ("ක යන්න").
  final saidJoined = said.replaceAll(' ', '');

  for (final candidate in {ch, name}) {
    if (candidate.isEmpty) continue;
    if (said == candidate || saidJoined == candidate) return true;
    // "කයන්න කයි" or "අකුර කයන්න" — the answer is in there as a word.
    if (said.split(' ').contains(candidate)) return true;
  }

  // Name said without its ending, e.g. "කයන" for "කයන්න".
  if (name.length > 2 && said.length >= ch.length) {
    final stem = name.substring(0, name.length - 2);
    if (said == stem ||
        saidJoined == stem ||
        said.split(' ').contains(stem)) {
      return true;
    }
  }

  // The MMS recogniser gets the consonants of a letter name right but not
  // its vowel signs, and runs on into a suffix: "කයන්න" comes back as
  // "කියන්නෙක්", "හයන්න" as "හොයන්නෙක්", "තයන්න" as "තියෙන්ව". Comparing
  // consonant skeletons from the start of a word accepts those while still
  // telling letters apart — the letter itself is the first consonant, and
  // that must match. The bare letter is never prefix-matched: every word
  // starting with ක would pass.
  final nameSkeleton = _skeleton(name);
  if (nameSkeleton.length >= 3) {
    final stemSkeleton = nameSkeleton.substring(0, nameSkeleton.length - 1);
    for (final word in {...said.split(' '), saidJoined}) {
      if (_skeleton(word).startsWith(stemSkeleton)) return true;
    }
  }
  return false;
}

/// [value] with dependent vowel signs and the virama removed, leaving the
/// consonants and independent vowels: "කියන්නෙක්" → "කයනනක".
String _skeleton(String value) =>
    value.replaceAll(RegExp('[\u0DCA-\u0DDF\u0DF2\u0DF3]'), '');

/// True when [a] and [b] are the same letter or two letters that sound the
/// same in everyday speech (ණ/න, ළ/ල, ශ/ෂ/ස, aspirated/plain pairs) — the
/// comparison Testing Mode uses on the speech classifier's answer.
bool sinhalaLettersSoundAlike(String a, String b) {
  final x = _normalise(a);
  return x.isNotEmpty && x == _normalise(b);
}

/// Canonical compositions (Unicode NFC) for the split vowel signs. Dart has
/// no built-in normaliser, and a CTC vocabulary can emit either form: "ො" may
/// arrive as "ෙ" + "ා". The three-mark rule runs first, so a fully split
/// "ෝ" composes in one step instead of stopping at "ො" + "්".
const List<(String, String)> _compositions = [
  ('\u0DD9\u0DCF\u0DCA', '\u0DDD'), // e-sign + aa-sign + virama -> oo-sign
  ('\u0DDC\u0DCA', '\u0DDD'), // o-sign + virama -> oo-sign
  ('\u0DD9\u0DCF', '\u0DDC'), // e-sign + aa-sign -> o-sign
  ('\u0DD9\u0DCA', '\u0DDA'), // e-sign + virama -> ee-sign
  ('\u0DD9\u0DDF', '\u0DDE'), // e-sign + gayanukitta -> au-sign
];

/// Letters a speaker says the same way in everyday Sinhala, folded onto one
/// representative. The recogniser only hears sound, so without this a learner
/// who correctly names ණ ("නයන්න") or ඛ ("කයන්න") would be marked wrong for
/// a distinction no microphone can pick up. Applied to both sides of the
/// comparison. Distinct-sounding pairs (e.g. ග/ඟ, ද/ඳ) are deliberately left
/// alone.
const Map<String, String> _homophones = {
  'ණ': 'න', // retroflex ණ ≈ dental න
  'ළ': 'ල', // retroflex ළ ≈ dental ල
  'ශ': 'ස', 'ෂ': 'ස', // sibilants
  // Aspirates (mahaprana) ≈ their unaspirated pairs in speech.
  'ඛ': 'ක', 'ඝ': 'ග', 'ඡ': 'ච', 'ඣ': 'ජ', 'ඨ': 'ට',
  'ඪ': 'ඩ', 'ථ': 'ත', 'ධ': 'ද', 'ඵ': 'ප', 'භ': 'බ',
};

/// Comparison form: composes split vowel signs, drops zero-width and
/// formatting characters (they matter in rendering, not meaning), folds
/// homophones, strips punctuation, collapses whitespace and lowercases any
/// Latin the recogniser let through.
String _normalise(String? value) {
  if (value == null) return '';
  var s = value
      // ZWNJ, ZWJ, zero-width space, BOM, soft hyphen.
      .replaceAll(RegExp('[\u200B-\u200D\uFEFF\u00AD]'), '');
  for (final (from, to) in _compositions) {
    s = s.replaceAll(from, to);
  }
  final folded = StringBuffer();
  for (final rune in s.runes) {
    final c = String.fromCharCode(rune);
    folded.write(_homophones[c] ?? c);
  }
  return folded
      .toString()
      .replaceAll(RegExp(r'[^඀-෿\sa-zA-Z0-9]'), ' ')
      .replaceAll(RegExp(r'\s+'), ' ')
      .trim()
      .toLowerCase();
}

/// True when [character] is something a learner can be asked to name aloud:
/// Sinhala script, not an indicator label ("[IND-A]") or a blank cell.
bool isNameableSinhala(String character) {
  final ch = character.trim();
  if (ch.isEmpty || ch.startsWith('[')) return false;
  return RegExp(r'[඀-෿]').hasMatch(ch);
}
