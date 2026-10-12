/// English Grade-1 (uncontracted) Braille letters by 6-dot cell code
/// (dot n sets bit n-1, the same code the cell CNN outputs — see
/// `braille_labels.json`'s `en` column).
library;

const Map<int, String> _letters = {
  1: 'a', 3: 'b', 9: 'c', 25: 'd', 17: 'e', 11: 'f', 27: 'g', 19: 'h',
  10: 'i', 26: 'j', 5: 'k', 7: 'l', 13: 'm', 29: 'n', 21: 'o', 15: 'p',
  31: 'q', 23: 'r', 14: 's', 30: 't', 37: 'u', 39: 'v', 58: 'w', 45: 'x',
  61: 'y', 53: 'z',
};

/// The English letter for [code], or null for a blank, a capital/number
/// sign or a pattern with no Grade-1 letter.
String? englishBrailleLetter(int code) => _letters[code];
