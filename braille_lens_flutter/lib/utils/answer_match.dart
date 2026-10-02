/// Spoken-answer matching for Testing Mode.
///
/// Uses whole tokens, not substring [contains], so "okay" does not match "A"
/// and "please" does not match "E".
bool spokenAnswerMatches(String? spoken, String expected) {
  if (spoken == null) return false;
  final exp = expected.trim().toLowerCase();
  if (exp.isEmpty) return false;

  final tokens = spoken
      .toLowerCase()
      .replaceAll(RegExp(r'[^a-z0-9\s\u0d80-\u0dff]'), ' ')
      .split(RegExp(r'\s+'))
      .where((t) => t.isNotEmpty)
      .toList();
  if (tokens.isEmpty) return false;

  const letterNames = {
    'a': 'ay',
    'b': 'bee',
    'c': 'see',
    'd': 'dee',
    'e': 'ee',
    'f': 'eff',
    'g': 'gee',
    'h': 'aitch',
    'i': 'eye',
    'j': 'jay',
    'k': 'kay',
    'l': 'ell',
    'm': 'em',
    'n': 'en',
    'o': 'oh',
    'p': 'pee',
    'q': 'cue',
    'r': 'ar',
    's': 'ess',
    't': 'tee',
    'u': 'you',
    'v': 'vee',
    'w': 'doubleyou',
    'x': 'ex',
    'y': 'why',
    'z': 'zed',
  };

  bool tokenHits(String t) {
    if (t == exp) return true;
    if (exp.length == 1 && t.length == 1 && t == exp) return true;
    final name = letterNames[exp];
    if (name != null && t == name) return true;
    return false;
  }

  return tokens.any(tokenHits);
}

/// Home voice command: whole words only ("latest" must not open Testing).
String? parseVoiceModeCommand(String? spoken) {
  if (spoken == null) return null;
  final tokens = spoken
      .toLowerCase()
      .replaceAll(RegExp(r'[^a-z\s]'), ' ')
      .split(RegExp(r'\s+'))
      .where((t) => t.isNotEmpty);
  for (final t in tokens) {
    if (t == 'learning' || t == 'learn') return 'learning';
  }
  for (final t in tokens) {
    if (t == 'testing' || t == 'test') return 'testing';
  }
  return null;
}

/// True when the user said the whole word "stop" (not "stopped" / "stopwatch").
bool spokenContainsStopKeyword(String? spoken) {
  if (spoken == null) return false;
  final tokens = spoken
      .toLowerCase()
      .replaceAll(RegExp(r'[^a-z\s]'), ' ')
      .split(RegExp(r'\s+'))
      .where((t) => t.isNotEmpty);
  return tokens.contains('stop');
}

/// In-screen voice commands for Learning and Testing.
enum ScreenCommand { back, retry, help, capture }

/// English command words, matched as whole words so the app's own prompts
/// ("Ayeth try karanna", "Refresh cancelled") cannot trigger one.
const Map<String, ScreenCommand> _screenCommandWords = {
  'back': ScreenCommand.back,
  'home': ScreenCommand.back,
  'menu': ScreenCommand.back,
  'retry': ScreenCommand.retry,
  'rescan': ScreenCommand.retry,
  'restart': ScreenCommand.retry,
  'help': ScreenCommand.help,
  'capture': ScreenCommand.capture,
};

/// The command in [spoken], or null. The recogniser's close mishearings of
/// the short words ("bak", "re try") are folded in too.
ScreenCommand? parseScreenCommand(String? spoken) {
  if (spoken == null) return null;
  final text = spoken
      .toLowerCase()
      .replaceAll(RegExp(r'[^a-z\s]'), ' ')
      .replaceAllMapped(RegExp(r'\bre\s+(try|scan)\b'), (m) => 're${m[1]}')
      .replaceAll(RegExp(r'\s+'), ' ')
      .trim();
  for (final word in text.split(' ')) {
    final hit = _screenCommandWords[word == 'bak' ? 'back' : word];
    if (hit != null) return hit;
  }
  return null;
}
