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

  // Each letter's name and the spellings the device recogniser returns for
  // it said alone ("L" comes back as "el", "elle", "ell"). Everyday words that
  // are easy to say by accident ("okay", "hey", "and", "yes") are left out.
  const letterNames = <String, Set<String>>{
    'a': {'ay', 'eh'},
    'b': {'bee', 'be', 'bea'},
    'c': {'see', 'sea', 'si'},
    'd': {'dee', 'di'},
    'e': {'ee'},
    'f': {'eff', 'ef'},
    'g': {'gee', 'ji', 'jee'},
    'h': {'aitch', 'eich', 'haitch'},
    'i': {'eye', 'aye'},
    'j': {'jay', 'jae'},
    'k': {'kay', 'cay', 'kei'},
    'l': {'ell', 'el', 'elle'},
    'm': {'em', 'emm'},
    'n': {'en', 'enn'},
    'o': {'oh', 'owe'},
    'p': {'pee', 'pea'},
    'q': {'cue', 'queue', 'kew'},
    'r': {'ar', 'are'},
    's': {'ess', 'es'},
    't': {'tee', 'tea', 'ti'},
    'u': {'you', 'yu'},
    'v': {'vee', 'vi'},
    'w': {'doubleyou', 'doubleu'},
    'x': {'ex', 'ecks'},
    'y': {'why', 'wye'},
    'z': {'zed', 'zee'},
  };

  bool tokenHits(String t) {
    if (t == exp) return true;
    return letterNames[exp]?.contains(t) ?? false;
  }

  // "double you" comes back as two words.
  if (exp == 'w' && tokens.join(' ').contains(RegExp(r'\bdouble\s*(you|u)\b'))) {
    return true;
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

/// Home language command: 'english' or 'sinhala' (whole words), or null.
String? parseVoiceLanguageCommand(String? spoken) {
  if (spoken == null) return null;
  final tokens = spoken
      .toLowerCase()
      .replaceAll(RegExp(r'[^a-z\s]'), ' ')
      .split(RegExp(r'\s+'))
      .where((t) => t.isNotEmpty);
  for (final t in tokens) {
    if (t == 'english') return 'english';
    // The English recogniser's spellings of "Sinhala".
    if (t == 'sinhala' || t == 'sinhalese' || t == 'singhala') {
      return 'sinhala';
    }
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
