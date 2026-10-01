/// Sinhala-script prompts, spoken by the Sinhala TTS voice
/// ([AudioService.speakSinhala]) on Home, Learning and Testing.
///
/// The English command words inside them ('Learning', 'Back', 'Retry', …)
/// are what the English recogniser listens for, so they stay in English.
library;

import 'dart:ui' show Offset;

class SinhalaPrompts {
  SinhalaPrompts._();

  /// Home, repeated on return to the menu and for "Help".
  static const homeInstructions =
      "ඉගෙනුම් මාදිලියට පිවිසීමට 'Learning' ලෙසත්, පරීක්ෂණ මාදිලියට පිවිසීමට 'Testing' ලෙසත් පවසන්න. උපදෙස් ඇසීමට 'Help' ලෙස පවසන්න.";

  static const welcome =
      'ආයුබෝවන්! බ්‍රේල් ලෙන්ස් වෙත සාදරයෙන් පිළිගනිමු.';

  static const enterLearning = "ඉගෙනුම් මාදිලියට පිවිසුණා. ආපසු ප්‍රධාන මෙනුවට යාමට 'Back' ලෙස පවසන්න.";

  static const enterTesting = "පරීක්ෂණ මාදිලියට පිවිසුණා. ආපසු ප්‍රධාන මෙනුවට යාමට 'Back' ලෙස පවසන්න.";

  /// Stage 1: scan the page automatically after 8 beats.
  static const prescan = 'කරුණාකර සම්පූර්ණ පිටුව හොඳින් පෙනෙන සේ කැමරාවට '
      'අල්ලාගෙන සිටින්න. නාද අටකට පසුව පිටුව ස්වයංක්‍රීයව ස්කෑන් වනු ඇත.';

  /// Stage 1 done → Stage 2: hold a finger on a letter.
  static const pageReady = 'පිටුව සූදානම්. දැන් ඔබේ ඇඟිල්ල අකුරක් මත '
      'තත්පර තුනක් නොසෙල්වී තබාගෙන සිටින්න.';

  /// Testing, Stage 3: name the letter.
  static const askLetter =
      'ඔබේ ඇඟිල්ල යට ඇති අකුර කුමක්ද? කරුණාකර එය ශබ්ද නඟා කියන්න.';

  static String correct(String phonetic) =>
      'ඉතා නිවැරදියි! එම අක්ෂරය $phonetic.';

  static String incorrect(String transcript, String phonetic) =>
      'වැරදියි. ඔබ කීවේ $transcript, නමුත් නිවැරදි අක්ෂරය $phonetic.';

  /// Page not found, capture failed, no finger or no letter found.
  static const retryError = 'මට එය පැහැදිලි මදි. කරුණාකර නැවත උත්සාහ '
      "කරන්න, නැතහොත් 'Retry' ලෙස පවසන්න.";

  /// Session lines ([HandsFreeLearningSession], shared and left unchanged)
  /// that Learning and Testing replace with the prompts above.
  static const Map<String, String> _sessionOverrides = {
    'පිටුව ස්කෑන් කර අවසන්. දැන් ඔබේ ඇඟිල්ල අකුරක් මත තබන්න.': pageReady,
    'Page scan failed. Hold the page steady with good lighting.': retryError,
  };

  /// The Sinhala prompt replacing a session line, or null to keep it.
  static String? forSessionLine(String text) => _sessionOverrides[text];

  // ── Learning and Testing: the rest of the flow ─────────────────────────────
  // DRAFT wording, not yet approved — review before pinning these in
  // test/sinhala_utterances_test.dart like the prompts above.

  /// Every spoken line of the hands-free session, English → Sinhala, so
  /// neither Learning nor Testing speaks English. Keys are the session's exact
  /// text; test/sinhala_prompts_test.dart walks the session and fails if a
  /// line is missing here.
  static const Map<String, String> _learningSession = {
    'Clear the page — keep fingers out of the frame.':
        'පිටුවෙන් අත ඉවත් කරන්න. ඇඟිලි කැමරාවට නොපෙනෙන සේ තබන්න.',
    'Refresh cancelled. Keep reading.':
        'පිටුව යළි ස්කෑන් කිරීම නැවැත්වුණා. දිගටම කියවන්න.',
    'Finger moved. Hold still on one letter.':
        'ඇඟිල්ල සෙලවුණා. එක අකුරක් මත නොසෙල්වී තබාගෙන සිටින්න.',
    'Please take your hand off the page for the automatic scan.':
        'ස්වයංක්‍රීය ස්කෑන් කිරීම සඳහා කරුණාකර පිටුවෙන් අත ඉවත් කරන්න.',
    'Move your finger onto a letter.': 'ඇඟිල්ල අකුරක් මතට ගෙන යන්න.',
    'Hold still.': 'නොසෙල්වී සිටින්න.',
    'Ready for the next letter.': 'ඊළඟ අකුරට සූදානම්.',
    'Move to the next letter.': 'ඊළඟ අකුරට යන්න.',
    'Scanning the page. Please wait.':
        'පිටුව ස්කෑන් කරමින් පවතී. කරුණාකර රැඳී සිටින්න.',
    'Could not refresh the page map. Keep reading.':
        'පිටුව යළි ස්කෑන් කිරීමට නොහැකි වුණා. දිගටම කියවන්න.',
    'Page updated. Place your finger on a letter.':
        'පිටුව යාවත්කාලීන වුණා. ඔබේ ඇඟිල්ල අකුරක් මත තබන්න.',
  };

  static final RegExp _pageCountdown = RegExp(
    r'^Keep your hands off the page\. Scanning after (\d+) beeps\.$',
  );
  static final RegExp _softRescanCountdown = RegExp(
    r'^Finger lifted\. Refreshing the page map after (\d+) beeps\. '
    r'Touch a letter to cancel\.$',
  );

  /// Sinhala for any spoken session line (including the two in
  /// [forSessionLine]), or null for a line not translated yet.
  static String? sessionLine(String text) {
    final fixed = forSessionLine(text) ?? _learningSession[text];
    if (fixed != null) return fixed;

    final page = _pageCountdown.firstMatch(text);
    if (page != null) {
      return 'පිටුවෙන් අත් ඉවත් කර තබන්න. '
          'නාද ${_countKata(int.parse(page.group(1)!))} පසු පිටුව ස්කෑන් කෙරේ.';
    }
    final soft = _softRescanCountdown.firstMatch(text);
    if (soft != null) {
      return 'ඇඟිල්ල ඉවත් කළා. නාද '
          '${_countKata(int.parse(soft.group(1)!))} පසු පිටුව යළි ස්කෑන් '
          'කෙරේ. එය නැවැත්වීමට අකුරක් ස්පර්ශ කරන්න.';
    }
    return null;
  }

  static String cellsFound(int n) => 'බ්‍රේල් කොටු $n ක් හමු වුණා.';

  static const fingerStillOnPage =
      'ඇඟිල්ල තවමත් පිටුව මත ඇති නිසා, පැරණි පිටු සිතියම තබා ගනු ලැබේ.';

  static const notALetter = 'මෙය තනි අකුරක් නොව, දර්ශක කොටුවකි.';

  /// "Dots 1, 2, 4" from a dot string such as "124".
  static String dots(String dotDigits) {
    final words = dotDigits
        .split('')
        .where(_digitWords.containsKey)
        .map((d) => _digitWords[d]!)
        .join(', ');
    return words.isEmpty ? '' : 'තිත් $words.';
  }

  static const learningHelp = 'ඇඟිල්ල අකුරක් මත තබා, නාද අවසන් වන තුරු '
      "නොසෙල්වී සිටින්න. ප්‍රධාන මෙනුවට යාමට 'Back', පිටුව නැවත ස්කෑන් "
      "කිරීමට 'Retry', මෙම උපදෙස් නැවත ඇසීමට 'Help' ලෙස පවසන්න.";

  static const returningHome = 'ප්‍රධාන මෙනුවට ආපසු යමින්.';

  static const noCamera = 'කැමරාවක් නොමැත.';
  static const readerFailed = 'අවවාදයයි: බ්‍රේල් කියවනය පූරණය වුණේ නැහැ.';
  static const fingerDetectorFailed = 'ඇඟිලි හඳුනාගැනීම පූරණය වුණේ නැහැ. '
      'තිරය මත ඔබේ ඇඟිලි තුඩ තට්ටු කරන්න.';

  static const sttFailed =
      'කථන ආකෘතිය පූරණය වුණේ නැහැ. එබැවින් මම අකුර ඔබට කියන්නම්.';

  // ── Testing only ───────────────────────────────────────────────────────────

  static const testingHelp = 'ඇඟිල්ල අකුරක් මත තබා, නාද අවසන් වන තුරු '
      'නොසෙල්වී සිටින්න. නාදයෙන් පසු එම අකුර ශබ්ද නඟා කියන්න. '
      "ප්‍රධාන මෙනුවට යාමට 'Back', පිටුව නැවත ස්කෑන් කිරීමට 'Retry', "
      "මෙම උපදෙස් නැවත ඇසීමට 'Help' ලෙස පවසන්න.";

  /// Nothing heard on the first try: ask again.
  static const didNotHear = 'මට ඇසුණේ නැහැ. කරුණාකර නැවත කියන්න.';

  /// Nothing heard on the last try.
  static const gaveUpListening = 'මට ඇසුණේ නැහැ.';

  /// The answer, when it cannot be scored (no speech model, or the learner
  /// was not heard).
  static String reveal(String phonetic) => 'මෙම අක්ෂරය $phonetic.';

  static const moveRight = 'ඇඟිල්ල තව ටිකක් දකුණට ගෙන යන්න.';
  static const moveLeft = 'ඇඟිල්ල තව ටිකක් වමට ගෙන යන්න.';
  static const moveUp = 'ඇඟිල්ල තව ටිකක් ඉහළට ගෙන යන්න.';
  static const moveDown = 'ඇඟිල්ල තව ටිකක් පහළට ගෙන යන්න.';

  /// Which way to move a fingertip at [tip] to reach a cell centred at
  /// [target], both in page-image pixels (x right, y down the page).
  ///
  /// Null when the tip is already within half a cell — the learner is on it,
  /// and "move a little" would send them past it. The larger axis wins: one
  /// instruction at a time is easier to follow by touch than a diagonal.
  static String? direction(Offset tip, Offset target, double cellWidth) {
    if (cellWidth <= 0) return null;
    final d = target - tip;
    if (d.distance < cellWidth * 0.5) return null;
    if (d.dx.abs() >= d.dy.abs()) {
      return d.dx > 0 ? moveRight : moveLeft;
    }
    return d.dy > 0 ? moveDown : moveUp;
  }

  static const Map<String, String> _digitWords = {
    '1': 'එක',
    '2': 'දෙක',
    '3': 'තුන',
    '4': 'හතර',
    '5': 'පහ',
    '6': 'හය',
  };

  /// "N-kata" form for counts ("අටකට" = after eight); digits past ten.
  static String _countKata(int n) {
    const words = {
      1: 'එකකට',
      2: 'දෙකකට',
      3: 'තුනකට',
      4: 'හතරකට',
      5: 'පහකට',
      6: 'හයකට',
      7: 'හතකට',
      8: 'අටකට',
      9: 'නවයකට',
      10: 'දහයකට',
    };
    return words[n] ?? '$n කට';
  }
}