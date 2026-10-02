import 'package:braille_lens_flutter/services/hands_free_learning_session.dart';

/// Every line the session wants spoken, from both action kinds.
List<String> _spoken(List<HandsFreeAction> actions) => [
      for (final a in actions)
        if (a is HandsFreeStatus && a.speech != null)
          a.speech!
        else if (a is HandsFreeSpeak)
          a.text,
    ];

/// Walks the shared hands-free session through every path that speaks and
/// returns the distinct lines it asked to have spoken. The Sinhala
/// translation test checks each one, so a line reworded in the session
/// fails there instead of silently falling back to English.
Set<String> spokenSessionLines() {
  final t0 = DateTime.utc(2026, 1, 1);
  DateTime at(int ms) => t0.add(Duration(milliseconds: ms));
  final said = <String>[];

  var s = HandsFreeLearningSession()..start();
  said.addAll(_spoken(s.onSample(now: at(0), tipPresent: true)));
  said.addAll(_spoken(s.onSample(now: at(100), tipPresent: false)));
  said.addAll(_spoken(s.onSample(now: at(1000), tipPresent: false)));
  said.addAll(_spoken(s.onSample(now: at(1100), tipPresent: true)));
  s.onSample(now: at(1200), tipPresent: false);
  s.onSample(now: at(2100), tipPresent: false);
  said.addAll(_spoken(s.onCountdownFinished(HandsFreeCountdownKind.page)));
  said.addAll(_spoken(s.onPrescanFinished(success: false)));
  said.addAll(_spoken(s.onPrescanFinished(success: true)));

  // Reading: tip off a cell, then dwell to lock, then move.
  said.addAll(_spoken(s.onSample(now: at(3000), tipPresent: true)));
  s.onSample(now: at(3100), tipPresent: true, cellId: 7);
  said.addAll(
      _spoken(s.onSample(now: at(3700), tipPresent: true, cellId: 7)));
  said.addAll(
      _spoken(s.onSample(now: at(3800), tipPresent: true, cellId: 8)));

  // Lock → announce → cooldown → leave.
  s.onSample(now: at(4000), tipPresent: true, cellId: 7);
  s.onSample(now: at(4600), tipPresent: true, cellId: 7);
  s.onCountdownFinished(HandsFreeCountdownKind.lock);
  said.addAll(_spoken(s.onAnnounceFinished(announcedCellId: 7)));
  said.addAll(
      _spoken(s.onSample(now: at(5000), tipPresent: true, cellId: 9)));

  // Tip gone long enough → soft rescan, then its outcomes.
  s.onSample(now: at(5100), tipPresent: false);
  s.onSample(now: at(6000), tipPresent: false);
  said.addAll(_spoken(s.onSample(now: at(6900), tipPresent: false)));
  said.addAll(_spoken(s.onSample(now: at(7000), tipPresent: true)));
  said.addAll(_spoken(s.onSoftRescanFinished(success: false)));
  said.addAll(_spoken(s.onSoftRescanFinished(success: true)));

  s = HandsFreeLearningSession()..start();
  said.addAll(_spoken(s.onCountdownFinished(HandsFreeCountdownKind.page)));
  s.phase = HandsFreePhase.pageCountdown;
  said.addAll(_spoken(s.onCountdownFinished(HandsFreeCountdownKind.page)));

  return said.toSet();
}
