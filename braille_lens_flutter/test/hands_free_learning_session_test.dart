import 'package:flutter_test/flutter_test.dart';
import 'package:braille_lens_flutter/services/hands_free_learning_session.dart';

void main() {
  late HandsFreeLearningSession session;
  late DateTime t0;

  setUp(() {
    session = HandsFreeLearningSession(
      config: const HandsFreeConfig(
        dwellMs: 3000,
        lockBeepCount: 5,
        lockBeepIntervalMs: 500,
        moveAbsentMs: 1500,
        pageStableMs: 800,
        softRescanBeepCount: 3,
      ),
    );
    t0 = DateTime.utc(2026, 1, 1, 12, 0, 0);
    session.start();
  });

  DateTime at(int ms) => t0.add(Duration(milliseconds: ms));

  test('page hunt starts countdown after stable no-tip window', () {
    session.onSample(now: at(0), tipPresent: false, cellId: null);
    final mid = session.onSample(now: at(400), tipPresent: false, cellId: null);
    expect(mid.whereType<HandsFreePlayCountdown>(), isEmpty);

    final ready =
        session.onSample(now: at(900), tipPresent: false, cellId: null);
    expect(session.phase, HandsFreePhase.pageCountdown);
    expect(
      ready.whereType<HandsFreePlayCountdown>().single.kind,
      HandsFreeCountdownKind.page,
    );
  });

  test('tip during page countdown aborts', () {
    session.onSample(now: at(0), tipPresent: false, cellId: null);
    session.onSample(now: at(900), tipPresent: false, cellId: null);
    expect(session.phase, HandsFreePhase.pageCountdown);

    final abort =
        session.onSample(now: at(1000), tipPresent: true, cellId: null);
    expect(session.phase, HandsFreePhase.pageHunt);
    expect(abort.whereType<HandsFreeAbortCountdown>(), hasLength(1));
  });

  test('dwell fires lock beeps once then finger capture', () {
    // Skip to reading with a map.
    session.phase = HandsFreePhase.reading;

    session.onSample(now: at(0), tipPresent: true, cellId: 7);
    expect(session.phase, HandsFreePhase.dwelling);

    // Advance just under lockStart (500 ms).
    session.onSample(now: at(400), tipPresent: true, cellId: 7);
    expect(session.phase, HandsFreePhase.dwelling);

    final lock =
        session.onSample(now: at(600), tipPresent: true, cellId: 7);
    expect(session.phase, HandsFreePhase.lockBeeps);
    expect(
      lock.whereType<HandsFreePlayCountdown>().single.kind,
      HandsFreeCountdownKind.lock,
    );

    final after = session.onCountdownFinished(HandsFreeCountdownKind.lock);
    expect(session.phase, HandsFreePhase.announce);
    expect(after.whereType<HandsFreeRequestFingerCapture>(), hasLength(1));
  });

  test('leaving cell resets dwell before lock', () {
    session.phase = HandsFreePhase.reading;
    session.onSample(now: at(0), tipPresent: true, cellId: 1);
    session.onSample(now: at(400), tipPresent: true, cellId: 1);

    final moved =
        session.onSample(now: at(800), tipPresent: true, cellId: 2);
    expect(session.phase, HandsFreePhase.dwelling);
    expect(moved.whereType<HandsFreePlayCountdown>(), isEmpty);

    // New cell must re-accumulate; not immediately at lock.
    session.onSample(now: at(1000), tipPresent: true, cellId: 2);
    expect(session.phase, HandsFreePhase.dwelling);
  });

  test('announce cooldown suppresses same cell until tip leaves', () {
    session.phase = HandsFreePhase.announce;
    final cool = session.onAnnounceFinished(announcedCellId: 5);
    expect(session.phase, HandsFreePhase.cooldown);
    expect(cool.whereType<HandsFreeStatus>().first.message, contains('Move'));

    // Same cell still under tip — stay in cooldown.
    session.onSample(now: at(0), tipPresent: true, cellId: 5);
    expect(session.phase, HandsFreePhase.cooldown);

    // Tip leaves → reading again.
    final back =
        session.onSample(now: at(200), tipPresent: false, cellId: null);
    expect(session.phase, HandsFreePhase.reading);
    expect(back.whereType<HandsFreeStatus>().first.message, contains('Ready'));
  });

  test('dwell fires only once per visit (cooldown after announce)', () {
    session.phase = HandsFreePhase.reading;
    session.onSample(now: at(0), tipPresent: true, cellId: 3);
    session.onSample(now: at(600), tipPresent: true, cellId: 3);
    expect(session.phase, HandsFreePhase.lockBeeps);
    session.onCountdownFinished(HandsFreeCountdownKind.lock);
    session.onAnnounceFinished(announcedCellId: 3);

    // Still on cell 3 — must not lock again.
    final again =
        session.onSample(now: at(1000), tipPresent: true, cellId: 3);
    expect(session.phase, HandsFreePhase.cooldown);
    expect(again.whereType<HandsFreePlayCountdown>(), isEmpty);
  });

  test('tip absent for move window starts soft rescan', () {
    session.phase = HandsFreePhase.reading;
    // dt is capped at 1000 ms per sample, so accumulate across ticks.
    session.onSample(now: at(0), tipPresent: false, cellId: null);
    session.onSample(now: at(800), tipPresent: false, cellId: null);
    final soft =
        session.onSample(now: at(1600), tipPresent: false, cellId: null);
    expect(session.phase, HandsFreePhase.softRescan);
    expect(
      soft.whereType<HandsFreePlayCountdown>().single.kind,
      HandsFreeCountdownKind.softRescan,
    );

    final req = session.onCountdownFinished(HandsFreeCountdownKind.softRescan);
    expect(req.whereType<HandsFreeRequestSoftRescan>(), hasLength(1));

    final done = session.onSoftRescanFinished(success: true);
    expect(session.phase, HandsFreePhase.reading);
    expect(done.whereType<HandsFreeSpeak>(), isNotEmpty);
  });

  test('tip during soft-rescan countdown aborts', () {
    session.phase = HandsFreePhase.reading;
    session.onSample(now: at(0), tipPresent: false, cellId: null);
    session.onSample(now: at(800), tipPresent: false, cellId: null);
    session.onSample(now: at(1600), tipPresent: false, cellId: null);
    expect(session.phase, HandsFreePhase.softRescan);

    final abort =
        session.onSample(now: at(1800), tipPresent: true, cellId: 1);
    expect(session.phase, HandsFreePhase.reading);
    expect(abort.whereType<HandsFreeAbortCountdown>(), hasLength(1));
  });

  test('prescan failure returns to page hunt', () {
    session.phase = HandsFreePhase.buildingMap;
    final fail = session.onPrescanFinished(success: false);
    expect(session.phase, HandsFreePhase.pageHunt);
    expect(fail.whereType<HandsFreeSpeak>(), isNotEmpty);
  });
}
