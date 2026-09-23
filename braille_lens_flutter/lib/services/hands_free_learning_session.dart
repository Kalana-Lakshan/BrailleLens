/// SRS FR5/FR6/FR8 hands-free Learning session (dwell-triggered, not a fixed 8s clock).
///
/// Pure state machine: the host feeds tip samples and reports when captures /
/// countdowns finish; this class emits [HandsFreeAction]s. No camera or ONNX.
library;

enum HandsFreePhase {
  idle,
  pageHunt,
  pageCountdown,
  buildingMap,
  reading,
  dwelling,
  lockBeeps,
  announce,
  cooldown,
  softRescan,
}

class HandsFreeConfig {
  /// Tip must stay on the same cell this long before announce (SRS ~2–3 s).
  final int dwellMs;

  /// Beeps played at the end of dwell / for page baseline.
  final int lockBeepCount;
  final int lockBeepIntervalMs;

  /// Tip absent this long while reading → soft CellMap refresh.
  final int moveAbsentMs;

  /// Tip absent this long in page hunt before page countdown.
  final int pageStableMs;

  /// Soft-rescan uses a shorter beep train.
  final int softRescanBeepCount;

  const HandsFreeConfig({
    this.dwellMs = 3000,
    this.lockBeepCount = 5,
    this.lockBeepIntervalMs = 500,
    this.moveAbsentMs = 1500,
    this.pageStableMs = 800,
    this.softRescanBeepCount = 3,
  });

  /// Dwell elapsed when lock beeps should start.
  int get lockStartMs {
    final beepWindow = lockBeepCount * lockBeepIntervalMs;
    final start = dwellMs - beepWindow;
    return start < 0 ? 0 : start;
  }
}

sealed class HandsFreeAction {
  const HandsFreeAction();
}

class HandsFreeStatus extends HandsFreeAction {
  final String message;
  const HandsFreeStatus(this.message);
}

class HandsFreePlayCountdown extends HandsFreeAction {
  final int count;
  final int intervalMs;
  final HandsFreeCountdownKind kind;
  const HandsFreePlayCountdown({
    required this.count,
    required this.intervalMs,
    required this.kind,
  });
}

enum HandsFreeCountdownKind { page, lock, softRescan }

class HandsFreeAbortCountdown extends HandsFreeAction {
  const HandsFreeAbortCountdown();
}

class HandsFreeRequestPrescan extends HandsFreeAction {
  const HandsFreeRequestPrescan();
}

class HandsFreeRequestFingerCapture extends HandsFreeAction {
  const HandsFreeRequestFingerCapture();
}

class HandsFreeRequestSoftRescan extends HandsFreeAction {
  const HandsFreeRequestSoftRescan();
}

class HandsFreeSpeak extends HandsFreeAction {
  final String text;
  final bool sinhala;
  const HandsFreeSpeak(this.text, {this.sinhala = false});
}

/// Vision-driven dwell session for Learning Mode.
class HandsFreeLearningSession {
  HandsFreeLearningSession({
    this.config = const HandsFreeConfig(),
  });

  final HandsFreeConfig config;

  HandsFreePhase phase = HandsFreePhase.idle;
  bool paused = false;

  DateTime? _lastSampleAt;
  int _absentMs = 0;
  int _stableNoTipMs = 0;
  int _dwellMs = 0;
  int? _dwellCellId;
  int? _announcedCellId;
  int? _cooldownCellId;

  void start() {
    phase = HandsFreePhase.pageHunt;
    _resetTrackers();
    paused = false;
  }

  void pause() => paused = true;

  void resume() => paused = false;

  void resetToPageHunt() {
    phase = HandsFreePhase.pageHunt;
    _resetTrackers();
    _announcedCellId = null;
    _cooldownCellId = null;
  }

  void _resetTrackers() {
    _lastSampleAt = null;
    _absentMs = 0;
    _stableNoTipMs = 0;
    _dwellMs = 0;
    _dwellCellId = null;
  }

  /// Feed one tip observation. [cellId] is a cheap mapped CellMap id, or null.
  List<HandsFreeAction> onSample({
    required DateTime now,
    required bool tipPresent,
    int? cellId,
  }) {
    if (paused || phase == HandsFreePhase.idle) return const [];
    if (phase == HandsFreePhase.buildingMap ||
        phase == HandsFreePhase.announce ||
        phase == HandsFreePhase.softRescan ||
        phase == HandsFreePhase.pageCountdown ||
        phase == HandsFreePhase.lockBeeps) {
      // Host owns the countdown / capture; only watch for abort conditions.
      return _sampleDuringHostOwned(now: now, tipPresent: tipPresent, cellId: cellId);
    }

    final dt = _dtMs(now);
    _lastSampleAt = now;

    switch (phase) {
      case HandsFreePhase.pageHunt:
        return _pageHunt(dt: dt, tipPresent: tipPresent);
      case HandsFreePhase.reading:
      case HandsFreePhase.dwelling:
        return _readingOrDwelling(
          dt: dt,
          tipPresent: tipPresent,
          cellId: cellId,
        );
      case HandsFreePhase.cooldown:
        return _cooldown(tipPresent: tipPresent, cellId: cellId);
      default:
        return const [];
    }
  }

  List<HandsFreeAction> _sampleDuringHostOwned({
    required DateTime now,
    required bool tipPresent,
    int? cellId,
  }) {
    _lastSampleAt = now;
    if (phase == HandsFreePhase.pageCountdown && tipPresent) {
      phase = HandsFreePhase.pageHunt;
      _stableNoTipMs = 0;
      return [
        const HandsFreeAbortCountdown(),
        const HandsFreeSpeak('Clear the page — keep fingers out of the frame.'),
        const HandsFreeStatus('Finger seen — waiting for a clear page'),
      ];
    }
    if (phase == HandsFreePhase.softRescan && tipPresent) {
      phase = HandsFreePhase.reading;
      _absentMs = 0;
      return [
        const HandsFreeAbortCountdown(),
        const HandsFreeStatus('Finger returned — page refresh cancelled'),
      ];
    }
    if (phase == HandsFreePhase.lockBeeps) {
      if (!tipPresent ||
          (cellId != null &&
              _dwellCellId != null &&
              cellId != _dwellCellId) ||
          (cellId == null && tipPresent)) {
        // Tip left or moved to another cell before lock finished.
        phase = HandsFreePhase.reading;
        _dwellMs = 0;
        _dwellCellId = null;
        _absentMs = tipPresent ? 0 : 0;
        return [
          const HandsFreeAbortCountdown(),
          const HandsFreeStatus('Finger moved — dwell reset'),
        ];
      }
    }
    return const [];
  }

  List<HandsFreeAction> _pageHunt({
    required int dt,
    required bool tipPresent,
  }) {
    if (tipPresent) {
      _stableNoTipMs = 0;
      return const [HandsFreeStatus('Clear the page for auto scan')];
    }
    _stableNoTipMs += dt;
    if (_stableNoTipMs >= config.pageStableMs) {
      phase = HandsFreePhase.pageCountdown;
      _stableNoTipMs = 0;
      return [
        const HandsFreeStatus('Page countdown…'),
        HandsFreePlayCountdown(
          count: config.lockBeepCount,
          intervalMs: config.lockBeepIntervalMs,
          kind: HandsFreeCountdownKind.page,
        ),
      ];
    }
    return [
      HandsFreeStatus(
        'Hold still… ${(config.pageStableMs - _stableNoTipMs).clamp(0, config.pageStableMs)} ms',
      ),
    ];
  }

  List<HandsFreeAction> _readingOrDwelling({
    required int dt,
    required bool tipPresent,
    int? cellId,
  }) {
    if (!tipPresent) {
      _dwellMs = 0;
      _dwellCellId = null;
      phase = HandsFreePhase.reading;
      _absentMs += dt;
      if (_absentMs >= config.moveAbsentMs) {
        _absentMs = 0;
        phase = HandsFreePhase.softRescan;
        return [
          const HandsFreeStatus('Refreshing page map…'),
          HandsFreePlayCountdown(
            count: config.softRescanBeepCount,
            intervalMs: config.lockBeepIntervalMs,
            kind: HandsFreeCountdownKind.softRescan,
          ),
        ];
      }
      return [
        HandsFreeStatus('Place finger on a cell'),
      ];
    }

    _absentMs = 0;

    // Tip present but no cheap cell id yet — stay in reading, do not dwell.
    if (cellId == null) {
      _dwellMs = 0;
      _dwellCellId = null;
      phase = HandsFreePhase.reading;
      return const [HandsFreeStatus('Finger seen — align over a cell')];
    }

    // Suppress re-announce of the cell we just spoke until tip leaves
    // (handled in cooldown). If somehow still in reading with same id:
    if (_announcedCellId != null && cellId == _announcedCellId) {
      phase = HandsFreePhase.cooldown;
      _cooldownCellId = cellId;
      return const [HandsFreeStatus('Move to another cell')];
    }

    if (_dwellCellId != cellId) {
      _dwellCellId = cellId;
      _dwellMs = 0;
      phase = HandsFreePhase.dwelling;
      return [HandsFreeStatus('Dwelling on cell $cellId…')];
    }

    _dwellMs += dt;
    phase = HandsFreePhase.dwelling;

    if (_dwellMs >= config.lockStartMs) {
      phase = HandsFreePhase.lockBeeps;
      return [
        HandsFreeStatus('Hold still — capturing…'),
        HandsFreePlayCountdown(
          count: config.lockBeepCount,
          intervalMs: config.lockBeepIntervalMs,
          kind: HandsFreeCountdownKind.lock,
        ),
      ];
    }

    final left = config.dwellMs - _dwellMs;
    return [HandsFreeStatus('Dwelling… ${(left / 1000).toStringAsFixed(1)} s')];
  }

  List<HandsFreeAction> _cooldown({
    required bool tipPresent,
    int? cellId,
  }) {
    final left =
        !tipPresent || cellId == null || cellId != _cooldownCellId;
    if (left) {
      _announcedCellId = null;
      _cooldownCellId = null;
      _dwellMs = 0;
      _dwellCellId = null;
      _absentMs = 0;
      phase = HandsFreePhase.reading;
      return const [HandsFreeStatus('Ready — place finger on a cell')];
    }
    return const [HandsFreeStatus('Move to another cell')];
  }

  /// Host finished playing a countdown (page / lock / soft rescan).
  List<HandsFreeAction> onCountdownFinished(HandsFreeCountdownKind kind) {
    if (paused) return const [];
    switch (kind) {
      case HandsFreeCountdownKind.page:
        if (phase != HandsFreePhase.pageCountdown) return const [];
        phase = HandsFreePhase.buildingMap;
        return [
          const HandsFreeStatus('Scanning page…'),
          const HandsFreeRequestPrescan(),
        ];
      case HandsFreeCountdownKind.lock:
        if (phase != HandsFreePhase.lockBeeps) return const [];
        phase = HandsFreePhase.announce;
        return [
          const HandsFreeStatus('Reading character…'),
          const HandsFreeRequestFingerCapture(),
        ];
      case HandsFreeCountdownKind.softRescan:
        if (phase != HandsFreePhase.softRescan) return const [];
        return [
          const HandsFreeStatus('Updating page map…'),
          const HandsFreeRequestSoftRescan(),
        ];
    }
  }

  List<HandsFreeAction> onPrescanFinished({required bool success}) {
    if (!success) {
      phase = HandsFreePhase.pageHunt;
      _stableNoTipMs = 0;
      return [
        const HandsFreeStatus('Page scan failed — try again'),
        const HandsFreeSpeak(
          'Page scan failed. Hold the page steady with good lighting.',
        ),
      ];
    }
    phase = HandsFreePhase.reading;
    _resetTrackers();
    _announcedCellId = null;
    _cooldownCellId = null;
    return [
      const HandsFreeStatus('Page ready — place your finger on a cell'),
      const HandsFreeSpeak(
        'පිටුව ස්කෑන් කර අවසන්. දැන් ඔබේ ඇඟිල්ල අකුරක් මත තබන්න.',
        sinhala: true,
      ),
    ];
  }

  List<HandsFreeAction> onAnnounceFinished({int? announcedCellId}) {
    _announcedCellId = announcedCellId;
    _cooldownCellId = announcedCellId;
    _dwellMs = 0;
    _dwellCellId = null;
    if (announcedCellId != null) {
      phase = HandsFreePhase.cooldown;
      return const [HandsFreeStatus('Move to another cell')];
    }
    phase = HandsFreePhase.reading;
    return const [HandsFreeStatus('No character — try again')];
  }

  List<HandsFreeAction> onSoftRescanFinished({required bool success}) {
    if (!success) {
      phase = HandsFreePhase.reading;
      _absentMs = 0;
      return [
        const HandsFreeStatus('Page refresh failed'),
        const HandsFreeSpeak('Could not refresh the page map. Keep reading.'),
      ];
    }
    phase = HandsFreePhase.reading;
    _resetTrackers();
    _announcedCellId = null;
    _cooldownCellId = null;
    return [
      const HandsFreeStatus('Page map updated'),
      const HandsFreeSpeak(
        'Page updated. Place your finger on a letter.',
      ),
    ];
  }

  int _dtMs(DateTime now) {
    final last = _lastSampleAt;
    if (last == null) return 0;
    final ms = now.difference(last).inMilliseconds;
    // Cap so a long GC pause does not skip dwell logic.
    if (ms < 0) return 0;
    if (ms > 1000) return 1000;
    return ms;
  }
}
