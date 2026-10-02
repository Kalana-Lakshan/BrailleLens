import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../models/braille_cell.dart';
import '../services/audio_service.dart';
import '../services/camera_source.dart';
import '../services/cell_hit_test.dart';
import '../services/covered_cell_service.dart';
import '../services/fingertip_onnx_service.dart';
import '../services/glass_device_service.dart';
import '../services/hands_free_learning_session.dart';
import '../services/prescan_bridge.dart';
import '../services/screen_command_listener.dart';
import '../services/tip_dwell_tracker.dart';
import '../services/stt_onnx_service.dart';
import '../services/voice_capture_service.dart';
import '../utils/answer_match.dart';
import '../utils/sinhala_prompts.dart';
import '../theme/app_theme.dart';
import '../utils/image_decode.dart';
import '../utils/sinhala_phonetics.dart';
import '../widgets/frozen_image_view.dart';
import '../widgets/tap_fingertip_dialog.dart';

enum _TestStage { prescan, fingerResult }

/// Testing Mode: Learning Mode's flow end to end, with one difference at the
/// end.
///
/// Same as [LearningScreen] (kept as a separate copy so Learning stays
/// untouched): auto page baseline when no tip is present, ~3 s cell dwell
/// with lock earcons, then geometry lookup via
/// [CoveredCellService.resolveAligned]. Yellow button / glasses frame button
/// remain manual overrides.
///
/// Instead of announcing the letter, the learner is asked to name it. The
/// answer is recorded (glasses mic when connected, phone otherwise),
/// transcribed by the on-device STT model (sinhala_mms_ctc_quantized.onnx) and
/// compared with the letter the CNN detected under the finger; the learner
/// is told whether it was right.
class TestingScreen extends StatefulWidget {
  final AudioService audioService;

  const TestingScreen({super.key, required this.audioService});

  @override
  State<TestingScreen> createState() => _TestingScreenState();
}

class _TestingScreenState extends State<TestingScreen> {
  final CameraSourceController _camera = CameraSourceController();
  final PrescanBridge _prescanBridge = PrescanBridge();
  final FingertipOnnxService _fingertipOnnx = FingertipOnnxService();
  final CoveredCellService _coveredCell = CoveredCellService();
  final HandsFreeLearningSession _session = HandsFreeLearningSession();

  StreamSubscription<GlassButtonClicked>? _glassButtonSub;
  Timer? _sampleTimer;
  bool _sampleInFlight = false;

  /// Consecutive sample ticks that got no frame from the camera.
  int _missedFrames = 0;
  /// Bumped to cancel the running countdown; each countdown owns one value.
  int _countdownGen = 0;
  bool _handsFreeEnabled = true;

  static const _repeatVoiceAfter = Duration(seconds: 12);

  /// How long a finger result stays up before the CellMap view returns.
  static const _resultHold = Duration(seconds: 4);
  Timer? _resultClearTimer;
  String? _lastVoiced;
  DateTime _lastVoicedAt = DateTime.fromMillisecondsSinceEpoch(0);

  /// Dwell identity from fingertip stillness (see [TipDwellTracker]).
  final TipDwellTracker _tipDwell = TipDwellTracker();

  /// English "back" / "retry" / "help" while the screen is idle.
  late final ScreenCommandListener _commands = ScreenCommandListener(
    audio: widget.audioService,
    canListen: _commandsCanListen,
    onCommand: _onCommand,
  );

  _TestStage _stage = _TestStage.prescan;
  bool _cameraReady = false;
  bool _busy = false;
  bool _isExiting = false;
  String? _statusLine;

  Uint8List? _prescanJpeg;
  CellMap? _cellMap;
  Uint8List? _fingerJpeg;
  CoveredCellResult? _covered;
  FingertipDetection? _fingertip;

  // ── Quiz state ─────────────────────────────────────────────────────────────

  final SttOnnxService _stt = SttOnnxService.instance;
  final VoiceCaptureService _voice = VoiceCaptureService();
  bool _sttReady = false;

  /// Recordings per round before giving up on hearing the learner.
  static const _listenAttempts = 2;

  /// True from the "name the letter" prompt until the answer is scored; the
  /// capture control is ignored meanwhile so a press cannot start a new round.
  bool _answering = false;
  bool _listening = false;

  /// This round's letter under the finger and its spoken name (ක → කයන්න).
  /// Hidden on screen until the answer is scored.
  String _targetChar = '';
  String _targetPhonetic = '';
  String? _spokenResult;
  bool? _lastCorrect; // null = this round not scored yet
  int _correct = 0;
  int _total = 0;

  @override
  void initState() {
    super.initState();
    _boot();
  }

  Future<void> _boot() async {
    await _camera.initialize();
    await _camera.configureCapturePolicy();

    _glassButtonSub = GlassDeviceService.instance.buttonClicks.listen((_) {
      if (!_camera.usingGlasses) return;
      widget.audioService.hapticLight();
      _onCapturePressed();
    });
    _camera.addListener(_onCameraSourceChanged);

    final cnnReady = await PrescanBridge.ensureOnDeviceReady();
    final tipReady = await _fingertipOnnx.initialize();
    // Speech is only needed at the end of a round, but loading it here keeps
    // the first answer from stalling on a cold ~79 MB model load.
    final sttReady = await _stt.initialize();
    if (!mounted) return;
    setState(() {
      _cameraReady = _camera.isReady;
      _sttReady = sttReady;
      _statusLine = [
        'Camera: ${_camera.label}',
        cnnReady ? 'CNN: braille_cnn.onnx' : 'CNN failed',
        tipReady
            ? 'YOLO: ${_fingertipOnnx.loadedAsset?.split('/').last}'
            : 'YOLO failed — tap fingertip',
        sttReady
            ? 'STT: ${_stt.loadedAsset?.split('/').last}'
            : 'STT unavailable',
        'Hands-free: on',
      ].join(' · ');
    });
    if (!sttReady) {
      // Not fatal: rounds still resolve the letter and read it out, they
      // just cannot score a spoken answer.
      debugPrint('[Testing] STT unavailable: ${_stt.lastError}');
    }

    for (final (failed, warning) in [
      (!_cameraReady, SinhalaPrompts.noCamera),
      (!cnnReady, SinhalaPrompts.readerFailed),
      (!tipReady, SinhalaPrompts.fingerDetectorFailed),
      (!sttReady, SinhalaPrompts.sttFailed),
    ]) {
      if (failed) await widget.audioService.speakSinhala(warning);
    }
    await widget.audioService.speakSinhala(SinhalaPrompts.enterTesting);
    await widget.audioService.speakSinhala(SinhalaPrompts.prescan);

    if (!mounted || !_cameraReady) return;
    _session.start();
    _dispatch(_session.onSample(
      now: DateTime.now(),
      tipPresent: false,
      cellId: null,
    ));
    _startSampleLoop();
    _commands.start();
  }

  void _onCameraSourceChanged() {
    if (!mounted) return;
    unawaited(_camera.configureCapturePolicy());
    setState(() => _cameraReady = _camera.isReady);
  }

  String get _captureHint => _camera.usingGlasses
      ? 'press the button on your glasses'
      : 'tap capture';

  // ── Hands-free sampling ────────────────────────────────────────────────────

  void _startSampleLoop() {
    _sampleTimer?.cancel();
    // ~2.5 Hz — tip YOLO + optional cheap cell map; full resolve only on fire.
    _sampleTimer = Timer.periodic(const Duration(milliseconds: 400), (_) {
      unawaited(_onSampleTick());
    });
  }

  void _stopSampleLoop() {
    _sampleTimer?.cancel();
    _sampleTimer = null;
  }

  Future<void> _onSampleTick() async {
    if (!_handsFreeEnabled ||
        _isExiting ||
        !_cameraReady ||
        _sampleInFlight ||
        _session.paused) {
      return;
    }
    // Do not steal frames while a capture pipeline is running.
    if (_busy) return;
    final phase = _session.phase;
    if (phase == HandsFreePhase.buildingMap ||
        phase == HandsFreePhase.announce) {
      return;
    }

    _sampleInFlight = true;
    try {
      final jpeg = await _camera.captureSampleJpeg();
      if (!mounted) return;
      if (jpeg == null) {
        // The phone camera always returns a photo; the glasses' video feed
        // can have no frame (stream down or mid-reconnect). Without a sample
        // the session cannot advance, so make the stall visible instead of
        // returning silently.
        _missedFrames++;
        if (_missedFrames == 5 || _missedFrames % 25 == 0) {
          debugPrint('[Testing] no frame x$_missedFrames: '
              '${_camera.source?.lastError}');
          setState(() => _statusLine = _camera.usingGlasses
              ? 'Waiting for the glasses video…'
              : 'Waiting for the camera…');
        }
        return;
      }
      if (_missedFrames >= 5) {
        debugPrint('[Testing] frames back after $_missedFrames misses');
      }
      _missedFrames = 0;

      final tip = await _fingertipOnnx.detect(jpeg);
      int? cellId;
      final map = _cellMap;
      if (tip == null) {
        _tipDwell.lost();
      } else if (map != null &&
          (phase == HandsFreePhase.reading ||
              phase == HandsFreePhase.dwelling ||
              phase == HandsFreePhase.lockBeeps ||
              phase == HandsFreePhase.cooldown ||
              phase == HandsFreePhase.pageCountdown)) {
        cellId = _tipDwell.track(tip.contactPoint, tip.imageWidth, map);
      }

      if (!mounted) return;
      await _dispatch(_session.onSample(
        now: DateTime.now(),
        tipPresent: tip != null,
        cellId: cellId,
      ));
    } finally {
      _sampleInFlight = false;
    }
  }


  /// "Move right / left / up / down" from [tipInPrescan] to the closest
  /// letter cell, or null when there is none nearby or the tip is on it.
  String? _directionToNearestCell(Offset? tipInPrescan) {
    final map = _cellMap;
    if (tipInPrescan == null || map == null) return null;
    // Wider than the hit test's reach: a tip that missed by a cell or two is
    // exactly the case that needs steering.
    final target =
        CellHitTest.nearestCell(tipInPrescan, map, withinCells: 3.0);
    if (target == null) return null;
    return SinhalaPrompts.direction(
      tipInPrescan,
      target.center,
      CellHitTest.medianCellWidth(map),
    );
  }

  Future<void> _dispatch(List<HandsFreeAction> actions) async {
    for (final a in actions) {
      if (!mounted || _isExiting) return;
      switch (a) {
        case HandsFreeStatus(:final message, :final speech):
          setState(() => _statusLine = message);
          if (speech != null) await _voiceStatus(speech);
        case HandsFreeSpeak(:final text, :final sinhala):
          final replacement = SinhalaPrompts.sessionLine(text);
          if (replacement != null) {
            await widget.audioService.speakSinhala(replacement);
          } else if (sinhala) {
            await widget.audioService.speakSinhala(text);
          } else {
            await widget.audioService.speak(text);
          }
        case HandsFreePlayCountdown(
            :final count,
            :final intervalMs,
            :final kind
          ):
          _commands.pause();
          // Not awaited: sampling must continue so a returning tip can abort.
          unawaited(_runCountdown(count, intervalMs, kind));
        case HandsFreeAbortCountdown():
          _countdownGen++;
        case HandsFreeRequestPrescan():
          await _autoPrescan();
        case HandsFreeRequestFingerCapture():
          await _autoFinger();
        case HandsFreeRequestSoftRescan():
          await _autoSoftRescan();
      }
    }
  }

  /// Speaks a status line, skipping a back-to-back repeat of the same phrase
  /// within [_repeatVoiceAfter] (the sampler re-emits statuses every tick).
  Future<void> _voiceStatus(String text) async {
    final now = DateTime.now();
    if (text == _lastVoiced &&
        now.difference(_lastVoicedAt) < _repeatVoiceAfter) {
      return;
    }
    _lastVoiced = text;
    _lastVoicedAt = now;
    final sinhala = SinhalaPrompts.sessionLine(text) ??
        (_isSinhalaScript(text) ? text : null);
    if (sinhala != null) {
      await widget.audioService.speakSinhala(sinhala);
    } else {
      debugPrint('[Testing] no Sinhala for "$text"');
      await widget.audioService.speak(text);
    }
  }

  static bool _isSinhalaScript(String text) =>
      RegExp('[\u0D80-\u0DFF]').hasMatch(text);

  Future<void> _voiceCaptureFailed() =>
      widget.audioService.speakSinhala(SinhalaPrompts.retryError);

  Future<void> _runCountdown(
    int count,
    int intervalMs,
    HandsFreeCountdownKind kind,
  ) async {
    final gen = ++_countdownGen;
    final ok = await widget.audioService.playCountdownBeeps(
      count,
      gap: Duration(milliseconds: intervalMs),
      shouldAbort: () => gen != _countdownGen || _isExiting,
    );
    if (!mounted || _isExiting) return;
    if (!ok || gen != _countdownGen) return;
    await _dispatch(_session.onCountdownFinished(kind));
  }

  /// Waits for an in-flight sample capture to return before a full capture:
  /// the camera plugin rejects a second takePicture while one is pending.
  /// Bounded, so a stuck sample cannot hold the round forever.
  Future<void> _waitForSampleIdle() async {
    final deadline = DateTime.now().add(const Duration(seconds: 3));
    while (_sampleInFlight && DateTime.now().isBefore(deadline)) {
      await Future<void>.delayed(const Duration(milliseconds: 50));
    }
  }

  Future<void> _autoPrescan() async {
    _session.pause();
    _commands.pause();
    await _waitForSampleIdle();
    setState(() {
      _busy = true;
      _statusLine = 'Scanning the page for Braille cells…';
    });
    final jpeg = await _camera.captureJpeg();
    if (jpeg == null) {
      setState(() {
        _busy = false;
        _statusLine = _camera.source?.lastError ?? 'Camera capture failed';
      });
      await _voiceCaptureFailed();
      await _dispatch(_session.onPrescanFinished(success: false));
      _session.resume();
      return;
    }
    final ok = await _runPrescanFromJpeg(jpeg, speakOnSuccess: false);
    await _dispatch(_session.onPrescanFinished(success: ok));
    _session.resume();
  }

  Future<void> _autoFinger() async {
    _session.pause();
    _commands.pause();
    await _waitForSampleIdle();
    setState(() {
      _busy = true;
      _statusLine = 'Detecting fingertip…';
    });
    final jpeg = await _camera.captureJpeg();
    if (jpeg == null) {
      setState(() {
        _busy = false;
        _statusLine = _camera.source?.lastError ?? 'Camera capture failed';
      });
      await _voiceCaptureFailed();
      await _dispatch(_session.onAnnounceFinished(announcedCellId: null));
      _session.resume();
      return;
    }
    final cellId = await _runFingerLookupFromJpeg(
      jpeg,
      allowTapFallback: false,
      playSuccessEarcon: true,
    );
    await _dispatch(_session.onAnnounceFinished(announcedCellId: cellId));
    _session.resume();
  }

  Future<void> _autoSoftRescan() async {
    _session.pause();
    _commands.pause();
    await _waitForSampleIdle();
    _resultClearTimer?.cancel();
    setState(() {
      _busy = true;
      _statusLine = 'Refreshing page map…';
      _fingerJpeg = null;
      _covered = null;
      _fingertip = null;
    });
    final jpeg = await _camera.captureJpeg();
    if (jpeg == null) {
      setState(() {
        _busy = false;
        _statusLine = _camera.source?.lastError ?? 'Camera capture failed';
      });
      await _voiceCaptureFailed();
      await _dispatch(_session.onSoftRescanFinished(success: false));
      _session.resume();
      return;
    }
    // Soft rescan must be hand-free; if a tip is visible, skip.
    final tip = await _fingertipOnnx.detect(jpeg);
    if (tip != null) {
      setState(() {
        _busy = false;
        _statusLine = 'Finger still visible — refresh skipped';
      });
      await _voiceStatus(SinhalaPrompts.fingerStillOnPage);
      await _dispatch(_session.onSoftRescanFinished(success: false));
      _session.resume();
      return;
    }
    final ok = await _runPrescanFromJpeg(jpeg, speakOnSuccess: false);
    await _dispatch(_session.onSoftRescanFinished(success: ok));
    _session.resume();
  }

  // ── Shared pipeline (manual + auto) ────────────────────────────────────────

  Future<void> _onCapturePressed() async {
    if (_busy || _answering || _isExiting) return;
    _countdownGen++;
    _session.pause();
    try {
      if (_stage == _TestStage.prescan) {
        await _capturePrescanManual();
      } else {
        await _captureFingerManual();
      }
    } finally {
      if (_handsFreeEnabled && mounted && !_isExiting) {
        if (_cellMap != null) {
          // After manual map build, sit in reading; after finger, cooldown-like.
          if (_session.phase == HandsFreePhase.pageHunt ||
              _session.phase == HandsFreePhase.pageCountdown ||
              _session.phase == HandsFreePhase.buildingMap) {
            if (_cellMap != null) {
              _session.onPrescanFinished(success: true);
            }
          }
        }
        _session.resume();
      }
    }
  }

  Future<void> _capturePrescanManual() async {
    if (_busy) return;
    setState(() {
      _busy = true;
      _statusLine = 'Scanning the page for Braille cells…';
    });
    final jpeg = await _camera.captureJpeg();
    if (jpeg == null) {
      setState(() {
        _busy = false;
        _statusLine = _camera.source?.lastError ?? 'Camera capture failed';
      });
      await widget.audioService.speakSinhala(SinhalaPrompts.retryError);
      return;
    }
    await _runPrescanFromJpeg(jpeg, speakOnSuccess: true);
  }

  Future<void> _captureFingerManual() async {
    if (_busy || _cellMap == null) return;
    setState(() {
      _busy = true;
      _statusLine = 'Detecting fingertip…';
    });
    final jpeg = await _camera.captureJpeg();
    if (jpeg == null) {
      setState(() {
        _busy = false;
        _statusLine = _camera.source?.lastError ?? 'Camera capture failed';
      });
      await widget.audioService.speakSinhala(SinhalaPrompts.retryError);
      return;
    }
    await _runFingerLookupFromJpeg(
      jpeg,
      allowTapFallback: true,
      playSuccessEarcon: false,
    );
  }

  /// Returns true when a non-empty CellMap was stored.
  Future<bool> _runPrescanFromJpeg(
    Uint8List jpeg, {
    required bool speakOnSuccess,
  }) async {
    try {
      final map = await _prescanBridge.prescanPage(
        jpeg,
        onProgress: (done, total) {
          if (!mounted) return;
          setState(() => _statusLine = 'Classifying cells $done / $total…');
        },
      );
      if (map.cells.isEmpty) {
        throw Exception('No cells detected');
      }
      final decoded = decodeUpright(jpeg);
      final w = decoded?.width ?? map.imageWidth;
      final h = decoded?.height ?? map.imageHeight;
      final fixed = CellMap(cells: map.cells, imageWidth: w, imageHeight: h);

      if (!mounted) return false;
      _resultClearTimer?.cancel();
      setState(() {
        _prescanJpeg = jpeg;
        _cellMap = fixed;
        _stage = _TestStage.fingerResult;
        _fingerJpeg = null;
        _covered = null;
        _fingertip = null;
        _busy = false;
        _statusLine =
            '${fixed.cells.length} cells found · place a finger, then $_captureHint';
      });
      await _voiceStatus(SinhalaPrompts.cellsFound(fixed.cells.length));
      if (speakOnSuccess) {
        await widget.audioService.speakSinhala(SinhalaPrompts.pageReady);
      }
      return true;
    } on PrescanUnavailableException catch (e) {
      if (!mounted) return false;
      setState(() {
        _busy = false;
        _statusLine = e.message;
      });
      if (speakOnSuccess) {
        await widget.audioService.speakSinhala(SinhalaPrompts.retryError);
      }
      return false;
    } catch (e) {
      if (!mounted) return false;
      setState(() {
        _busy = false;
        _statusLine = 'Prescan error: $e';
      });
      return false;
    }
  }

  /// Returns the announced CellMap id, or null on miss.
  Future<int?> _runFingerLookupFromJpeg(
    Uint8List jpeg, {
    required bool allowTapFallback,
    required bool playSuccessEarcon,
  }) async {
    FingertipDetection? tip = await _fingertipOnnx.detect(jpeg);
    if (tip == null && allowTapFallback) {
      tip = await _promptTapFingertip(jpeg);
    }

    if (tip == null) {
      if (!mounted) return null;
      setState(() {
        _busy = false;
        _statusLine = 'No fingertip — tap on your finger tip on screen';
      });
      if (!allowTapFallback) {
        await widget.audioService.speakSinhala(SinhalaPrompts.retryError);
      }
      return null;
    }

    if (mounted) {
      setState(() => _statusLine = 'Aligning this frame with the scanned page…');
    }
    final liveCells = await _prescanBridge.detectCellBoxes(jpeg);

    final result = await _coveredCell.resolveAligned(
      fingerJpeg: jpeg,
      tipInFingerImage: tip.contactPoint,
      cellMap: _cellMap!,
      fingerImageWidth: tip.imageWidth,
      fingerImageHeight: tip.imageHeight,
      fingerFrameCells: liveCells,
      fingertipBox: tip.box,
      prescanJpeg: _prescanJpeg,
    );

    // Indicators ("[IND-A]") and blank cells have no spoken name, so there
    // is nothing to quiz on; they are reported and the learner moves on.
    final quizzable = result.hasHit && isNameableSinhala(result.headline);

    if (!mounted) return null;
    setState(() {
      _fingerJpeg = jpeg;
      _fingertip = tip;
      _covered = result;
      _busy = false;
      _statusLine = !result.hasHit
          ? result.subtitle
          : quizzable
              // The quiz must not print the answer before it is given.
              ? 'Cell found · name the letter'
              : '${result.headline} · not a letter — move on';
    });

    if (quizzable) {
      // Hands-free dwell is paused by the caller for the whole round, so the
      // sampler cannot start a new capture while the learner answers.
      await _askAndScore(result.headline);
      _scheduleResultClear();
      return result.cell?.id;
    }

    if (result.hasHit) {
      if (playSuccessEarcon) {
        await widget.audioService.playSuccessTone();
        await widget.audioService.hapticLight();
      }
      await widget.audioService.speakSinhala(SinhalaPrompts.notALetter);
      return result.cell?.id;
    }

    await widget.audioService.speakSinhala(SinhalaPrompts.retryError);
    // Raw-ratio mapping ('scale'): a direction from it would be a guess.
    if (!result.alignMode.startsWith('scale')) {
      final hint = _directionToNearestCell(result.tipInPrescan);
      if (hint != null) await widget.audioService.speakSinhala(hint);
    }
    return null;
  }

  // ── Quiz: prompt → record → transcribe → score ─────────────────────────────

  /// Asks the learner to name [target], records the answer, transcribes it
  /// with the on-device STT model and compares it with the letter the CNN
  /// resolved under the finger (the letter itself or its Sinhala name).
  Future<void> _askAndScore(String target) async {
    setState(() {
      _answering = true;
      _targetChar = target;
      _targetPhonetic = sinhalaLetterName(target);
      _spokenResult = null;
      _lastCorrect = null;
    });
    _commands.pause();
    try {
      await widget.audioService.speakSinhala(SinhalaPrompts.askLetter);
      if (!mounted || _isExiting) return;

      if (!_sttReady || !_stt.canRecognise(target)) {
        // Nothing to score against: either the speech model is missing, or
        // this is a sign (ං, ඃ, a vowel sign) outside the letters the model
        // knows. Read the answer out rather than fail the learner.
        setState(() => _statusLine = _sttReady
            ? '$target is not a letter the speech model can check'
            : 'Speech model missing — add sinhala_mms_small_int8.onnx + '
                'letters.json to assets/models/');
        await widget.audioService
            .speakSinhala(SinhalaPrompts.reveal(_targetPhonetic));
        return;
      }

      // Silence or a too-short clip is not a wrong answer: ask once more
      // before giving up, and never score an answer that was not heard.
      SpokenLetter? answer;
      VoiceCapture? capture;
      for (var attempt = 1; attempt <= _listenAttempts; attempt++) {
        if (attempt > 1) {
          await widget.audioService.speakSinhala(SinhalaPrompts.didNotHear);
          if (!mounted || _isExiting) return;
        }
        setState(() {
          _listening = true;
          _statusLine = 'Listening…';
        });
        // Awaiting the prompt before opening the mic keeps the app's own
        // voice out of the recording.
        await widget.audioService.playMicOpen();
        capture = await _voice.record();
        await widget.audioService.playMicClose();
        if (!mounted || _isExiting) return;

        setState(() {
          _listening = false;
          _statusLine = capture!.hasAudio
              ? 'Checking your answer…'
              : (capture.error ?? 'No audio recorded');
        });
        if (!capture.hasAudio) continue;

        // The model runs off the UI isolate, so frames keep rendering
        // while the answer is classified.
        final result = await _stt.classify(capture.samples);
        if (!mounted || _isExiting) return;
        if (result != null && result.heard) {
          answer = result;
          break;
        }
      }

      if (answer == null) {
        setState(() => _spokenResult = '');
        await widget.audioService.speakSinhala(SinhalaPrompts.gaveUpListening);
        await widget.audioService
            .speakSinhala(SinhalaPrompts.reveal(_targetPhonetic));
        return;
      }

      // Letters that sound the same (ණ/න, ඛ/ක…) count as the same answer:
      // no microphone can tell them apart.
      final spoken = answer.letter;
      final matched = sinhalaLettersSoundAlike(spoken, target);
      final percent = (answer.confidence * 100).round();
      debugPrint('[Testing] target=$target heard=$spoken ($percent%) '
          'matched=$matched in ${answer.elapsed.inMilliseconds} ms');
      setState(() {
        _spokenResult = '$spoken ($percent%)';
        _lastCorrect = matched;
        _total++;
        if (matched) _correct++;
        _statusLine = 'You said: $spoken · $percent% sure · '
            '${capture?.source.name} mic';
      });

      if (matched) {
        await widget.audioService.hapticHeavy();
        await widget.audioService.playSuccessTone();
        await widget.audioService
            .speakSinhala(SinhalaPrompts.correct(_targetPhonetic));
      } else {
        await widget.audioService.hapticError();
        await widget.audioService.playErrorTone();
        // "You said <name of the heard letter>, but the correct letter…"
        await widget.audioService.speakSinhala(SinhalaPrompts.incorrect(
            sinhalaLetterName(spoken), _targetPhonetic));
      }
    } finally {
      if (mounted) {
        setState(() {
          _answering = false;
          _listening = false;
        });
      }
    }
  }

  void _scheduleResultClear() {
    _resultClearTimer?.cancel();
    _resultClearTimer = Timer(_resultHold, () {
      if (!mounted || _isExiting) return;
      setState(() {
        _fingerJpeg = null;
        _fingertip = null;
        _covered = null;
      });
    });
  }

  Future<FingertipDetection?> _promptTapFingertip(Uint8List jpeg) async {
    final decoded = decodeUpright(jpeg);
    if (decoded == null) return null;

    final tap = await showDialog<Offset>(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => TapFingertipDialog(jpeg: jpeg),
    );
    if (tap == null) return null;

    return FingertipDetection(
      contactPoint: tap,
      box: Rect.fromCenter(center: tap, width: 48, height: 48),
      confidence: 1.0,
      imageWidth: decoded.width,
      imageHeight: decoded.height,
    );
  }

  /// Commands are heard only while the learner is between letters or
  /// hunting for the page — not during a prompt, countdown or capture.
  bool _commandsCanListen() {
    if (_isExiting || _busy || _answering || _session.paused) {
      return false;
    }
    // With the glasses as a Bluetooth headset, starting the recogniser tears
    // the headset audio link down and up, which drops the video feed the
    // finger sampling reads from — the flow then stalls at the finger step.
    // The frame button still captures.
    if (_camera.usingGlasses) return false;
    final phase = _session.phase;
    return phase == HandsFreePhase.pageHunt ||
        phase == HandsFreePhase.reading ||
        phase == HandsFreePhase.cooldown;
  }

  Future<void> _onCommand(ScreenCommand command) async {
    switch (command) {
      case ScreenCommand.back:
        await _exit();
      case ScreenCommand.retry:
        _rescan();
      case ScreenCommand.capture:
        await _onCapturePressed();
      case ScreenCommand.help:
        await widget.audioService.speakSinhala(SinhalaPrompts.testingHelp);
    }
  }

  void _rescan() {
    _countdownGen++;
    _resultClearTimer?.cancel();
    _tipDwell.lost();
    setState(() {
      _stage = _TestStage.prescan;
      _prescanJpeg = null;
      _cellMap = null;
      _fingerJpeg = null;
      _covered = null;
      _fingertip = null;
      _statusLine = null;
      // The score covers the whole test, so it survives a rescan.
      _targetChar = '';
      _targetPhonetic = '';
      _spokenResult = null;
      _lastCorrect = null;
    });
    _session.resetToPageHunt();
    // Back to Stage 1.
    widget.audioService.speakSinhala(SinhalaPrompts.prescan);
  }

  Future<void> _exit() async {
    if (_isExiting) return;
    setState(() => _isExiting = true);
    _commands.stop();
    _countdownGen++;
    _handsFreeEnabled = false;
    _stopSampleLoop();
    _session.pause();
    await widget.audioService.stopSpeech();
    await widget.audioService.hapticLight();
    await widget.audioService.speakSinhala(SinhalaPrompts.returningHome);
    if (mounted) Navigator.pop(context);
  }

  @override
  void dispose() {
    _commands.stop();
    _countdownGen++;
    _resultClearTimer?.cancel();
    _handsFreeEnabled = false;
    _stopSampleLoop();
    _glassButtonSub?.cancel();
    _camera.removeListener(_onCameraSourceChanged);
    unawaited(_camera.restoreHardwareShutter());
    _fingertipOnnx.dispose();
    _voice.dispose();
    _camera.dispose();
    // The STT session is app-wide and shared, so it is not disposed here.
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onDoubleTap: _exit,
      child: Scaffold(
        backgroundColor: Colors.black,
        body: Stack(
          fit: StackFit.expand,
          children: [
            _buildImageArea(),
            _buildFramingHint(),
            _buildTopBar(),
            _buildBottomPanel(),
            if (_busy) _buildBusyIndicator(),
          ],
        ),
      ),
    );
  }

  Widget _buildBusyIndicator() {
    return IgnorePointer(
      child: Center(
        child: Container(
          padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(
            color: Colors.black.withValues(alpha: 0.55),
            shape: BoxShape.circle,
          ),
          child: const SizedBox(
            width: 22,
            height: 22,
            child: CircularProgressIndicator(
              strokeWidth: 2.5,
              color: AppTheme.primaryYellow,
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildImageArea() {
    if (_fingerJpeg != null) {
      return FrozenImageView(
        jpeg: _fingerJpeg!,
        fingertip: _fingertip,
      );
    }
    if (_prescanJpeg != null && _cellMap != null) {
      return FrozenImageView(
        jpeg: _prescanJpeg!,
        cellMap: _cellMap,
      );
    }
    return _camera.buildPreview();
  }

  Widget _pill({required Widget child, VoidCallback? onTap}) {
    final content = Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(20),
      ),
      child: child,
    );
    if (onTap == null) return content;
    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(20),
        onTap: onTap,
        child: content,
      ),
    );
  }

  Widget _buildTopBar() {
    final title = _stage == _TestStage.prescan
        ? '1 · SCAN BRAILLE PAGE'
        : (_answering ? '2 · NAME THE LETTER' : '2 · POINT TO A CELL');

    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _pill(
              onTap: _exit,
              child: const Text('Exit',
                  style: TextStyle(
                      color: AppTheme.primaryYellow,
                      fontWeight: FontWeight.bold)),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Center(
                child: _pill(
                  child: Text(
                    title,
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      color: AppTheme.primaryYellow,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1.2,
                      fontSize: 13,
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(width: 8),
            if (_cellMap != null)
              _pill(
                onTap: _busy ? null : _rescan,
                child: const Text('Rescan',
                    style: TextStyle(color: Colors.white)),
              )
            else
              const SizedBox(width: 8),
          ],
        ),
      ),
    );
  }

  Widget _buildFramingHint() {
    if (!(_stage == _TestStage.prescan && _cellMap == null)) {
      return const SizedBox.shrink();
    }
    return Align(
      alignment: Alignment.topCenter,
      child: SafeArea(
        child: Padding(
          padding: const EdgeInsets.only(top: 56),
          child: IgnorePointer(
            child: _pill(
              child: const Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    'CENTER THE PAGE · KEEP FINGERS OUT',
                    style: TextStyle(
                      color: AppTheme.primaryYellow,
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                      letterSpacing: 0.5,
                    ),
                  ),
                  SizedBox(height: 2),
                  Text(
                    'Auto-scan after hold · or capture manually',
                    style: TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  /// Testing Mode's result: the letter stays hidden ("?") until the spoken
  /// answer is scored, then shows the letter, what was heard and the verdict.
  List<Widget> _buildQuizResult() {
    final scored = _lastCorrect != null;
    final color = !scored
        ? AppTheme.primaryYellow
        : (_lastCorrect! ? const Color(0xFF00E5FF) : const Color(0xFFFF5252));
    final waiting = _targetChar.isNotEmpty && !scored && _answering;

    return [
      Text(
        _listening
            ? 'LISTENING… SAY THE LETTER'
            : waiting
                ? 'NAME THE LETTER UNDER YOUR FINGER'
                : (scored ? 'LAST ANSWER' : 'READY'),
        style: const TextStyle(
            color: Colors.white70, fontSize: 12, letterSpacing: 1.1),
      ),
      const SizedBox(height: 4),
      Text(
        _targetChar.isEmpty ? '—' : (scored || !_answering ? _targetChar : '?'),
        style: TextStyle(
            fontSize: 56, fontWeight: FontWeight.bold, color: color),
      ),
      if (scored && _targetPhonetic.isNotEmpty)
        Text(
          '$_targetPhonetic'
          '${_covered?.hasHit == true ? ' · dots ${_covered!.compactDots}' : ''}',
          textAlign: TextAlign.center,
          style: const TextStyle(color: Colors.white70, fontSize: 16),
        ),
      if (_spokenResult != null) ...[
        const SizedBox(height: 4),
        Text(
          'You said: ${_spokenResult!.isEmpty ? '—' : _spokenResult!}',
          textAlign: TextAlign.center,
          style: const TextStyle(color: Colors.white, fontSize: 18),
        ),
      ],
      if (scored) ...[
        const SizedBox(height: 4),
        Text(
          _lastCorrect! ? 'නිවැරදියි! Correct' : 'වැරදියි — Incorrect',
          style: TextStyle(color: color, fontWeight: FontWeight.w600),
        ),
      ],
      const SizedBox(height: 6),
      Text(
        'Score: $_correct / $_total',
        style: const TextStyle(color: Colors.white70, fontSize: 15),
      ),
    ];
  }

  Widget _buildBottomPanel() {
    return Positioned(
      left: 0,
      right: 0,
      bottom: 0,
      child: Container(
        padding: const EdgeInsets.fromLTRB(20, 12, 20, 20),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.75),
          border: const Border(
              top: BorderSide(color: AppTheme.primaryYellow, width: 2)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (_stage == _TestStage.fingerResult)
              ..._buildQuizResult()
            else
              const Text(
                'Clear the page for auto-scan, or capture',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.white70, fontSize: 15),
              ),
            if (_statusLine != null) ...[
              const SizedBox(height: 8),
              Text(
                _statusLine!,
                textAlign: TextAlign.center,
                style: TextStyle(
                    color: Colors.white.withValues(alpha: 0.45), fontSize: 12),
              ),
            ],
            const SizedBox(height: 16),
            if (!_camera.usingGlasses)
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: (_busy || _answering || !_cameraReady)
                      ? null
                      : _onCapturePressed,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.primaryYellow,
                    foregroundColor: Colors.black,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                  ),
                  child: Text(
                    _stage == _TestStage.prescan
                        ? 'SCAN BRAILLE PAGE'
                        : 'READ MY FINGER & ASK ME',
                    style: const TextStyle(
                        fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                ),
              )
            else
              Text(
                _stage == _TestStage.prescan
                    ? 'Auto-scan when clear · or press glasses button'
                    : 'Hold finger ~3 s · or press glasses button',
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: AppTheme.primaryYellow,
                  fontWeight: FontWeight.w600,
                  fontSize: 15,
                ),
              ),
          ],
        ),
      ),
    );
  }
}
