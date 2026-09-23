import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../models/braille_cell.dart';
import '../services/audio_service.dart';
import '../services/camera_source.dart';
import '../services/glass_device_service.dart';
import '../services/covered_cell_service.dart';
import '../services/fingertip_onnx_service.dart';
import '../services/prescan_bridge.dart';
import '../services/stt_onnx_service.dart';
import '../services/voice_capture_service.dart';
import '../theme/app_theme.dart';
import '../utils/image_decode.dart';
import '../utils/sinhala_phonetics.dart';
import '../widgets/frozen_image_view.dart';
import '../widgets/tap_fingertip_dialog.dart';

enum _TestStage { prescan, quiz }

/// Two-stage Testing Mode, same on-device pipeline as Learning Mode:
/// 1. Capture hand-free page → prescan builds CellMap (yellow boxes).
///    Runs once per page; "Rescan" if the page moves.
/// 2. The learner puts a finger on any cell and captures. The geometry-only
///    hit-test resolves which cell that is (no CNN on the finger photo,
///    where the hand hides the dots), and instead of being told the answer
///    the learner is asked to say it. The spoken answer is recorded — glasses
///    mic when connected, phone mic otherwise — transcribed by the on-device
///    CTC model, and compared against the cell's letter and its Sinhala name.
///    The page map is kept between rounds, so each press tests a new cell.
class TestingScreen extends StatefulWidget {
  final AudioService audioService;

  const TestingScreen({super.key, required this.audioService});

  @override
  State<TestingScreen> createState() => _TestingScreenState();
}

class _TestingScreenState extends State<TestingScreen> {
  /// Glasses when connected, phone camera otherwise.
  final CameraSourceController _camera = CameraSourceController();
  final PrescanBridge _prescanBridge = PrescanBridge();
  final FingertipOnnxService _fingertipOnnx = FingertipOnnxService();
  final CoveredCellService _coveredCell = CoveredCellService();

  /// Frame-button presses, routed to the same entry point as the on-screen
  /// capture control.
  StreamSubscription<GlassButtonClicked>? _glassButtonSub;

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

  final VoiceCaptureService _voice = VoiceCaptureService();
  final SttOnnxService _stt = SttOnnxService.instance;

  /// The letter resolved under the finger this round, and what the learner
  /// should say for it (ක → කයන්න).
  String _targetChar = '';
  String _targetPhonetic = '';
  String? _spokenResult;
  bool _sttReady = false;
  bool _listening = false;

  bool? _lastCorrect; // null = no round evaluated yet this cell map
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

    // With glasses: frame button only (native shutter off → RTSP snapshot).
    // Phone: yellow button; ignore frame-button events if any.
    _glassButtonSub = GlassDeviceService.instance.buttonClicks.listen((_) {
      if (!_camera.usingGlasses) return;
      widget.audioService.hapticLight();
      _onCapturePressed();
    });
    _camera.addListener(_onCameraSourceChanged);

    final cnnReady = await PrescanBridge.ensureOnDeviceReady();
    final tipReady = await _fingertipOnnx.initialize();
    if (!mounted) return;

    // Speech is only needed in stage 2, but loading it here keeps the first
    // round from stalling on a cold model load.
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
        sttReady ? 'STT: ${_stt.loadedAsset?.split('/').last}' : 'STT unavailable',
        if (_camera.usingGlasses) 'Capture: glasses button',
      ].join(' · ');
    });

    if (!cnnReady) {
      await widget.audioService.speak(
        'Testing Mode could not load the character models. Check the model files and restart.',
      );
      return;
    }
    if (!sttReady) {
      // Not fatal: the round still resolves the letter and reads it out, it
      // just cannot score a spoken answer.
      debugPrint('[Testing] STT unavailable: ${_stt.lastError}');
    }

    final trigger = _camera.usingGlasses
        ? 'press the button on your glasses'
        : 'tap capture';
    await widget.audioService.speak(
      'Testing Mode. Stage 1: hold the Braille page still with no finger, '
      'then $trigger. Then put a finger on a letter and $trigger again '
      'and I will ask you to name it.',
    );
  }

  Future<void> _exit() async {
    if (_isExiting) return;
    setState(() => _isExiting = true);
    await widget.audioService.stopSpeech();
    await widget.audioService.hapticLight();
    await widget.audioService.speak('Returning to main menu.');
    if (mounted) Navigator.pop(context);
  }

  /// The camera source swapped underneath us (glasses connected or dropped).
  void _onCameraSourceChanged() {
    if (!mounted) return;
    unawaited(_camera.configureCapturePolicy());
    setState(() => _cameraReady = _camera.isReady);
  }

  /// Single capture entry point shared by the on-screen button and the
  /// glasses frame button.
  Future<void> _onCapturePressed() async {
    if (_busy || _isExiting) return;
    if (_stage == _TestStage.prescan) {
      await _capturePrescan();
    } else {
      await _checkFinger();
    }
  }

  // ── Stage 1 ──────────────────────────────────────────────────────────────────

  Future<void> _capturePrescan() async {
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
      await widget.audioService.speak('Capture failed. Try again.');
      return;
    }

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

      setState(() {
        _prescanJpeg = jpeg;
        _cellMap = fixed;
        _stage = _TestStage.quiz;
        _busy = false;
      });
      // "Page scan complete. Now place your finger on a letter."
      await widget.audioService.speakSinhala(
        'පිටුව ස්කෑන් කර අවසන්. දැන් ඔබේ ඇඟිල්ල අකුරක් මත තබන්න.',
      );
    } on PrescanUnavailableException catch (e) {
      setState(() {
        _busy = false;
        _statusLine = e.message;
      });
      await widget.audioService.speak(
        'Page scan failed. Hold the page steady with good lighting and try again.',
      );
    } catch (e) {
      setState(() {
        _busy = false;
        _statusLine = 'Prescan error: $e';
      });
    }
  }

  void _rescan() {
    setState(() {
      _stage = _TestStage.prescan;
      _prescanJpeg = null;
      _cellMap = null;
      _fingerJpeg = null;
      _covered = null;
      _fingertip = null;
      _targetChar = '';
      _targetPhonetic = '';
      _spokenResult = null;
      _lastCorrect = null;
      _statusLine = null;
    });
    widget.audioService.speak(
      'Rescanning page. ${_camera.usingGlasses ? 'Press the glasses button' : 'Capture'} when ready.',
    );
  }

  // ── Stage 2: prompt + check ─────────────────────────────────────────────────

  /// One round: resolve the cell under the finger, ask the learner to name
  /// it, record the answer, transcribe it and score it. The page map is kept,
  /// so the next press tests another cell without rescanning.
  Future<void> _checkFinger() async {
    if (_busy || _cellMap == null) return;
    setState(() {
      _busy = true;
      _spokenResult = null;
      _lastCorrect = null;
      _statusLine = 'Detecting fingertip…';
    });

    final jpeg = await _camera.captureJpeg();
    if (jpeg == null) {
      setState(() {
        _busy = false;
        _statusLine = _camera.source?.lastError ?? 'Camera capture failed';
      });
      await widget.audioService.speak('Capture failed. Try again.');
      return;
    }

    FingertipDetection? tip = await _fingertipOnnx.detect(jpeg);
    tip ??= await _promptTapFingertip(jpeg);

    if (tip == null) {
      setState(() {
        _busy = false;
        _statusLine = 'No fingertip — tap on your finger tip on screen';
      });
      return;
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

    final target = result.hasHit ? result.headline : '';
    if (!mounted) return;
    setState(() {
      _fingerJpeg = jpeg;
      _fingertip = tip;
      _covered = result;
      _targetChar = target;
      _targetPhonetic = sinhalaLetterName(target);
      _busy = false;
      _statusLine = result.hasHit
          ? '${result.compactDots} · under your finger'
          : result.subtitle;
    });

    if (!result.hasHit || target.isEmpty || target == '—') {
      await widget.audioService.speak('No character found under your finger.');
      return;
    }
    await _askAndScore();
  }

  /// Prompt → record → transcribe → score.
  Future<void> _askAndScore() async {
    // "What is the letter under your finger? Say it aloud."
    await widget.audioService.speakSinhala(
      'ඔබේ ඇඟිල්ල යට ඇති අකුර කුමක්ද? ශබ්ද නඟා කියන්න.',
    );

    if (!_sttReady) {
      // Without the speech model there is nothing to score against, so read
      // the answer out instead of failing the learner on a missing asset.
      if (mounted) {
        setState(() => _statusLine =
            'Speech model missing — add sinhala_mms_ctc.onnx to assets/models/');
      }
      await widget.audioService.speakSinhala('අක්ෂරය $_targetPhonetic');
      return;
    }

    if (mounted) {
      setState(() {
        _listening = true;
        _statusLine = 'Listening…';
      });
    }
    // Waits for the prompt to finish before opening the mic, so the
    // recording does not capture the app's own voice.
    await widget.audioService.playMicOpen();
    final capture = await _voice.record();
    await widget.audioService.playMicClose();

    if (!mounted) return;
    setState(() {
      _listening = false;
      _statusLine = 'Checking your answer…';
    });

    if (!capture.hasAudio) {
      setState(() => _statusLine = capture.error ?? 'No audio recorded');
      await widget.audioService.speak('I could not hear you. Try again.');
      return;
    }

    final spoken = await _stt.transcribe(capture.samples);
    final matched = sinhalaAnswerMatches(spoken, _targetChar);

    if (!mounted) return;
    setState(() {
      _spokenResult = spoken;
      _lastCorrect = matched;
      _total++;
      if (matched) _correct++;
      _statusLine = 'You said: ${spoken?.isNotEmpty == true ? spoken : '—'}'
          ' · ${capture.source.name} mic';
    });

    if (matched) {
      await widget.audioService.hapticHeavy();
      await widget.audioService.playSuccessTone();
      // "Correct! The letter <name>."
      await widget.audioService.speakSinhala('නිවැරදියි! අක්ෂරය $_targetPhonetic.');
    } else {
      await widget.audioService.hapticError();
      await widget.audioService.playErrorTone();
      // "Wrong. You said <x>, but the correct letter is <name>."
      final said = spoken?.trim().isNotEmpty == true ? spoken!.trim() : '—';
      await widget.audioService.speakSinhala(
        'වැරදියි. ඔබ කීවේ $said, නමුත් නිවැරදි අක්ෂරය $_targetPhonetic.',
      );
    }

    // Stay in stage 2: the next press tests another cell on the same page.
    if (mounted) {
      setState(() => _statusLine =
          'Move to another cell and press capture for the next letter.');
    }
  }

  /// Fallback when ONNX fingertip model is missing: user taps contact point.
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

  @override
  void dispose() {
    _glassButtonSub?.cancel();
    _camera.removeListener(_onCameraSourceChanged);
    unawaited(_camera.restoreHardwareShutter());
    _fingertipOnnx.dispose();
    _voice.dispose();
    _camera.dispose();
    // The STT session is app-wide and shared, so it is not disposed here.
    super.dispose();
  }

  // ── Build ────────────────────────────────────────────────────────────────────

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
            _buildTopBar(),
            _buildBottomPanel(),
            if (_busy)
              const ColoredBox(
                color: Color(0x88000000),
                child: Center(child: CircularProgressIndicator(color: AppTheme.primaryYellow)),
              ),
          ],
        ),
      ),
    );
  }

  /// Boxes over the prescan only — see the note on Learning Mode's image area.
  Widget _buildImageArea() {
    if (_fingerJpeg != null) {
      return FrozenImageView(
        jpeg: _fingerJpeg!,
        fingertip: _fingertip,
      );
    }
    if (_prescanJpeg != null && _cellMap != null) {
      return FrozenImageView(jpeg: _prescanJpeg!, cellMap: _cellMap);
    }
    return _camera.buildPreview();
  }

  Widget _buildTopBar() {
    final title = _stage == _TestStage.prescan
        ? '1 · SCAN BRAILLE PAGE'
        : '2 · NAME THE LETTER';

    return SafeArea(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        color: Colors.black.withValues(alpha: 0.7),
        child: Row(
          children: [
            TextButton(
              onPressed: _exit,
              child: const Text('Exit', style: TextStyle(color: AppTheme.primaryYellow)),
            ),
            Expanded(
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
            if (_cellMap != null)
              TextButton(
                onPressed: _busy ? null : _rescan,
                child: const Text('Rescan', style: TextStyle(color: Colors.white70)),
              )
            else
              const SizedBox(width: 56),
          ],
        ),
      ),
    );
  }

  Widget _buildBottomPanel() {
    final resultColor = _lastCorrect == null
        ? AppTheme.primaryYellow
        : (_lastCorrect! ? const Color(0xFF00E5FF) : const Color(0xFFFF5252));

    return Positioned(
      left: 0,
      right: 0,
      bottom: 0,
      child: Container(
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 32),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.92),
          border: Border(top: BorderSide(color: resultColor, width: 2)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (_stage == _TestStage.prescan)
              const Padding(
                padding: EdgeInsets.only(bottom: 18),
                child: Column(
                  children: [
                    Icon(Icons.document_scanner_outlined, color: AppTheme.primaryYellow, size: 34),
                    SizedBox(height: 6),
                    Text(
                      'CENTER THE BRAILLE PAGE IN THE CAMERA',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        color: AppTheme.primaryYellow,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 0.7,
                      ),
                    ),
                    SizedBox(height: 4),
                    Text(
                      'Keep fingers out · use bright, even light',
                      textAlign: TextAlign.center,
                      style: TextStyle(color: Colors.white70, fontSize: 13),
                    ),
                  ],
                ),
              )
            else ...[
              Text(
                _listening ? 'LISTENING…' : 'UNDER YOUR FINGER',
                style: const TextStyle(
                  color: Colors.white70,
                  fontSize: 12,
                  letterSpacing: 1.1,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                _targetChar.isEmpty ? '—' : _targetChar,
                style: TextStyle(
                  fontSize: 56,
                  fontWeight: FontWeight.bold,
                  color: resultColor,
                ),
              ),
              if (_covered?.hasHit == true) ...[
                const SizedBox(height: 2),
                Text(
                  'dots ${_covered!.compactDots}',
                  style: const TextStyle(color: Colors.white38, fontSize: 13),
                ),
              ],
              if (_targetPhonetic.isNotEmpty) ...[
                const SizedBox(height: 4),
                Text(
                  _targetPhonetic,
                  style: const TextStyle(
                    color: Colors.white70,
                    fontSize: 18,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
              if (_spokenResult != null) ...[
                const SizedBox(height: 4),
                Text(
                  'You said: ${_spokenResult!.isEmpty ? '—' : _spokenResult!}',
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.white54, fontSize: 14),
                ),
              ],
              if (_lastCorrect != null) ...[
                const SizedBox(height: 4),
                Text(
                  _lastCorrect! ? 'නිවැරදියි! Correct' : 'වැරදියි — Incorrect',
                  style: TextStyle(color: resultColor, fontWeight: FontWeight.w600),
                ),
              ],
            ],
            const SizedBox(height: 6),
            Text(
              'Score: $_correct / $_total',
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.white70, fontSize: 15),
            ),
            if (_statusLine != null) ...[
              const SizedBox(height: 8),
              Text(
                _statusLine!,
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.white.withValues(alpha: 0.45), fontSize: 12),
              ),
            ],
            const SizedBox(height: 16),
            if (!_camera.usingGlasses)
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: (_busy || !_cameraReady) ? null : _onCapturePressed,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.primaryYellow,
                    foregroundColor: Colors.black,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                  ),
                  child: Text(
                    _stage == _TestStage.prescan
                        ? 'SCAN BRAILLE PAGE'
                        : 'READ MY FINGER & ASK ME',
                    style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                ),
              )
            else
              Text(
                _stage == _TestStage.prescan
                    ? 'Press the glasses button to scan the page'
                    : 'Press the glasses button to read your finger',
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
