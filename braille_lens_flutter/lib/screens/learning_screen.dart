import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../models/braille_cell.dart';
import '../services/audio_service.dart';
import '../services/camera_source.dart';
import '../services/cell_hit_test.dart';
import '../services/coordinate_mapper.dart';
import '../services/covered_cell_service.dart';
import '../services/fingertip_onnx_service.dart';
import '../services/glass_device_service.dart';
import '../services/hands_free_learning_session.dart';
import '../services/prescan_bridge.dart';
import '../theme/app_theme.dart';
import '../utils/homography.dart';
import '../utils/image_decode.dart';
import '../utils/sinhala_phonetics.dart';
import '../widgets/frozen_image_view.dart';
import '../widgets/tap_fingertip_dialog.dart';

enum _LearningStage { prescan, fingerResult }

/// Two-stage Learning Mode with SRS hands-free dwell (FR5/FR6/FR8).
///
/// Auto page baseline when no tip is present, ~3 s cell dwell with lock
/// earcons, then geometry lookup via [CoveredCellService.resolveAligned].
/// Yellow button / glasses frame button remain manual overrides.
class LearningScreen extends StatefulWidget {
  final AudioService audioService;

  const LearningScreen({super.key, required this.audioService});

  @override
  State<LearningScreen> createState() => _LearningScreenState();
}

class _LearningScreenState extends State<LearningScreen> {
  final CameraSourceController _camera = CameraSourceController();
  final PrescanBridge _prescanBridge = PrescanBridge();
  final FingertipOnnxService _fingertipOnnx = FingertipOnnxService();
  final CoveredCellService _coveredCell = CoveredCellService();
  final HandsFreeLearningSession _session = HandsFreeLearningSession();

  StreamSubscription<GlassButtonClicked>? _glassButtonSub;
  Timer? _sampleTimer;
  bool _sampleInFlight = false;
  bool _countdownAborted = false;
  bool _handsFreeEnabled = true;

  /// Last successful live→prescan transform for cheap dwell cell ids.
  Homography? _lastAlignH;

  _LearningStage _stage = _LearningStage.prescan;
  bool _cameraReady = false;
  bool _busy = false;
  bool _isExiting = false;
  String? _statusLine;

  Uint8List? _prescanJpeg;
  CellMap? _cellMap;
  Uint8List? _fingerJpeg;
  CoveredCellResult? _covered;
  FingertipDetection? _fingertip;

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
    if (!mounted) return;
    setState(() {
      _cameraReady = _camera.isReady;
      _statusLine = [
        'Camera: ${_camera.label}',
        cnnReady ? 'CNN: braille_cnn.onnx' : 'CNN failed',
        tipReady
            ? 'YOLO: ${_fingertipOnnx.loadedAsset?.split('/').last}'
            : 'YOLO failed — tap fingertip',
        'Hands-free: on',
      ].join(' · ');
    });

    await widget.audioService.speak(
      'Learning Mode. Clear the page for an automatic scan. '
      'Then place your finger on a letter and hold still. '
      'You can also capture manually.',
    );

    if (!mounted || !_cameraReady) return;
    _session.start();
    _dispatch(_session.onSample(
      now: DateTime.now(),
      tipPresent: false,
      cellId: null,
    ));
    _startSampleLoop();
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
      if (jpeg == null || !mounted) return;

      final tip = await _fingertipOnnx.detect(jpeg);
      int? cellId;
      final map = _cellMap;
      if (tip != null &&
          map != null &&
          (phase == HandsFreePhase.reading ||
              phase == HandsFreePhase.dwelling ||
              phase == HandsFreePhase.lockBeeps ||
              phase == HandsFreePhase.cooldown ||
              phase == HandsFreePhase.pageCountdown)) {
        cellId = await _cheapCellId(jpeg, tip);
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

  /// Cheap cell identity for dwell (not the announce path).
  Future<int?> _cheapCellId(Uint8List jpeg, FingertipDetection tip) async {
    final map = _cellMap;
    if (map == null) return null;

    final liveBoxes = await _prescanBridge.detectCellBoxes(jpeg);
    Offset probe = tip.contactPoint;
    for (final b in liveBoxes) {
      if (b.contains(tip.contactPoint)) {
        probe = b.center;
        break;
      }
    }

    final mapped = _lastAlignH != null
        ? _lastAlignH!.transform(probe)
        : CoordinateMapper.mapFingerTipToPrescan(
            tipInFingerImage: probe,
            prescanWidth: map.imageWidth,
            prescanHeight: map.imageHeight,
            fingerImageWidth: tip.imageWidth,
            fingerImageHeight: tip.imageHeight,
          );

    return CellHitTest.hitTest(mapped, map, skipEmpty: true)?.id;
  }

  Future<void> _dispatch(List<HandsFreeAction> actions) async {
    for (final a in actions) {
      if (!mounted || _isExiting) return;
      switch (a) {
        case HandsFreeStatus(:final message):
          setState(() => _statusLine = message);
        case HandsFreeSpeak(:final text, :final sinhala):
          if (sinhala) {
            await widget.audioService.speakSinhala(text);
          } else {
            await widget.audioService.speak(text);
          }
        case HandsFreePlayCountdown(
            :final count,
            :final intervalMs,
            :final kind
          ):
          await _runCountdown(count, intervalMs, kind);
        case HandsFreeAbortCountdown():
          _countdownAborted = true;
        case HandsFreeRequestPrescan():
          await _autoPrescan();
        case HandsFreeRequestFingerCapture():
          await _autoFinger();
        case HandsFreeRequestSoftRescan():
          await _autoSoftRescan();
      }
    }
  }

  Future<void> _runCountdown(
    int count,
    int intervalMs,
    HandsFreeCountdownKind kind,
  ) async {
    _countdownAborted = false;
    // Keep sampling so a returning fingertip can abort page/lock/soft beeps.
    final ok = await widget.audioService.playCountdownBeeps(
      count,
      gap: Duration(milliseconds: intervalMs),
      shouldAbort: () => _countdownAborted || _isExiting,
    );
    if (!mounted || _isExiting) return;
    if (!ok || _countdownAborted) return;
    await _dispatch(_session.onCountdownFinished(kind));
  }

  Future<void> _autoPrescan() async {
    _session.pause();
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
    setState(() {
      _busy = true;
      _statusLine = 'Refreshing page map…';
      _fingerJpeg = null;
      _covered = null;
      _fingertip = null;
    });
    final jpeg = await _camera.captureJpeg();
    if (jpeg == null) {
      setState(() => _busy = false);
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
      await _dispatch(_session.onSoftRescanFinished(success: false));
      _session.resume();
      return;
    }
    final ok = await _runPrescanFromJpeg(jpeg, speakOnSuccess: false);
    if (ok) _lastAlignH = null;
    await _dispatch(_session.onSoftRescanFinished(success: ok));
    _session.resume();
  }

  // ── Shared pipeline (manual + auto) ────────────────────────────────────────

  Future<void> _onCapturePressed() async {
    if (_busy || _isExiting) return;
    _countdownAborted = true;
    _session.pause();
    try {
      if (_stage == _LearningStage.prescan) {
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
      await widget.audioService.speak('Capture failed. Try again.');
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
      await widget.audioService.speak('Capture failed. Try again.');
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
      setState(() {
        _prescanJpeg = jpeg;
        _cellMap = fixed;
        _stage = _LearningStage.fingerResult;
        _fingerJpeg = null;
        _covered = null;
        _fingertip = null;
        _busy = false;
        _statusLine =
            '${fixed.cells.length} cells found · place a finger, then $_captureHint';
      });
      if (speakOnSuccess) {
        await widget.audioService.speakSinhala(
          'පිටුව ස්කෑන් කර අවසන්. දැන් ඔබේ ඇඟිල්ල අකුරක් මත තබන්න.',
        );
      }
      return true;
    } on PrescanUnavailableException catch (e) {
      if (!mounted) return false;
      setState(() {
        _busy = false;
        _statusLine = e.message;
      });
      if (speakOnSuccess) {
        await widget.audioService.speak(
          'Page scan failed. Hold the page steady with good lighting and try again.',
        );
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
        await widget.audioService.speak('No fingertip detected. Try again.');
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

    // Cache a cheap similarity from tip→prescan for the next dwell samples.
    if (result.hasHit) {
      final dx = result.tipInPrescan.dx - tip.contactPoint.dx;
      final dy = result.tipInPrescan.dy - tip.contactPoint.dy;
      _lastAlignH = Homography([1, 0, dx, 0, 1, dy, 0, 0, 1]);
    }

    if (!mounted) return null;
    setState(() {
      _fingerJpeg = jpeg;
      _fingertip = tip;
      _covered = result;
      _busy = false;
      _statusLine = result.hasHit
          ? '${result.compactDots} · ${result.headline}'
          : result.subtitle;
    });

    if (result.hasHit) {
      if (playSuccessEarcon) {
        await widget.audioService.playSuccessTone();
        await widget.audioService.hapticLight();
      }
      final ch = result.headline;
      if (ch == '—') {
        await widget.audioService.speak(
          'The detected cell is an indicator, not a standalone character.',
        );
      } else {
        final name = sinhalaLetterName(ch);
        await widget.audioService.speakSinhala(
          name.isEmpty ? ch : 'අක්ෂරය $name',
        );
      }
      return result.cell?.id;
    }

    await widget.audioService.speak('No character found under your finger.');
    return null;
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

  void _rescan() {
    _countdownAborted = true;
    _lastAlignH = null;
    setState(() {
      _stage = _LearningStage.prescan;
      _prescanJpeg = null;
      _cellMap = null;
      _fingerJpeg = null;
      _covered = null;
      _fingertip = null;
      _statusLine = null;
    });
    _session.resetToPageHunt();
    widget.audioService.speak(
      'Rescanning page. Clear the page for automatic scan, or capture manually.',
    );
  }

  Future<void> _exit() async {
    if (_isExiting) return;
    setState(() => _isExiting = true);
    _countdownAborted = true;
    _handsFreeEnabled = false;
    _stopSampleLoop();
    _session.pause();
    await widget.audioService.stopSpeech();
    await widget.audioService.hapticLight();
    await widget.audioService.speak('Returning to main menu.');
    if (mounted) Navigator.pop(context);
  }

  @override
  void dispose() {
    _countdownAborted = true;
    _handsFreeEnabled = false;
    _stopSampleLoop();
    _glassButtonSub?.cancel();
    _camera.removeListener(_onCameraSourceChanged);
    unawaited(_camera.restoreHardwareShutter());
    _fingertipOnnx.dispose();
    _camera.dispose();
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
    final title = _stage == _LearningStage.prescan
        ? '1 · SCAN BRAILLE PAGE'
        : (_fingerJpeg != null ? '2 · DETECTED CHARACTER' : '2 · POINT TO A CELL');

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
    if (!(_stage == _LearningStage.prescan && _cellMap == null)) {
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

  Widget _buildBottomPanel() {
    final covered = _covered;

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
            if (_fingertip != null)
              const Padding(
                padding: EdgeInsets.only(bottom: 8),
                child: Text(
                  'FINGER TRACKED',
                  style: TextStyle(
                      color: Color(0xFF00E5FF), fontWeight: FontWeight.bold),
                ),
              ),
            Text(
              covered?.hasHit == true ? covered!.compactDots : '—',
              style: const TextStyle(
                fontSize: 56,
                fontWeight: FontWeight.bold,
                color: AppTheme.primaryYellow,
              ),
            ),
            const SizedBox(height: 6),
            Text(
              covered?.hasHit == true
                  ? covered!.subtitle
                  : (covered?.subtitle ??
                      (_cellMap != null
                          ? 'Hold finger on a cell (~3 s) or $_captureHint'
                          : 'Clear the page for auto-scan, or capture')),
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.white70, fontSize: 15),
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
                  onPressed:
                      (_busy || !_cameraReady) ? null : _onCapturePressed,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.primaryYellow,
                    foregroundColor: Colors.black,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                  ),
                  child: Text(
                    _stage == _LearningStage.prescan
                        ? 'SCAN BRAILLE PAGE'
                        : 'DETECT CHARACTER UNDER FINGER',
                    style: const TextStyle(
                        fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                ),
              )
            else
              Text(
                _stage == _LearningStage.prescan
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
