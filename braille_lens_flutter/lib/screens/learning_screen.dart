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
import '../theme/app_theme.dart';
import '../utils/image_decode.dart';
import '../widgets/frozen_image_view.dart';
import '../widgets/tap_fingertip_dialog.dart';

enum _LearningStage { prescan, fingerResult }

/// Two-stage Learning Mode:
/// 1. Capture hand-free page → prescan builds CellMap (yellow boxes).
/// 2. Capture finger on page → fingertip hit-test → show Sinhala letter from map.
///
/// Covered-character identification uses **geometry only**: the finger frame
/// is aligned onto the prescan and the label is read off the prescan map. The
/// CNN never sees the finger frame, where the hand hides the very dots that
/// would have to be classified.
class LearningScreen extends StatefulWidget {
  final AudioService audioService;

  const LearningScreen({super.key, required this.audioService});

  @override
  State<LearningScreen> createState() => _LearningScreenState();
}

class _LearningScreenState extends State<LearningScreen> {
  /// Glasses when connected, phone camera otherwise — swaps itself if the
  /// glasses drop mid-session.
  final CameraSourceController _camera = CameraSourceController();
  final PrescanBridge _prescanBridge = PrescanBridge();
  final FingertipOnnxService _fingertipOnnx = FingertipOnnxService();
  final CoveredCellService _coveredCell = CoveredCellService();

  /// Frame-button presses, routed to [_onCapturePressed] so the hardware
  /// button and the on-screen control run one path.
  StreamSubscription<GlassButtonClicked>? _glassButtonSub;

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

    // The frame button and the on-screen button both land on
    // _onCapturePressed, so the two controls can never diverge.
    _glassButtonSub = GlassDeviceService.instance.buttonClicks.listen((_) {
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
      ].join(' · ');
    });

    final trigger = _camera.usingGlasses
        ? 'press the button on your glasses'
        : 'tap capture';
    await widget.audioService.speak(
      'Learning Mode. Stage 1: hold the Braille page still with no finger, '
      'then $trigger. Stage 2: place your finger on a cell and $trigger again.',
    );
  }

  /// The camera source swapped underneath us (glasses connected or dropped).
  /// Prescan state is deliberately kept — only the viewfinder changes.
  void _onCameraSourceChanged() {
    if (!mounted) return;
    setState(() => _cameraReady = _camera.isReady);
  }

  /// Single capture entry point shared by the on-screen button and the
  /// glasses frame button.
  Future<void> _onCapturePressed() async {
    if (_busy || _isExiting) return;
    if (_stage == _LearningStage.prescan) {
      await _capturePrescan();
    } else {
      await _captureFinger();
    }
  }

  Future<void> _exit() async {
    if (_isExiting) return;
    setState(() => _isExiting = true);
    await widget.audioService.stopSpeech();
    await widget.audioService.hapticLight();
    await widget.audioService.speak('Returning to main menu.');
    if (mounted) Navigator.pop(context);
  }

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
        _stage = _LearningStage.fingerResult;
        _busy = false;
        _statusLine =
            '${fixed.cells.length} cells found · place a finger, then tap capture';
      });
      await widget.audioService.speak(
        '${fixed.cells.length} cells found. Place your finger on a character and tap capture.',
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

  Future<void> _captureFinger() async {
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

    if (!mounted) return;
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
      final ch = result.headline;
      if (ch == '—') {
        await widget.audioService.speak(
          'The detected cell is an indicator, not a standalone character.',
        );
      } else {
        await widget.audioService.speakSinhalaCharacter(ch);
      }
    } else {
      await widget.audioService.speak('No character found under your finger.');
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
      box: Rect.fromCenter(
        center: tap,
        width: 48,
        height: 48,
      ),
      confidence: 1.0,
      imageWidth: decoded.width,
      imageHeight: decoded.height,
    );
  }

  void _rescan() {
    setState(() {
      _stage = _LearningStage.prescan;
      _prescanJpeg = null;
      _cellMap = null;
      _fingerJpeg = null;
      _covered = null;
      _fingertip = null;
      _statusLine = null;
    });
    widget.audioService.speak('Rescanning page. Capture when ready.');
  }

  @override
  void dispose() {
    _glassButtonSub?.cancel();
    _camera.removeListener(_onCameraSourceChanged);
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

  /// Small, non-obscuring "working" badge — deliberately does *not* dim the
  /// rest of the screen (a full-screen translucent veil made the camera
  /// preview, status text, and buttons hard to read while scanning).
  /// [_statusLine] already carries the actual progress text; this is just a
  /// small spinner so it's clear something is happening.
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

  /// The cell boxes belong to the page they were found on, so they are drawn
  /// over the prescan and nowhere else. Drawing them over the finger frame
  /// would put them where the *scan* saw cells rather than where the cells
  /// are in the picture being shown, which reads as a broken detector even
  /// when the lookup underneath is right.
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

  /// Small rounded badge with just enough backing to stay legible over a
  /// bright camera feed (e.g. white paper) -- deliberately not a full-width
  /// bar, so the camera preview around it stays at full brightness.
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
                  style: TextStyle(color: AppTheme.primaryYellow, fontWeight: FontWeight.bold)),
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
                child: const Text('Rescan', style: TextStyle(color: Colors.white)),
              )
            else
              const SizedBox(width: 8),
          ],
        ),
      ),
    );
  }

  /// Framing guidance as a small floating pill instead of a permanent block
  /// baked into the bottom sheet -- keeps it visible without eating into
  /// the camera viewport's height.
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
                    'Use bright, even light',
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
        // Trimmed from the original fromLTRB(20,16,20,32) + a large
        // icon+instruction block baked in above the button -- that block
        // (now a small floating pill over the viewport instead, see
        // _buildFramingHint) plus this padding used to eat a third or more
        // of the screen's height, squeezing the camera preview into a thin
        // strip and reading as if the whole feed were covered by a scrim.
        padding: const EdgeInsets.fromLTRB(20, 12, 20, 20),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.75),
          border: const Border(top: BorderSide(color: AppTheme.primaryYellow, width: 2)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (_fingertip != null)
              const Padding(
                padding: EdgeInsets.only(bottom: 8),
                child: Text(
                  'FINGER TRACKED',
                  style: TextStyle(color: Color(0xFF00E5FF), fontWeight: FontWeight.bold),
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
                          ? 'Place a finger, then tap capture'
                          : 'Capture a hand-free page photo')),
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
                  _stage == _LearningStage.prescan
                      ? 'SCAN BRAILLE PAGE'
                      : 'DETECT CHARACTER UNDER FINGER',
                  style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

