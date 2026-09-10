import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../models/braille_cell.dart';
import '../services/audio_service.dart';
import '../services/camera_service.dart';
import '../services/classifier_service.dart';
import '../services/covered_cell_service.dart';
import '../services/coordinate_mapper.dart';
import '../services/fingertip_onnx_service.dart';
import '../services/prescan_bridge.dart';
import '../theme/app_theme.dart';
import '../utils/image_decode.dart';
import '../widgets/frozen_image_view.dart';
import '../widgets/tap_fingertip_dialog.dart';

enum _TestStage { prescan, quiz }

/// Two-stage Testing Mode, same on-device pipeline as Learning Mode:
/// 1. Capture hand-free page → prescan builds CellMap (yellow boxes).
///    Runs once per page; "Rescan" if the page moves.
/// 2. The app names a random target character from the CNN's own label set.
///    Learner places a finger on that cell and taps "Detect" — geometry-only
///    fingertip hit-test (no CNN on the finger photo) resolves the covered
///    cell, and correctness is `resolved.code == target.code`. Repeats with
///    a new target each round, reusing the same page map.
class TestingScreen extends StatefulWidget {
  final AudioService audioService;

  const TestingScreen({super.key, required this.audioService});

  @override
  State<TestingScreen> createState() => _TestingScreenState();
}

class _TestingScreenState extends State<TestingScreen> {
  final CameraService _camera = CameraService();
  final PrescanBridge _prescanBridge = PrescanBridge();
  final FingertipOnnxService _fingertipOnnx = FingertipOnnxService();
  final CoveredCellService _coveredCell = CoveredCellService();

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

  List<BrailleLabel> _deck = const [];
  int _deckIndex = 0;
  BrailleLabel? _target;
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
    final cnnReady = await PrescanBridge.ensureOnDeviceReady();
    final tipReady = await _fingertipOnnx.initialize();
    if (!mounted) return;

    if (cnnReady) {
      _deck = List.of(PrescanBridge.classifier.labels)
        ..removeWhere((l) => l.code == 0 || l.si.trim().isEmpty)
        ..shuffle();
    }

    setState(() {
      _cameraReady = true;
      _statusLine = [
        cnnReady ? 'CNN: braille_cnn.onnx (${_deck.length} letters)' : 'CNN failed',
        tipReady
            ? 'YOLO: ${_fingertipOnnx.loadedAsset?.split('/').last}'
            : 'YOLO failed — tap fingertip',
      ].join(' · ');
    });

    if (_deck.isEmpty) {
      await widget.audioService.speak(
        'Testing Mode could not load the character set. Check the model files and restart.',
      );
      return;
    }

    await widget.audioService.speak(
      'Testing Mode. Stage 1: hold the Braille page still with no finger, '
      'then tap capture. I will then name a letter for you to find each round.',
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

  // ── Stage 1 ──────────────────────────────────────────────────────────────────

  Future<void> _capturePrescan() async {
    if (_busy || !_cameraReady || _deck.isEmpty) return;
    setState(() {
      _busy = true;
      _statusLine = 'Scanning the page for Braille cells…';
    });

    final jpeg = await _camera.captureJpeg();
    if (jpeg == null) {
      setState(() {
        _busy = false;
        _statusLine = 'Camera capture failed';
      });
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
      await widget.audioService.speak(
        '${fixed.cells.length} cells found.',
      );
      _nextTarget();
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
      _target = null;
      _lastCorrect = null;
      _statusLine = null;
    });
    widget.audioService.speak('Rescanning page. Capture when ready.');
  }

  // ── Stage 2: prompt + check ─────────────────────────────────────────────────

  void _nextTarget() {
    if (_deck.isEmpty) return;
    if (_deckIndex >= _deck.length) {
      _deckIndex = 0;
      _deck.shuffle();
    }
    final target = _deck[_deckIndex];
    _deckIndex++;

    setState(() {
      _target = target;
      _lastCorrect = null;
      _fingerJpeg = null;
      _fingertip = null;
      _covered = null;
      _statusLine = 'Find the cell with ${target.dots}, then tap Detect.';
    });
    widget.audioService.speak('සොයන්න ${target.si}');
    widget.audioService.speak('Find the cell with ${target.dots}.');
  }

  Future<void> _checkFinger() async {
    if (_busy || _cellMap == null || _target == null) return;
    setState(() {
      _busy = true;
      _statusLine = 'Detecting fingertip…';
    });

    final jpeg = await _camera.captureJpeg();
    if (jpeg == null) {
      setState(() {
        _busy = false;
        _statusLine = 'Camera capture failed';
      });
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

    final result = _coveredCell.resolve(
      tipInFingerImage: tip.contactPoint,
      cellMap: _cellMap!,
      fingerImageWidth: tip.imageWidth,
      fingerImageHeight: tip.imageHeight,
      fingertipBox: tip.box,
    );

    final target = _target!;
    final matched = result.hasHit && result.cell!.code == target.code;

    setState(() {
      _fingerJpeg = jpeg;
      _fingertip = tip;
      _covered = result;
      _lastCorrect = matched;
      _total++;
      if (matched) _correct++;
      _busy = false;
      _statusLine = result.hasHit
          ? '${result.cell!.detectedCellLabel} detected under finger'
          : 'No cell under fingertip — try again';
    });

    if (matched) {
      await widget.audioService.playSuccessTone();
      await widget.audioService.hapticDouble();
      await widget.audioService.speak('නිවැරදියි! ඔබ හඳුනාගත්තේ ${target.si}');
    } else {
      await widget.audioService.playErrorTone();
      await widget.audioService.hapticError();
      await widget.audioService.speak('වැරදියි. නිවැරදි අක්ෂරය ${target.si}');
    }

    await Future.delayed(const Duration(milliseconds: 2000));
    if (!mounted) return;
    _nextTarget();
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

  CellMap? _mappedCellsForFingerFrame() {
    final map = _cellMap;
    final tip = _fingertip;
    if (map == null || tip == null) return map;
    final mapped = map.cells.map((c) {
      final r = CoordinateMapper.mapCellToFingerImage(
        cell: c,
        prescanWidth: map.imageWidth,
        prescanHeight: map.imageHeight,
        fingerImageWidth: tip.imageWidth,
        fingerImageHeight: tip.imageHeight,
      );
      return BrailleCell(
        id: c.id,
        x0: r.left,
        y0: r.top,
        x1: r.right,
        y1: r.bottom,
        char: c.char,
        pattern: c.pattern,
        code: c.code,
        conf: c.conf,
        line: c.line,
        col: c.col,
      );
    }).toList();
    return CellMap(cells: mapped, imageWidth: tip.imageWidth, imageHeight: tip.imageHeight);
  }

  @override
  void dispose() {
    _fingertipOnnx.dispose();
    _camera.dispose();
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

  Widget _buildImageArea() {
    if (_fingerJpeg != null) {
      return FrozenImageView(
        jpeg: _fingerJpeg!,
        cellMap: _mappedCellsForFingerFrame(),
        highlighted: _covered?.cell,
        fingertip: _fingertip,
      );
    }
    if (_prescanJpeg != null && _cellMap != null) {
      return FrozenImageView(jpeg: _prescanJpeg!, cellMap: _cellMap);
    }
    if (_cameraReady && _camera.controller != null) {
      return ColoredBox(
        color: Colors.black,
        child: Center(child: CameraPreview(_camera.controller!)),
      );
    }
    return const Center(child: CircularProgressIndicator(color: AppTheme.primaryYellow));
  }

  Widget _buildTopBar() {
    final title = _stage == _TestStage.prescan ? '1 · SCAN BRAILLE PAGE' : '2 · FIND THE LETTER';

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
    final target = _target;
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
              const Text('FIND', style: TextStyle(color: Colors.white70, fontSize: 12, letterSpacing: 1.1)),
              const SizedBox(height: 4),
              Text(
                target?.si ?? '—',
                style: TextStyle(fontSize: 56, fontWeight: FontWeight.bold, color: resultColor),
              ),
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
            if (_cellMap != null)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text(
                  'Stage 1: ${_cellMap!.cells.length} cells @ ${_cellMap!.imageWidth}x${_cellMap!.imageHeight}px',
                  style: TextStyle(color: Colors.white.withValues(alpha: 0.35), fontSize: 11),
                ),
              ),
            const SizedBox(height: 16),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: _busy ? null : (_stage == _TestStage.prescan ? _capturePrescan : _checkFinger),
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.primaryYellow,
                  foregroundColor: Colors.black,
                  padding: const EdgeInsets.symmetric(vertical: 14),
                ),
                child: Text(
                  _stage == _TestStage.prescan ? 'SCAN BRAILLE PAGE' : 'DETECT & CHECK',
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
