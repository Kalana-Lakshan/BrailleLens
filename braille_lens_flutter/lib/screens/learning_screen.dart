import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../models/braille_cell.dart';
import '../services/audio_service.dart';
import '../services/camera_service.dart';
import '../services/covered_cell_service.dart';
import '../services/coordinate_mapper.dart';
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
/// Covered-character identification uses **geometry only** (no CNN on finger photo).
class LearningScreen extends StatefulWidget {
  final AudioService audioService;

  const LearningScreen({super.key, required this.audioService});

  @override
  State<LearningScreen> createState() => _LearningScreenState();
}

class _LearningScreenState extends State<LearningScreen> {
  final CameraService _camera = CameraService();
  final PrescanBridge _prescanBridge = PrescanBridge();
  final FingertipOnnxService _fingertipOnnx = FingertipOnnxService();
  final CoveredCellService _coveredCell = CoveredCellService();

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
    final cnnReady = await PrescanBridge.ensureOnDeviceReady();
    final tipReady = await _fingertipOnnx.initialize();
    if (!mounted) return;
    setState(() {
      _cameraReady = true;
      _statusLine = [
        cnnReady ? 'CNN: braille_cnn.onnx' : 'CNN failed',
        tipReady
            ? 'YOLO: ${_fingertipOnnx.loadedAsset?.split('/').last}'
            : 'YOLO failed — tap fingertip',
      ].join(' · ');
    });
    await widget.audioService.speak(
      'Learning Mode. Stage 1: hold the Braille page still with no finger, '
      'then tap capture. Stage 2: place your finger on a cell and tap capture.',
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

  Future<void> _capturePrescan() async {
    if (_busy || !_cameraReady) return;
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

    setState(() {
      _fingerJpeg = jpeg;
      _fingertip = tip;
      _covered = result;
      _busy = false;
      _statusLine = result.hasHit
          ? '${result.cell!.detectedCellLabel} detected under finger'
          : 'No cell under fingertip — rescan page or adjust finger';
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
    return CellMap(
      cells: mapped,
      imageWidth: tip.imageWidth,
      imageHeight: tip.imageHeight,
    );
  }

  @override
  void dispose() {
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

  Widget _buildImageArea() {
    if (_stage == _LearningStage.fingerResult && _fingerJpeg != null) {
      return FrozenImageView(
        jpeg: _fingerJpeg!,
        cellMap: _mappedCellsForFingerFrame(),
        highlighted: _covered?.cell,
        fingertip: _fingertip,
      );
    }
    if (_prescanJpeg != null && _cellMap != null) {
      return FrozenImageView(
        jpeg: _prescanJpeg!,
        cellMap: _cellMap,
      );
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
    final title = _stage == _LearningStage.prescan
        ? '1 · SCAN BRAILLE PAGE'
        : (_fingerJpeg != null ? '2 · DETECTED CHARACTER' : '2 · POINT TO A CELL');

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
    final covered = _covered;
    final headline = covered?.headline ?? '—';
    final subtitle = covered?.subtitle ??
        (_cellMap != null
            ? '${_cellMap!.cells.length} cells ready'
            : 'Capture a hand-free page photo');

    return Positioned(
      left: 0,
      right: 0,
      bottom: 0,
      child: Container(
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 32),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.92),
          border: const Border(top: BorderSide(color: AppTheme.primaryYellow, width: 2)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (_stage == _LearningStage.prescan && _cellMap == null)
              const Padding(
                padding: EdgeInsets.only(bottom: 18),
                child: Column(
                  children: [
                    Icon(Icons.document_scanner_outlined,
                        color: AppTheme.primaryYellow, size: 34),
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
              ),
            if (_fingertip != null)
              const Padding(
                padding: EdgeInsets.only(bottom: 8),
                child: Text(
                  '👆 FINGER TRACKED',
                  style: TextStyle(color: Color(0xFF00E5FF), fontWeight: FontWeight.bold),
                ),
              ),
            if (covered?.hasHit == true) ...[
              const Text(
                'SINHALA CHARACTER',
                style: TextStyle(color: Colors.white70, fontSize: 12, letterSpacing: 1.1),
              ),
              const SizedBox(height: 4),
              Text(
                headline,
                style: const TextStyle(fontSize: 56, fontWeight: FontWeight.bold, color: AppTheme.primaryYellow),
              ),
              const SizedBox(height: 4),
              Text(
                'Detected ${covered!.cell!.detectedCellLabel}',
                style: const TextStyle(color: Color(0xFF00E5FF), fontWeight: FontWeight.w600),
              ),
            ] else
              Text(
                headline,
                style: const TextStyle(fontSize: 42, fontWeight: FontWeight.bold, color: AppTheme.primaryYellow),
              ),
            const SizedBox(height: 6),
            Text(
              subtitle,
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
                onPressed: _busy
                    ? null
                    : (_stage == _LearningStage.prescan ? _capturePrescan : _captureFinger),
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

