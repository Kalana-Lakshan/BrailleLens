import 'dart:async';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

import '../models/braille_cell.dart';
import '../services/audio_service.dart';
import '../services/camera_service.dart';
import '../services/covered_cell_service.dart';
import '../services/fingertip_onnx_service.dart';
import '../services/prescan_bridge.dart';
import '../theme/app_theme.dart';
import '../utils/answer_match.dart';
import '../widgets/tap_fingertip_dialog.dart';

enum _TestPhase { scanPage, aimFinger, quiz }

enum _TestState {
  idle,
  prompting,
  listening,
  evaluating,
  feedbackCorrect,
  feedbackIncorrect,
}

/// Testing Mode: same two-photo covered cell as Learning, then spoken quiz.
class TestingScreen extends StatefulWidget {
  final AudioService audioService;

  const TestingScreen({super.key, required this.audioService});

  @override
  State<TestingScreen> createState() => _TestingScreenState();
}

class _TestingScreenState extends State<TestingScreen>
    with TickerProviderStateMixin {
  final CameraService _camera = CameraService();
  final PrescanBridge _prescanBridge = PrescanBridge();
  final FingertipOnnxService _fingertipOnnx = FingertipOnnxService();
  final CoveredCellService _coveredCell = CoveredCellService();

  _TestPhase _phase = _TestPhase.scanPage;
  _TestState _testState = _TestState.idle;
  bool _cameraReady = false;
  bool _busy = false;
  bool _loopActive = false;
  bool _isExiting = false;
  String? _statusLine;

  CellMap? _cellMap;
  Uint8List? _fingerJpeg;
  CoveredCellResult? _covered;
  FingertipDetection? _fingertip;

  String _expectedChar = '';
  String _spokenAnswer = '';
  int _correct = 0;
  int _total = 0;

  late AnimationController _micPulseCtrl;
  late AnimationController _resultCtrl;
  late Animation<double> _micPulse;
  late Animation<double> _resultScale;

  @override
  void initState() {
    super.initState();

    _micPulseCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 700),
    )..repeat(reverse: true);

    _resultCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 450),
    );

    _micPulse = Tween<double>(begin: 0.9, end: 1.1).animate(
      CurvedAnimation(parent: _micPulseCtrl, curve: Curves.easeInOut),
    );

    _resultScale = TweenSequence<double>([
      TweenSequenceItem(tween: Tween(begin: 0.0, end: 1.1), weight: 2),
      TweenSequenceItem(tween: Tween(begin: 1.1, end: 1.0), weight: 1),
    ]).animate(CurvedAnimation(parent: _resultCtrl, curve: Curves.easeOut));

    _boot();
  }

  Future<void> _boot() async {
    await _camera.initialize();
    await PrescanBridge.ensureOnDeviceReady();
    await _fingertipOnnx.initialize();
    if (!mounted) return;
    setState(() => _cameraReady = _camera.isInitialized);
    if (!_camera.isInitialized) {
      await widget.audioService.speak(
        'Camera is not available. Grant camera permission and try again.',
      );
      return;
    }
    await widget.audioService.hapticMedium();
    await widget.audioService.speak(
      'Testing Mode. First scan the page with no finger, then place your finger on a cell. '
      'I will ask you to say that character. Say stop to exit. Double-tap to return to the menu.',
    );
  }

  Future<void> _exitScreen() async {
    if (_isExiting) return;
    setState(() {
      _isExiting = true;
      _loopActive = false;
    });
    widget.audioService.stopListening();
    await widget.audioService.stopSpeech();
    await widget.audioService.hapticLight();
    await widget.audioService.speak('Returning to main menu.');
    if (mounted) Navigator.pop(context);
  }

  Future<void> _capturePrescan() async {
    if (_busy || !_cameraReady) return;
    setState(() {
      _busy = true;
      _statusLine = 'Scanning page…';
    });
    final jpeg = await _camera.captureJpeg();
    if (!mounted) return;
    if (jpeg == null) {
      setState(() {
        _busy = false;
        _statusLine = 'Camera capture failed';
      });
      return;
    }
    try {
      final map = await _prescanBridge.prescanPage(jpeg);
      if (!mounted) return;
      if (map.cells.isEmpty) throw Exception('No cells detected');
      final decoded = img.decodeImage(jpeg);
      final w = decoded?.width ?? map.imageWidth;
      final h = decoded?.height ?? map.imageHeight;
      final fixed = CellMap(cells: map.cells, imageWidth: w, imageHeight: h);
      setState(() {
        _cellMap = fixed;
        _phase = _TestPhase.aimFinger;
        _busy = false;
        _statusLine = '${fixed.cells.length} cells · place a finger, then capture';
      });
      await widget.audioService.speak(
        '${fixed.cells.length} cells found. Place your finger on a character and tap capture.',
      );
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _statusLine = 'Prescan error: $e';
      });
      await widget.audioService.speak(
        'Page scan failed. Hold the page steady and try again.',
      );
    }
  }

  Future<void> _captureFinger() async {
    if (_busy || _cellMap == null) return;
    setState(() {
      _busy = true;
      _statusLine = 'Detecting fingertip…';
    });
    final jpeg = await _camera.captureJpeg();
    if (!mounted) return;
    if (jpeg == null) {
      setState(() {
        _busy = false;
        _statusLine = 'Camera capture failed';
      });
      return;
    }
    FingertipDetection? tip = await _fingertipOnnx.detect(jpeg);
    if (!mounted) return;
    tip ??= await _promptTapFingertip(jpeg);
    if (!mounted) return;
    if (tip == null) {
      setState(() {
        _busy = false;
        _statusLine = 'No fingertip — tap the tip on screen or recapture';
      });
      await widget.audioService.speak('No fingertip found. Try again.');
      return;
    }
    final result = _coveredCell.resolve(
      tipInFingerImage: tip.contactPoint,
      cellMap: _cellMap!,
      fingerImageWidth: tip.imageWidth,
      fingerImageHeight: tip.imageHeight,
      fingertipBox: tip.box,
    );
    if (!mounted) return;
    if (!result.hasHit) {
      setState(() {
        _fingerJpeg = jpeg;
        _fingertip = tip;
        _covered = result;
        _busy = false;
        _statusLine = 'No cell under fingertip';
      });
      await widget.audioService.speak('No character under your finger. Adjust and capture again.');
      return;
    }
    if (!mounted) return;
    setState(() {
      _fingerJpeg = jpeg;
      _fingertip = tip;
      _covered = result;
      _expectedChar = result.headline;
      _busy = false;
      _phase = _TestPhase.quiz;
      _loopActive = true;
      _statusLine = 'Say the character under your finger';
    });
    _runTestLoop();
  }

  Future<FingertipDetection?> _promptTapFingertip(Uint8List jpeg) async {
    final decoded = img.decodeImage(jpeg);
    if (decoded == null) return null;
    final tap = await showTapFingertipDialog(context: context, jpeg: jpeg);
    if (tap == null) return null;
    return FingertipDetection(
      contactPoint: tap,
      box: Rect.fromCenter(center: tap, width: 48, height: 48),
      confidence: 1.0,
      imageWidth: decoded.width,
      imageHeight: decoded.height,
    );
  }

  Future<void> _runTestLoop() async {
    while (mounted && _loopActive && _phase == _TestPhase.quiz) {
      if (!mounted || !_loopActive) break;
      setState(() {
        _testState = _TestState.prompting;
        _spokenAnswer = '';
      });

      await widget.audioService.speak(
        'Please state the character under your finger.',
      );
      await widget.audioService.hapticMedium();
      if (!mounted || !_loopActive) break;

      setState(() => _testState = _TestState.listening);
      await widget.audioService.playStartListeningTone();

      final outcome = await widget.audioService.listenUntilStop(
        timeout: const Duration(seconds: 10),
      );
      if (!mounted || !_loopActive) break;

      await widget.audioService.playStopListeningTone();
      await widget.audioService.hapticLight();

      if (outcome.stoppedByKeyword) {
        await _exitScreen();
        return;
      }

      String? spoken = outcome.words;
      if (outcome.sttUnavailable) {
        spoken = await _keyboardFallback();
        if (!mounted || !_loopActive) break;
        if (spoken == '__exit__') {
          await _exitScreen();
          return;
        }
      } else if (outcome.timedOut && (spoken == null || spoken.isEmpty)) {
        await widget.audioService.speak("I didn't catch that. Let's try again.");
        continue;
      }

      setState(() {
        _testState = _TestState.evaluating;
        _spokenAnswer = spoken ?? '';
        _total++;
      });
      await Future.delayed(const Duration(milliseconds: 250));
      if (!mounted || !_loopActive) break;

      final isCorrect = spokenAnswerMatches(spoken, _expectedChar);

      if (isCorrect) {
        _correct++;
        setState(() => _testState = _TestState.feedbackCorrect);
        _resultCtrl.forward(from: 0);
        await widget.audioService.playSuccessTone();
        await widget.audioService.hapticDouble();
        await widget.audioService.speak('Correct!');
      } else {
        setState(() => _testState = _TestState.feedbackIncorrect);
        _resultCtrl.forward(from: 0);
        final spokenStr =
            (spoken != null && spoken.isNotEmpty) ? spoken : 'nothing';
        await widget.audioService.playErrorTone();
        await widget.audioService.hapticError();
        await widget.audioService.speak(
          'Incorrect. You said $spokenStr, '
          'but the character is $_expectedChar.',
        );
      }

      await Future.delayed(const Duration(milliseconds: 1800));
      if (!mounted || !_loopActive) break;

      setState(() {
        _phase = _TestPhase.aimFinger;
        _fingerJpeg = null;
        _fingertip = null;
        _covered = null;
        _loopActive = false;
        _testState = _TestState.idle;
        _statusLine = 'Place your finger on the next cell, then capture';
      });
      await widget.audioService.speak(
        'Place your finger on the next cell and tap capture.',
      );
      break;
    }
  }

  Future<String?> _keyboardFallback() async {
    final controller = TextEditingController();
    final typed = await showDialog<String>(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => AlertDialog(
        backgroundColor: Colors.grey[900],
        title: const Text('Type the character', style: TextStyle(color: Colors.white)),
        content: TextField(
          controller: controller,
          autofocus: true,
          style: const TextStyle(color: Colors.white, fontSize: 28),
          decoration: const InputDecoration(hintText: 'Letter', hintStyle: TextStyle(color: Colors.white38)),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, '__exit__'),
            child: const Text('Exit'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, controller.text),
            child: const Text('OK'),
          ),
        ],
      ),
    );
    controller.dispose();
    return typed;
  }

  void _rescan() {
    _loopActive = false;
    setState(() {
      _phase = _TestPhase.scanPage;
      _cellMap = null;
      _fingerJpeg = null;
      _covered = null;
      _fingertip = null;
      _statusLine = null;
      _testState = _TestState.idle;
    });
    widget.audioService.speak('Rescanning page.');
  }

  @override
  void dispose() {
    _loopActive = false;
    _micPulseCtrl.dispose();
    _resultCtrl.dispose();
    _fingertipOnnx.dispose();
    _camera.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Semantics(
      label: 'Testing Mode. Scan the page, cover a cell, then speak the character. Double-tap to exit.',
      child: GestureDetector(
        onDoubleTap: _phase == _TestPhase.quiz ? _exitScreen : null,
        child: Scaffold(
          backgroundColor: Colors.black,
          body: Stack(
            fit: StackFit.expand,
            children: [
              _buildImageArea(),
              SafeArea(
                child: Column(
                  children: [
                    _buildTopBar(),
                    if (_phase == _TestPhase.quiz) Expanded(child: _buildCenterContent()),
                    if (_phase != _TestPhase.quiz) const Spacer(),
                    if (_phase == _TestPhase.quiz) _buildScoreBar(),
                    _buildBottomPanel(),
                  ],
                ),
              ),
              if (_busy)
                const ColoredBox(
                  color: Color(0x88000000),
                  child: Center(
                    child: CircularProgressIndicator(color: AppTheme.primaryYellow),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildImageArea() {
    if (_cameraReady &&
        _camera.controller != null &&
        _camera.controller!.value.isInitialized &&
        (_phase != _TestPhase.quiz || _fingerJpeg == null)) {
      return ColoredBox(
        color: Colors.black,
        child: Center(child: CameraPreview(_camera.controller!)),
      );
    }
    if (_fingerJpeg != null) {
      return Opacity(
        opacity: _phase == _TestPhase.quiz ? 0.35 : 1,
        child: Image.memory(
          _fingerJpeg!,
          fit: BoxFit.contain,
          width: double.infinity,
          height: double.infinity,
        ),
      );
    }
    return Container(color: AppTheme.backgroundBlack);
  }

  Widget _buildTopBar() {
    final title = switch (_phase) {
      _TestPhase.scanPage => 'TEST · SCAN PAGE',
      _TestPhase.aimFinger => 'TEST · PLACE FINGER',
      _TestPhase.quiz => 'TEST · SAY THE LETTER',
    };
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
      child: Row(
        children: [
          TextButton(
            onPressed: _exitScreen,
            child: const Text('Exit', style: TextStyle(color: Colors.white)),
          ),
          Expanded(
            child: Text(
              title,
              textAlign: TextAlign.center,
              style: const TextStyle(
                color: Colors.white70,
                fontSize: 13,
                fontWeight: FontWeight.w600,
                letterSpacing: 1.2,
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
    );
  }

  Widget _buildBottomPanel() {
    if (_phase == _TestPhase.quiz) {
      return Padding(
        padding: const EdgeInsets.fromLTRB(20, 8, 20, 28),
        child: Text(
          'Double-tap to exit  ·  Say "stop" to leave',
          textAlign: TextAlign.center,
          style: TextStyle(fontSize: 11, color: Colors.white.withValues(alpha: 0.35)),
        ),
      );
    }
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 8, 20, 28),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (_statusLine != null)
            Text(
              _statusLine!,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.white70, fontSize: 14),
            ),
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: _busy
                  ? null
                  : (_phase == _TestPhase.scanPage ? _capturePrescan : _captureFinger),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppTheme.primaryYellow,
                foregroundColor: Colors.black,
                padding: const EdgeInsets.symmetric(vertical: 14),
              ),
              child: Text(
                _phase == _TestPhase.scanPage
                    ? 'Capture page (no finger)'
                    : 'Capture finger on cell',
                style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCenterContent() {
    return Center(
      child: AnimatedSwitcher(
        duration: const Duration(milliseconds: 400),
        child: _buildStateWidget(),
      ),
    );
  }

  Widget _buildStateWidget() {
    switch (_testState) {
      case _TestState.idle:
        return const SizedBox.shrink();
      case _TestState.prompting:
        return _StateCard(
          key: const ValueKey('prompt'),
          icon: Icons.volume_up_rounded,
          iconColor: AppTheme.primaryYellow,
          label: 'Listen for the prompt',
          subLabel: 'Say the letter under your finger',
          cardColor: AppTheme.primaryYellow.withValues(alpha: 0.18),
          borderColor: AppTheme.primaryYellow.withValues(alpha: 0.4),
        );
      case _TestState.listening:
        return AnimatedBuilder(
          animation: _micPulse,
          key: const ValueKey('listen'),
          builder: (_, child) => Transform.scale(scale: _micPulse.value, child: child),
          child: _StateCard(
            icon: Icons.mic_rounded,
            iconColor: AppTheme.successCyan,
            label: 'Listening…',
            subLabel: 'Speak the character or say "stop" to exit',
            cardColor: AppTheme.successCyan.withValues(alpha: 0.15),
            borderColor: AppTheme.successCyan.withValues(alpha: 0.5),
          ),
        );
      case _TestState.evaluating:
        return _StateCard(
          key: const ValueKey('eval'),
          icon: Icons.psychology_rounded,
          iconColor: Colors.amber,
          label: 'You said: "${_spokenAnswer.isEmpty ? '…' : _spokenAnswer}"',
          subLabel: 'Evaluating…',
          cardColor: Colors.amber.withValues(alpha: 0.12),
          borderColor: Colors.amber.withValues(alpha: 0.35),
        );
      case _TestState.feedbackCorrect:
        return AnimatedBuilder(
          animation: _resultScale,
          key: const ValueKey('correct'),
          builder: (_, child) => Transform.scale(scale: _resultScale.value, child: child),
          child: _StateCard(
            icon: Icons.check_circle_rounded,
            iconColor: AppTheme.successCyan,
            label: 'Correct!',
            subLabel: 'Character $_expectedChar identified',
            cardColor: AppTheme.successCyan.withValues(alpha: 0.15),
            borderColor: AppTheme.successCyan.withValues(alpha: 0.5),
          ),
        );
      case _TestState.feedbackIncorrect:
        return AnimatedBuilder(
          animation: _resultScale,
          key: const ValueKey('incorrect'),
          builder: (_, child) => Transform.scale(scale: _resultScale.value, child: child),
          child: _StateCard(
            icon: Icons.cancel_rounded,
            iconColor: AppTheme.errorCoral,
            label: 'Incorrect',
            subLabel: 'You said "$_spokenAnswer"\nExpected: $_expectedChar',
            cardColor: AppTheme.errorCoral.withValues(alpha: 0.15),
            borderColor: AppTheme.errorCoral.withValues(alpha: 0.5),
          ),
        );
    }
  }

  Widget _buildScoreBar() {
    final pct = _total == 0 ? 0.0 : _correct / _total;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 36, vertical: 8),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Score: $_correct / $_total',
                style: const TextStyle(color: Colors.white70, fontSize: 13, fontWeight: FontWeight.w600),
              ),
              Text(
                '${(pct * 100).toStringAsFixed(0)}%',
                style: TextStyle(
                  color: pct >= 0.7
                      ? AppTheme.successCyan
                      : pct >= 0.4
                          ? Colors.amber
                          : AppTheme.errorCoral,
                  fontSize: 13,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _StateCard extends StatelessWidget {
  final IconData icon;
  final Color iconColor;
  final String label;
  final String subLabel;
  final Color cardColor;
  final Color? borderColor;

  const _StateCard({
    super.key,
    required this.icon,
    required this.iconColor,
    required this.label,
    required this.subLabel,
    required this.cardColor,
    this.borderColor,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 32),
      padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 28),
      decoration: BoxDecoration(
        color: cardColor,
        borderRadius: BorderRadius.circular(24),
        border: borderColor != null ? Border.all(color: borderColor!, width: 1.5) : null,
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 64, color: iconColor),
          const SizedBox(height: 16),
          Text(
            label,
            textAlign: TextAlign.center,
            style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Colors.white, height: 1.3),
          ),
          if (subLabel.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text(
              subLabel,
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 14, color: Colors.white.withValues(alpha: 0.6), height: 1.5),
            ),
          ],
        ],
      ),
    );
  }
}
