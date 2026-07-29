import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../services/audio_service.dart';
import '../services/camera_service.dart';
import '../theme/app_theme.dart';

// ── State Machine ─────────────────────────────────────────────────────────────

enum _TestState {
  idle,
  prompting,
  listening,
  evaluating,
  feedbackCorrect,
  feedbackIncorrect,
}

// ── Mock character list (replace with real label list later) ──────────────────

const List<String> _mockLabels = [
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
  'K', 'L', 'M', 'N', 'O',
];

/// Testing Mode — earcon-guided, voice-driven Braille character quiz.
///
/// Workflow loop:
///   1. TTS: "Please state the character under your finger."
///   2. Earcon: start-listening tone.
///   3. Open microphone (listen until silence/timeout or "stop" keyword).
///   4. Earcon: stop-listening tone.
///   5. Evaluate spoken answer vs. mock expected character.
///   6. Success: success earcon + haptic double + "Correct!"
///      Failure: error earcon + haptic error + "Incorrect. You said X, but the character is Y."
///   7. 2.5 s pause → next character → repeat.
///
/// Exit: double-tap anywhere, or say "stop" during listening → returns to HomeScreen.
class TestingScreen extends StatefulWidget {
  /// Shared [AudioService] from [HomeScreen]. Not disposed here.
  final AudioService audioService;

  const TestingScreen({super.key, required this.audioService});

  @override
  State<TestingScreen> createState() => _TestingScreenState();
}

class _TestingScreenState extends State<TestingScreen>
    with TickerProviderStateMixin {
  final CameraService _cameraService = CameraService();

  _TestState _testState = _TestState.idle;
  bool _loopActive = false;
  bool _isExiting = false;

  String _expectedChar = _mockLabels[0];
  String _spokenAnswer = '';
  int _correct = 0;
  int _total = 0;
  int _charIndex = 0;

  // Animations
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

    _setup();
  }

  // ── Setup ─────────────────────────────────────────────────────────────────────

  Future<void> _setup() async {
    await _cameraService.initialize();
    if (!mounted) return;

    await widget.audioService.hapticMedium();
    await widget.audioService.speak(
      'Testing Mode active. '
      'I will name a Braille character — say the letter when prompted. '
      'Say "stop" at any time to exit. '
      'Double-tap the screen to return to the main menu.',
    );

    if (mounted) {
      setState(() => _loopActive = true);
      _runTestLoop();
    }
  }

  // ── Test Loop ─────────────────────────────────────────────────────────────────

  Future<void> _runTestLoop() async {
    while (mounted && _loopActive) {
      // Pick next expected character (cycle through list)
      _expectedChar = _mockLabels[_charIndex % _mockLabels.length];
      _charIndex++;

      // ── Step 1: Prompt ────────────────────────────────────────────────────
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

      // ── Step 2: Start-listening earcon ────────────────────────────────────
      setState(() => _testState = _TestState.listening);
      await widget.audioService.playStartListeningTone();
      await Future.delayed(const Duration(milliseconds: 350));

      if (!mounted || !_loopActive) break;

      // ── Step 3: Listen until stop keyword or timeout ──────────────────────
      final spoken = await widget.audioService.listenUntilStop(
        timeout: const Duration(seconds: 10),
      );

      if (!mounted || !_loopActive) break;

      // ── Step 4: Stop-listening earcon ─────────────────────────────────────
      await widget.audioService.playStopListeningTone();
      await widget.audioService.hapticLight();

      // Check if user said "stop" as an exit command (null result means stop fired)
      // listenUntilStop returns null when "stop" was the only word spoken
      if (spoken == null && _testState == _TestState.listening) {
        // A timeout with no speech — skip this round
        await widget.audioService.speak("I didn't catch that. Let's try again.");
        await Future.delayed(const Duration(seconds: 1));
        continue;
      }

      if (!mounted || !_loopActive) break;

      // ── Step 5: Evaluate ──────────────────────────────────────────────────
      setState(() {
        _testState = _TestState.evaluating;
        _spokenAnswer = spoken ?? '';
        _total++;
      });

      await Future.delayed(const Duration(milliseconds: 250));
      if (!mounted || !_loopActive) break;

      final isCorrect = spoken != null &&
          spoken.isNotEmpty &&
          spoken.toLowerCase().contains(_expectedChar.toLowerCase());

      // ── Step 6: Feedback ──────────────────────────────────────────────────
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

      // ── Step 7: Pause before next round ──────────────────────────────────
      await Future.delayed(const Duration(milliseconds: 2500));
      if (!mounted || !_loopActive) break;
    }

    if (mounted) setState(() => _testState = _TestState.idle);
  }

  // ── Exit ─────────────────────────────────────────────────────────────────────

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

  // ── Dispose ───────────────────────────────────────────────────────────────────

  @override
  void dispose() {
    _loopActive = false;
    _micPulseCtrl.dispose();
    _resultCtrl.dispose();
    _cameraService.dispose();
    super.dispose();
  }

  // ── Build ─────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Semantics(
      label: 'Testing Mode. '
          'Listen for the prompt then speak the character. '
          'Double-tap anywhere to exit.',
      child: GestureDetector(
        onDoubleTap: _exitScreen,
        child: Scaffold(
          backgroundColor: Colors.black,
          body: Stack(
            fit: StackFit.expand,
            children: [
              // Camera feed (background)
              if (_cameraService.isInitialized &&
                  _cameraService.controller != null)
                Opacity(
                  opacity: 0.35,
                  child: CameraPreview(_cameraService.controller!),
                )
              else
                Container(color: AppTheme.backgroundBlack),

              // Dark gradient overlay
              Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                    colors: [
                      Colors.black.withValues(alpha: 0.65),
                      Colors.black.withValues(alpha: 0.85),
                    ],
                  ),
                ),
              ),

              // Main content
              SafeArea(
                child: Column(
                  children: [
                    _buildTopBar(),
                    Expanded(child: _buildCenterContent()),
                    _buildScoreBar(),
                    const SizedBox(height: 24),
                    _buildExitHint(),
                    const SizedBox(height: 28),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ── Private Widgets ───────────────────────────────────────────────────────────

  Widget _buildTopBar() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
      child: Row(
        children: [
          GestureDetector(
            onTap: _exitScreen,
            child: Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: Colors.white.withValues(alpha: 0.18),
                ),
              ),
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.arrow_back_ios_new,
                      color: Colors.white, size: 13),
                  SizedBox(width: 4),
                  Text('Exit',
                      style: TextStyle(color: Colors.white, fontSize: 13)),
                ],
              ),
            ),
          ),
          const Spacer(),
          const Text(
            'TESTING MODE',
            style: TextStyle(
              color: Colors.white60,
              fontSize: 13,
              fontWeight: FontWeight.w600,
              letterSpacing: 1.5,
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
        switchInCurve: Curves.easeOut,
        switchOutCurve: Curves.easeIn,
        child: _buildStateWidget(),
      ),
    );
  }

  Widget _buildStateWidget() {
    switch (_testState) {
      case _TestState.idle:
        return _StateCard(
          key: const ValueKey('idle'),
          icon: Icons.hourglass_top_rounded,
          iconColor: Colors.white38,
          label: 'Preparing…',
          subLabel: '',
          cardColor: Colors.white.withValues(alpha: 0.06),
        );

      case _TestState.prompting:
        return _StateCard(
          key: const ValueKey('prompt'),
          icon: Icons.volume_up_rounded,
          iconColor: AppTheme.primaryYellow,
          label: 'Listen for the prompt',
          subLabel: 'Braille character approaching…',
          cardColor: AppTheme.primaryYellow.withValues(alpha: 0.18),
          borderColor: AppTheme.primaryYellow.withValues(alpha: 0.4),
        );

      case _TestState.listening:
        return AnimatedBuilder(
          animation: _micPulse,
          key: const ValueKey('listen'),
          builder: (_, child) => Transform.scale(
            scale: _micPulse.value,
            child: child,
          ),
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
          builder: (_, child) => Transform.scale(
            scale: _resultScale.value,
            child: child,
          ),
          child: _StateCard(
            icon: Icons.check_circle_rounded,
            iconColor: AppTheme.successCyan,
            label: 'Correct!',
            subLabel: 'Character $_expectedChar identified ✓',
            cardColor: AppTheme.successCyan.withValues(alpha: 0.15),
            borderColor: AppTheme.successCyan.withValues(alpha: 0.5),
          ),
        );

      case _TestState.feedbackIncorrect:
        return AnimatedBuilder(
          animation: _resultScale,
          key: const ValueKey('incorrect'),
          builder: (_, child) => Transform.scale(
            scale: _resultScale.value,
            child: child,
          ),
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
      padding: const EdgeInsets.symmetric(horizontal: 36),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Score: $_correct / $_total',
                style: const TextStyle(
                  color: Colors.white70,
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
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
          const SizedBox(height: 6),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: pct,
              minHeight: 5,
              backgroundColor: Colors.white12,
              valueColor: AlwaysStoppedAnimation<Color>(
                pct >= 0.7
                    ? AppTheme.successCyan
                    : pct >= 0.4
                        ? Colors.amber
                        : AppTheme.errorCoral,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildExitHint() {
    return Text(
      'Double-tap to exit  ·  Say "stop" to end recording',
      style: TextStyle(
        fontSize: 11,
        color: Colors.white.withValues(alpha: 0.3),
        letterSpacing: 0.3,
      ),
    );
  }
}

// ── Reusable state card ───────────────────────────────────────────────────────

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
      padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 36),
      decoration: BoxDecoration(
        color: cardColor,
        borderRadius: BorderRadius.circular(24),
        border: borderColor != null
            ? Border.all(color: borderColor!, width: 1.5)
            : null,
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 72, color: iconColor),
          const SizedBox(height: 20),
          Text(
            label,
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.bold,
              color: Colors.white,
              height: 1.3,
            ),
          ),
          if (subLabel.isNotEmpty) ...[
            const SizedBox(height: 10),
            Text(
              subLabel,
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 14,
                color: Colors.white.withValues(alpha: 0.6),
                height: 1.5,
              ),
            ),
          ],
        ],
      ),
    );
  }
}
