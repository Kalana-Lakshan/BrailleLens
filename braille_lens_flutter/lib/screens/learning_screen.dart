import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../services/audio_service.dart';
import '../services/camera_service.dart';
import '../theme/app_theme.dart';

/// Learning Mode — displays a full-screen live camera feed while the
/// mock ML simulation loop periodically announces predicted Braille characters
/// via TTS.
///
/// Exit: Double-tap anywhere → TTS "Returning to main menu" → pop.
class LearningScreen extends StatefulWidget {
  /// Shared [AudioService] instance from [HomeScreen]. This screen does NOT
  /// dispose it — HomeScreen owns its lifecycle.
  final AudioService audioService;

  const LearningScreen({super.key, required this.audioService});

  @override
  State<LearningScreen> createState() => _LearningScreenState();
}

class _LearningScreenState extends State<LearningScreen>
    with TickerProviderStateMixin {
  final CameraService _cameraService = CameraService();

  bool _isInitialized = false;
  String _currentChar = '—';
  bool _isExiting = false;

  late AnimationController _liveCtrl;
  late AnimationController _charCtrl;
  late Animation<double> _charScale;

  @override
  void initState() {
    super.initState();

    _liveCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    )..repeat(reverse: true);

    _charCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 350),
    );

    _charScale = TweenSequence<double>([
      TweenSequenceItem(tween: Tween(begin: 1.0, end: 1.35), weight: 1),
      TweenSequenceItem(tween: Tween(begin: 1.35, end: 1.0), weight: 1),
    ]).animate(CurvedAnimation(parent: _charCtrl, curve: Curves.easeOut));

    _setup();
  }

  // ── Setup ─────────────────────────────────────────────────────────────────────

  Future<void> _setup() async {
    await _cameraService.initialize();
    if (!mounted) return;

    setState(() => _isInitialized = true);

    // Single haptic pulse: camera is locked
    await widget.audioService.hapticMedium();

    _cameraService.startMockSimulationLoop(
      widget.audioService,
      intervalSeconds: 4,
      onCharacterDetected: (char) {
        if (!mounted) return;
        setState(() => _currentChar = char);
        _charCtrl.forward(from: 0);
      },
    );

    await widget.audioService.speak(
      'Learning Mode active. '
      'Hold your Braille card under the camera. '
      'Characters will be announced every few seconds. '
      'Double tap to exit.',
    );
  }

  // ── Exit ─────────────────────────────────────────────────────────────────────

  Future<void> _exitScreen() async {
    if (_isExiting) return;
    setState(() => _isExiting = true);

    _cameraService.stopMockSimulationLoop();
    widget.audioService.stopListening();
    await widget.audioService.stopSpeech();
    await widget.audioService.hapticLight();
    await widget.audioService.speak('Returning to main menu.');

    if (mounted) Navigator.pop(context);
  }

  // ── Dispose ───────────────────────────────────────────────────────────────────

  @override
  void dispose() {
    _liveCtrl.dispose();
    _charCtrl.dispose();
    _cameraService.dispose();
    super.dispose();
  }

  // ── Build ─────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Semantics(
      label: 'Learning Mode. Camera is live. Double tap anywhere to exit.',
      child: GestureDetector(
        onDoubleTap: _exitScreen,
        child: Scaffold(
          backgroundColor: Colors.black,
          body: Stack(
            fit: StackFit.expand,
            children: [
              // ── Camera Preview ────────────────────────────────────────────
              if (_isInitialized && _cameraService.controller != null)
                CameraPreview(_cameraService.controller!)
              else
                _buildLoadingView(),

              // ── Top overlay ───────────────────────────────────────────────
              Positioned(
                top: 0, left: 0, right: 0,
                child: _buildTopOverlay(),
              ),

              // ── Bottom overlay ────────────────────────────────────────────
              Positioned(
                bottom: 0, left: 0, right: 0,
                child: _buildBottomOverlay(),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ── Private Widgets ───────────────────────────────────────────────────────────

  Widget _buildLoadingView() {
    return Container(
      color: AppTheme.backgroundBlack,
      child: const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(color: AppTheme.primaryYellow),
            SizedBox(height: 16),
            Text(
              'Initializing camera…',
              style: TextStyle(color: Colors.white60, fontSize: 15),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTopOverlay() {
    return SafeArea(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Colors.black.withValues(alpha: 0.75),
              Colors.transparent,
            ],
          ),
        ),
        child: Row(
          children: [
            // Exit tap target
            GestureDetector(
              onTap: _exitScreen,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                    color: Colors.white.withValues(alpha: 0.2),
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
            // LIVE indicator
            AnimatedBuilder(
              animation: _liveCtrl,
              builder: (_, __) => Row(children: [
                Container(
                  width: 8,
                  height: 8,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: Color.lerp(
                      Colors.red,
                      Colors.red.shade200,
                      _liveCtrl.value,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.red
                            .withValues(alpha: 0.5 * _liveCtrl.value),
                        blurRadius: 6,
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 6),
                const Text(
                  'LIVE',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 1.5,
                  ),
                ),
              ]),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBottomOverlay() {
    return Container(
      padding: const EdgeInsets.fromLTRB(24, 28, 24, 36),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.bottomCenter,
          end: Alignment.topCenter,
          colors: [
            Colors.black.withValues(alpha: 0.88),
            Colors.transparent,
          ],
        ),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Character display card
          AnimatedBuilder(
            animation: _charScale,
            builder: (_, child) => Transform.scale(
              scale: _charScale.value,
              child: child,
            ),
            child: Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 28, vertical: 14),
              decoration: BoxDecoration(
                color: AppTheme.primaryYellow.withValues(alpha: 0.22),
                borderRadius: BorderRadius.circular(18),
                border: Border.all(
                  color: AppTheme.primaryYellow.withValues(alpha: 0.55),
                  width: 1.5,
                ),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.hearing,
                      color: AppTheme.primaryYellow, size: 22),
                  const SizedBox(width: 12),
                  Text(
                    'Detected: $_currentChar',
                    style: const TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                      letterSpacing: 1,
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 14),
          Text(
            'Double-tap anywhere to exit',
            style: TextStyle(
              fontSize: 12,
              color: Colors.white.withValues(alpha: 0.4),
            ),
          ),
        ],
      ),
    );
  }
}
