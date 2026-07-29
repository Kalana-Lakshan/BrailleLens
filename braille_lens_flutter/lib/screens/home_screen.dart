import 'dart:async';
import 'package:flutter/material.dart';
import '../services/audio_service.dart';
import '../services/bluetooth_service.dart';
import '../theme/app_theme.dart';
import 'learning_screen.dart';
import 'testing_screen.dart';

/// Entry screen — accessible split-screen mode selector.
///
/// On launch:
///   1. Bluetooth scan runs in background.
///   2. TTS welcome prompt is spoken.
///   3. Voice-command loop listens for "learning" / "testing".
///   4. Two 50/50 full-height touch zones provide large tap targets.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  final AudioService _audioService = AudioService();
  final BluetoothService _btService = BluetoothService();

  bool _isListeningForCommand = false;
  bool _isCheckingBluetooth = true;
  BluetoothConnectionState _btState = BluetoothConnectionState.checking;
  bool _navigating = false;

  late AnimationController _pulseCtrl;
  late AnimationController _glowCtrl;
  late Animation<double> _pulseAnim;
  late Animation<double> _glowAnim;

  @override
  void initState() {
    super.initState();

    _pulseCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat(reverse: true);

    _glowCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2200),
    )..repeat(reverse: true);

    _pulseAnim = Tween<double>(begin: 0.93, end: 1.07).animate(
      CurvedAnimation(parent: _pulseCtrl, curve: Curves.easeInOut),
    );
    _glowAnim = Tween<double>(begin: 0.25, end: 0.75).animate(
      CurvedAnimation(parent: _glowCtrl, curve: Curves.easeInOut),
    );

    _initialize();
  }

  // ── Initialization ────────────────────────────────────────────────────────────

  Future<void> _initialize() async {
    // Run BT scan in background — update indicator when done
    _btService.scanForGlasses().then((found) {
      if (!mounted) return;
      setState(() {
        _isCheckingBluetooth = false;
        _btState = _btService.state;
      });
      if (found) {
        _audioService.hapticDouble();
        _audioService.speak('BrailleLens glasses connected.');
      } else {
        _audioService.hapticLight();
      }
    });

    // Brief pause then speak welcome
    await Future.delayed(const Duration(milliseconds: 700));
    if (!mounted) return;

    await _audioService.speak(
      'Welcome to BrailleLens. '
      "Tap the left half of the screen for Learning Mode, "
      "or the right half for Testing Mode. "
      "You can also say 'Learning' or 'Testing'.",
    );

    _startVoiceCommandLoop();
  }

  // ── Voice Command Loop ────────────────────────────────────────────────────────

  Future<void> _startVoiceCommandLoop() async {
    while (mounted && !_navigating) {
      if (!mounted || _navigating) break;

      setState(() => _isListeningForCommand = true);

      final command = await _audioService.listenForAnswer(
        timeout: const Duration(seconds: 8),
      );

      if (!mounted || _navigating) break;
      setState(() => _isListeningForCommand = false);

      if (command != null) {
        if (command.contains('learn')) {
          await _navigateTo('learning');
          break;
        } else if (command.contains('test')) {
          await _navigateTo('testing');
          break;
        }
      }

      // Brief gap before re-opening mic
      await Future.delayed(const Duration(milliseconds: 500));
    }
  }

  // ── Navigation ────────────────────────────────────────────────────────────────

  Future<void> _navigateTo(String mode) async {
    if (_navigating || !mounted) return;
    setState(() => _navigating = true);

    _audioService.stopListening();
    await _audioService.stopSpeech();
    await _audioService.hapticMedium();

    if (!mounted) return;

    await Navigator.push(
      context,
      PageRouteBuilder(
        pageBuilder: (_, __, ___) => mode == 'learning'
            ? LearningScreen(audioService: _audioService)
            : TestingScreen(audioService: _audioService),
        transitionsBuilder: (_, anim, __, child) => FadeTransition(
          opacity: anim,
          child: child,
        ),
        transitionDuration: const Duration(milliseconds: 400),
      ),
    );

    if (mounted) {
      setState(() => _navigating = false);
      // Re-start voice loop on return
      await Future.delayed(const Duration(milliseconds: 600));
      await _audioService.speak(
        "Back at main menu. Say 'Learning' or 'Testing', or tap a side.",
      );
      _startVoiceCommandLoop();
    }
  }

  // ── Dispose ───────────────────────────────────────────────────────────────────

  @override
  void dispose() {
    _pulseCtrl.dispose();
    _glowCtrl.dispose();
    _audioService.dispose();
    _btService.dispose();
    super.dispose();
  }

  // ── Build ─────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;
    return Scaffold(
      backgroundColor: AppTheme.backgroundBlack,
      body: Stack(
        children: [
          // Radial background glow
          Positioned.fill(
            child: DecoratedBox(
              decoration: BoxDecoration(
                gradient: RadialGradient(
                  center: const Alignment(0, -0.3),
                  radius: 1.4,
                  colors: [
                    const Color(0xFF1A0A3D),
                    AppTheme.backgroundBlack,
                  ],
                ),
              ),
            ),
          ),

          // Split touch zones
          Row(
            children: [
              // ── LEFT: Learning Mode ──────────────────────────────────────
              Expanded(
                child: Semantics(
                  label: 'Learning Mode button. Tap to enter Learning Mode.',
                  button: true,
                  child: GestureDetector(
                    onTap: () => _navigateTo('learning'),
                    child: _ModeZone(
                      title: 'LEARNING\nMODE',
                      subtitle: 'Live camera · Mock TTS',
                      icon: Icons.school_rounded,
                      gradientBegin: AppTheme.learningZoneBlue,
                      gradientEnd: AppTheme.learningZoneBlue,
                      glowAnim: _glowAnim,
                      alignLeft: true,
                    ),
                  ),
                ),
              ),

              // Divider
              Container(
                width: 1.5,
                color: Colors.white.withValues(alpha: 0.1),
              ),

              // ── RIGHT: Testing Mode ──────────────────────────────────────
              Expanded(
                child: Semantics(
                  label: 'Testing Mode button. Tap to enter Testing Mode.',
                  button: true,
                  child: GestureDetector(
                    onTap: () => _navigateTo('testing'),
                    child: _ModeZone(
                      title: 'TESTING\nMODE',
                      subtitle: 'Voice guess · Audio feedback',
                      icon: Icons.record_voice_over_rounded,
                      gradientBegin: AppTheme.testingZoneTeal,
                      gradientEnd: AppTheme.testingZoneTeal,
                      glowAnim: _glowAnim,
                      alignLeft: false,
                    ),
                  ),
                ),
              ),
            ],
          ),

          // Status bar (BT + title)
          Positioned(
            top: 0, left: 0, right: 0,
            child: _StatusBar(
              isChecking: _isCheckingBluetooth,
              btState: _btState,
              btDevice: _btService.connectedDevice,
            ),
          ),

          // Animated center logo (non-interactive overlay)
          Positioned.fill(
            child: IgnorePointer(
              child: Center(
                child: AnimatedBuilder(
                  animation: _pulseAnim,
                  builder: (_, __) => Transform.scale(
                    scale: _pulseAnim.value,
                    child: _LogoBadge(),
                  ),
                ),
              ),
            ),
          ),

          // Bottom mic indicator
          Positioned(
            bottom: 36, left: 0, right: 0,
            child: _MicStatus(isListening: _isListeningForCommand),
          ),
        ],
      ),
    );
  }
}

// ── Sub-widgets ───────────────────────────────────────────────────────────────

class _ModeZone extends StatelessWidget {
  final String title;
  final String subtitle;
  final IconData icon;
  final Color gradientBegin;
  final Color gradientEnd;
  final Animation<double> glowAnim;
  final bool alignLeft;

  const _ModeZone({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.gradientBegin,
    required this.gradientEnd,
    required this.glowAnim,
    required this.alignLeft,
  });

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: glowAnim,
      builder: (_, __) => Container(
        width: double.infinity,
        height: double.infinity,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: alignLeft ? Alignment.centerLeft : Alignment.centerRight,
            end: alignLeft ? Alignment.centerRight : Alignment.centerLeft,
            colors: [
              gradientBegin.withValues(alpha: 0.9),
              gradientEnd.withValues(alpha: 0.45 + glowAnim.value * 0.15),
            ],
          ),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, size: 76, color: Colors.white.withValues(alpha: 0.92)),
            const SizedBox(height: 22),
            Text(
              title,
              textAlign: TextAlign.center,
              style: const TextStyle(
                fontSize: 26,
                fontWeight: FontWeight.w800,
                color: Colors.white,
                letterSpacing: 2.0,
                height: 1.3,
              ),
            ),
            const SizedBox(height: 10),
            Text(
              subtitle,
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 12,
                color: Colors.white.withValues(alpha: 0.6),
                height: 1.5,
                letterSpacing: 0.3,
              ),
            ),
            const SizedBox(height: 30),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 7),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: Colors.white.withValues(alpha: 0.22),
                ),
              ),
              child: const Text(
                'TAP  OR  SPEAK',
                style: TextStyle(
                  fontSize: 10,
                  color: Colors.white,
                  letterSpacing: 2.0,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _LogoBadge extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      width: 68,
      height: 68,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: AppTheme.backgroundBlack,
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.18),
          width: 1.5,
        ),
        boxShadow: [
          BoxShadow(
            color: AppTheme.primaryYellow.withValues(alpha: 0.55),
            blurRadius: 28,
            spreadRadius: 6,
          ),
        ],
      ),
      child: const Icon(Icons.fingerprint, size: 34, color: AppTheme.primaryYellow),
    );
  }
}

class _StatusBar extends StatelessWidget {
  final bool isChecking;
  final BluetoothConnectionState btState;
  final BrailleDevice? btDevice;

  const _StatusBar({
    required this.isChecking,
    required this.btState,
    this.btDevice,
  });

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
        child: Row(
          children: [
            const Text(
              'BrailleLens',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Colors.white,
                letterSpacing: 0.8,
              ),
            ),
            const Spacer(),
            _buildBtIndicator(),
          ],
        ),
      ),
    );
  }

  Widget _buildBtIndicator() {
    if (isChecking) {
      return Row(children: [
        const SizedBox(
          width: 13,
          height: 13,
          child: CircularProgressIndicator(
            strokeWidth: 1.8,
            color: Colors.white54,
          ),
        ),
        const SizedBox(width: 7),
        const Text(
          'Scanning...',
          style: TextStyle(fontSize: 12, color: Colors.white38),
        ),
      ]);
    }
    final connected = btState == BluetoothConnectionState.connected;
    return Row(children: [
      Icon(
        connected ? Icons.bluetooth_connected : Icons.bluetooth_disabled,
        size: 15,
        color: connected ? const Color(0xFF00BFA5) : Colors.white30,
      ),
      const SizedBox(width: 6),
      Text(
        connected ? (btDevice?.name ?? 'Glasses') : 'No device',
        style: TextStyle(
          fontSize: 12,
          color: connected ? const Color(0xFF00BFA5) : Colors.white30,
        ),
      ),
    ]);
  }
}

class _MicStatus extends StatelessWidget {
  final bool isListening;
  const _MicStatus({required this.isListening});

  @override
  Widget build(BuildContext context) {
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 300),
      child: isListening
          ? Row(
              key: const ValueKey('on'),
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(Icons.mic, color: AppTheme.successCyan, size: 17),
                const SizedBox(width: 8),
                Text(
                  'Listening… say "Learning" or "Testing"',
                  style: TextStyle(
                    fontSize: 12,
                    color: AppTheme.successCyan.withValues(alpha: 0.9),
                  ),
                ),
              ],
            )
          : Row(
              key: const ValueKey('off'),
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.touch_app, color: Colors.white30, size: 15),
                const SizedBox(width: 8),
                const Text(
                  'Tap a side or speak a mode name',
                  style: TextStyle(fontSize: 12, color: Colors.white30),
                ),
              ],
            ),
    );
  }
}
