import 'dart:async';
import 'dart:math';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'audio_service.dart';

/// Manages the device camera and the mock character-detection simulation loop.
///
/// Architecture principle: [CameraService] owns the [CameraController] lifecycle.
/// Screens must call [initialize] on entry and [dispose] on exit.
///
/// Integration Hook: To replace mock predictions with real inference, swap
/// [_announceMockPrediction] with a call to [ClassifierService.predictFromFrame].
class CameraService {
  CameraController? _controller;
  Timer? _mockSimulationTimer;
  bool _isInitialized = false;

  // Characters used by the mock simulation loop
  static const List<String> _mockCharacters = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
  ];
  final Random _random = Random();

  CameraController? get controller => _controller;
  bool get isInitialized => _isInitialized;

  /// Initializes the rear-facing camera at medium resolution.
  Future<void> initialize() async {
    if (_isInitialized) return;
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        debugPrint('[CameraService] No cameras found on device.');
        return;
      }

      // Prefer rear camera; fall back to the first available
      final camera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      _controller = CameraController(
        camera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      await _controller!.initialize();
      _isInitialized = true;
      debugPrint('[CameraService] Camera initialized (${camera.name}).');
    } catch (e) {
      debugPrint('[CameraService] Initialization error: $e');
    }
  }

  /// Starts the periodic mock ML prediction loop.
  ///
  /// Every [intervalSeconds] the service picks a random character, speaks it
  /// via TTS, and calls [onCharacterDetected] so the UI can display it.
  ///
  /// Integration Hook: Replace [_announceMockPrediction] body with real inference:
  /// ```dart
  /// final frame = await _controller!.takePicture();
  /// final bytes = await frame.readAsBytes();
  /// final result = await classifierService.predict(bytes);
  /// onCharacterDetected?.call(result.character.toUpperCase());
  /// audioService.speak('Character ${result.character.toUpperCase()} detected');
  /// ```
  void startMockSimulationLoop(
    AudioService audioService, {
    int intervalSeconds = 4,
    void Function(String character)? onCharacterDetected,
  }) {
    stopMockSimulationLoop();
    debugPrint('[CameraService] Mock simulation loop started.');
    _mockSimulationTimer = Timer.periodic(
      Duration(seconds: intervalSeconds),
      (_) => _announceMockPrediction(audioService, onCharacterDetected),
    );
  }

  void _announceMockPrediction(
    AudioService audioService,
    void Function(String character)? onCharacterDetected,
  ) {
    final char = _mockCharacters[_random.nextInt(_mockCharacters.length)];
    debugPrint('[CameraService] Mock prediction: $char');
    onCharacterDetected?.call(char);
    audioService.speak('Simulated character $char');
  }

  /// Capture a still JPEG from the live preview (stage 1 / stage 2 photos).
  Future<Uint8List?> captureJpeg() async {
    if (_controller == null || !_isInitialized) return null;
    try {
      final file = await _controller!.takePicture();
      return await file.readAsBytes();
    } catch (e) {
      debugPrint('[CameraService] capture error: $e');
      return null;
    }
  }

  /// Stops the mock simulation loop without releasing the camera.
  void stopMockSimulationLoop() {
    _mockSimulationTimer?.cancel();
    _mockSimulationTimer = null;
  }

  /// Releases the camera controller and stops the simulation loop.
  Future<void> dispose() async {
    stopMockSimulationLoop();
    await _controller?.dispose();
    _controller = null;
    _isInitialized = false;
    debugPrint('[CameraService] Disposed.');
  }
}
