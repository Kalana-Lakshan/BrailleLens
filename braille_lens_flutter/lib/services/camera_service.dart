import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';

/// Manages the device camera for Learning and Testing still captures.
///
/// Screens must call [initialize] on entry and [dispose] on exit.
class CameraService {
  CameraController? _controller;
  bool _isInitialized = false;

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

  /// Releases the camera controller.
  Future<void> dispose() async {
    await _controller?.dispose();
    _controller = null;
    _isInitialized = false;
    debugPrint('[CameraService] Disposed.');
  }
}
