import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../theme/app_theme.dart';
import 'camera_service.dart';
import 'glass_device_service.dart';

/// Where frames come from, so Learning and Testing do not care whether the
/// wearer is using the glasses or the phone.
///
/// Both implementations return the same thing from [captureJpeg]: a full-size
/// JPEG in the orientation the ONNX pipeline already expects, so the prescan /
/// fingertip stages are untouched by the hardware swap.
abstract class CameraSource {
  /// Short human label for the status line, e.g. `Glasses` or `Phone`.
  String get label;

  bool get isReady;

  /// Non-null once the source has failed in a way the wearer should hear.
  String? get lastError;

  Future<bool> initialize();

  /// Captures one still. Returns null if the capture failed.
  Future<Uint8List?> captureJpeg();

  /// Live viewfinder for this source.
  Widget buildPreview();

  Future<void> dispose();
}

/// The existing on-device camera path, unchanged — still the fallback whenever
/// the glasses are absent or drop mid-session.
class PhoneCameraSource implements CameraSource {
  final CameraService _camera;

  PhoneCameraSource([CameraService? camera]) : _camera = camera ?? CameraService();

  /// Exposed so a screen can keep using [CameraService] directly if it needs
  /// something this interface deliberately does not cover.
  CameraService get service => _camera;

  String? _lastError;

  @override
  String get label => 'Phone';

  @override
  bool get isReady => _camera.isInitialized;

  @override
  String? get lastError => _lastError;

  @override
  Future<bool> initialize() async {
    await _camera.initialize();
    if (!_camera.isInitialized) {
      _lastError = 'Phone camera unavailable';
    }
    return _camera.isInitialized;
  }

  @override
  Future<Uint8List?> captureJpeg() async {
    final jpeg = await _camera.captureJpeg();
    if (jpeg == null) _lastError = 'Phone camera capture failed';
    return jpeg;
  }

  @override
  Widget buildPreview() {
    final c = _camera.controller;
    if (!_camera.isInitialized || c == null) {
      return const Center(
        child: CircularProgressIndicator(color: AppTheme.primaryYellow),
      );
    }
    return ColoredBox(
      color: Colors.black,
      child: Center(child: CameraPreview(c)),
    );
  }

  @override
  Future<void> dispose() => _camera.dispose();
}

/// Frames from the AI Glass.
///
/// Preview is the RTSP live feed, surfaced by the native bridge as a stream of
/// JPEG frames ([GlassDeviceService.videoFrames]) so the Dart side needs no
/// video plugin. Capture asks the glasses for a full-resolution still rather
/// than grabbing a preview frame — the live feed is downscaled and would cost
/// the dot detector real accuracy.
class GlassCameraSource implements CameraSource {
  final GlassDeviceService _glass;

  GlassCameraSource([GlassDeviceService? glass])
      : _glass = glass ?? GlassDeviceService.instance;

  Uint8List? _latestFrame;
  StreamSubscription<GlassVideoFrame>? _frameSub;
  final ValueNotifier<int> _frameTick = ValueNotifier<int>(0);
  String? _lastError;
  bool _streaming = false;

  @override
  String get label => 'Glasses';

  @override
  bool get isReady => _glass.isConnected && _streaming;

  @override
  String? get lastError => _lastError;

  @override
  Future<bool> initialize() async {
    if (!_glass.isConnected) {
      _lastError = 'Glasses not connected';
      return false;
    }

    _frameSub ??= _glass.videoFrames.listen((f) {
      _latestFrame = f.jpeg;
      _frameTick.value++;
    });

    _streaming = await _glass.startLiveStream();
    if (!_streaming) {
      _lastError = _glass.lastError ?? 'Could not start the glasses live feed';
    }
    return _streaming;
  }

  @override
  Future<Uint8List?> captureJpeg({
    Duration timeout = const Duration(seconds: 12),
  }) async {
    if (!_glass.isConnected) {
      _lastError = 'Glasses not connected';
      return null;
    }

    // Listen before issuing the command: takePhoto returns once the request is
    // queued, while the JPEG arrives asynchronously over the vendor channel.
    final completer = Completer<Uint8List?>();
    final ok = _glass.photos.listen((p) {
      if (!completer.isCompleted) completer.complete(p.jpeg);
    });
    final failed = _glass.events
        .where((e) => e is GlassPhotoFailed)
        .cast<GlassPhotoFailed>()
        .listen((e) {
      if (!completer.isCompleted) {
        _lastError = 'Glasses capture failed: ${e.reason}';
        completer.complete(null);
      }
    });
    final timer = Timer(timeout, () {
      if (!completer.isCompleted) {
        _lastError = 'Glasses capture timed out';
        completer.complete(null);
      }
    });

    try {
      if (!await _glass.takePhoto()) {
        _lastError = _glass.lastError ?? 'Glasses refused the capture command';
        return null;
      }
      return await completer.future;
    } finally {
      timer.cancel();
      await ok.cancel();
      await failed.cancel();
    }
  }

  @override
  Widget buildPreview() {
    return ValueListenableBuilder<int>(
      valueListenable: _frameTick,
      builder: (context, _, __) {
        final frame = _latestFrame;
        if (frame == null) {
          // No decoded frame yet. The RTSP session is up (the bridge emitted
          // LIVE_STREAM) but the H.264 -> JPEG decode path is not wired, so
          // say so plainly rather than spinning forever: capture works
          // regardless, because stills come over the vendor channel at full
          // resolution rather than out of this feed.
          return const ColoredBox(
            color: Colors.black,
            child: Center(
              child: Padding(
                padding: EdgeInsets.all(24),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.visibility_off_outlined,
                        color: AppTheme.primaryYellow, size: 40),
                    SizedBox(height: 12),
                    Text(
                      'GLASSES CAMERA ACTIVE',
                      style: TextStyle(
                        color: AppTheme.primaryYellow,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 1.1,
                      ),
                    ),
                    SizedBox(height: 6),
                    Text(
                      'Live preview is not available yet — capture still works.',
                      textAlign: TextAlign.center,
                      style: TextStyle(color: Colors.white70, fontSize: 13),
                    ),
                  ],
                ),
              ),
            ),
          );
        }
        return ColoredBox(
          color: Colors.black,
          child: Center(
            child: Image.memory(
              frame,
              gaplessPlayback: true,
              fit: BoxFit.contain,
            ),
          ),
        );
      },
    );
  }

  @override
  Future<void> dispose() async {
    await _frameSub?.cancel();
    _frameSub = null;
    _frameTick.dispose();
    if (_streaming) {
      await _glass.stopLiveStream();
      _streaming = false;
    }
  }
}

/// Picks the active [CameraSource] and swaps it when the glasses come or go.
///
/// Screens hold one of these instead of a [CameraService]; they read
/// [source] for capture/preview and listen to this notifier to rebuild when the
/// hardware underneath changes. Falling back is automatic: lose the glasses
/// mid-session and the phone camera takes over without the screen resetting its
/// prescan state.
class CameraSourceController extends ChangeNotifier {
  final GlassDeviceService _glass;

  CameraSourceController({GlassDeviceService? glass})
      : _glass = glass ?? GlassDeviceService.instance;

  CameraSource? _source;
  bool _switching = false;
  StreamSubscription<GlassEvent>? _glassSub;

  CameraSource? get source => _source;
  bool get isReady => _source?.isReady ?? false;
  bool get usingGlasses => _source is GlassCameraSource;
  String get label => _source?.label ?? '—';

  /// Brings up the best available source and starts watching for changes.
  Future<void> initialize() async {
    _glassSub ??= _glass.events.listen(_onGlassEvent);
    await _selectBest();
  }

  void _onGlassEvent(GlassEvent e) {
    if (e is! GlassConnectionChanged) return;
    final wantGlasses = e.state == GlassConnectionState.connected;
    if (wantGlasses != usingGlasses) _selectBest();
  }

  Future<void> _selectBest() async {
    if (_switching) return;
    _switching = true;
    try {
      final wantGlasses = _glass.isConnected;

      // Already on the right one and healthy — nothing to do.
      if (_source != null && usingGlasses == wantGlasses && isReady) return;

      final previous = _source;
      CameraSource next =
          wantGlasses ? GlassCameraSource(_glass) : PhoneCameraSource();

      var ok = await next.initialize();
      if (!ok && wantGlasses) {
        // Glasses claimed to be connected but the feed would not start; do not
        // strand the wearer with a dead viewfinder.
        debugPrint('[CameraSource] glasses feed failed, falling back to phone');
        await next.dispose();
        next = PhoneCameraSource();
        ok = await next.initialize();
      }

      _source = next;
      // Release the old one only after the new one is live, so the preview
      // never blanks between the two.
      await previous?.dispose();
      notifyListeners();
    } finally {
      _switching = false;
    }
  }

  /// Forces a re-evaluation, e.g. after the wearer reconnects manually.
  Future<void> refresh() => _selectBest();

  Future<Uint8List?> captureJpeg() async => _source?.captureJpeg();

  Widget buildPreview() =>
      _source?.buildPreview() ??
      const Center(child: CircularProgressIndicator(color: AppTheme.primaryYellow));

  @override
  void dispose() {
    _glassSub?.cancel();
    _source?.dispose();
    super.dispose();
  }
}
