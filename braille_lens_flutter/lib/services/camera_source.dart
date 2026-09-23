import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:media_kit/media_kit.dart';
import 'package:media_kit_video/media_kit_video.dart';
import 'package:path_provider/path_provider.dart';

import '../theme/app_theme.dart';
import 'camera_service.dart';
import 'glass_device_service.dart';

/// Where a still from the glasses comes from.
///
/// The two are not interchangeable, and the choice decides what the Braille
/// pipeline gets to work with:
///
/// * [vendorChannel] asks the device for a full-resolution original and waits
///   for it to transfer over Bluetooth SPP. Seconds, not milliseconds — but it
///   is the sensor image, which is what dot detection needs. This is why the
///   integration was built this way and it stays the default.
/// * [videoSnapshot] grabs the frame libmpv is already showing. It returns
///   almost instantly and needs no transfer, but the RTSP feed is downscaled,
///   so fine dot structure may not survive. Good for framing checks, rapid
///   dataset collection and anything where latency beats resolution.
enum GlassCaptureMode { vendorChannel, videoSnapshot }

/// Builds the libmpv player for the glasses RTSP preview, with audio off.
///
/// The glasses stream carries an audio track. If libmpv plays it, it opens an
/// OpenSL ES output track, and with the glasses as the phone's Bluetooth
/// headset that can pull the SCO/HFP call path up and down — the "call ended"
/// tone on the glasses, and the drops in the preview. The preview only needs
/// pictures.
///
/// media_kit 1.2.6's [PlayerConfiguration] has no audio switch (`muted` only
/// zeroes the volume; the output still opens), and it forces `ao=opensles` on
/// physical devices during init. So this sets the libmpv properties directly:
/// `aid=no` means no audio track is ever selected, so no decoder and no audio
/// output is created; `ao=null` is the backstop if a track gets selected
/// anyway. `setProperty` waits for player init, so these land after media_kit's
/// own defaults and before the caller's `open()`.
Future<Player> createGlassPreviewPlayer() async {
  final player = Player(
    configuration: const PlayerConfiguration(
      // Small buffer: this is a viewfinder, so latency matters far more than
      // smoothing over a dropped frame.
      bufferSize: 2 * 1024 * 1024,
    ),
  );
  final platform = player.platform;
  if (platform is NativePlayer) {
    await platform.setProperty('aid', 'no');
    await platform.setProperty('ao', 'null');
  }
  return player;
}

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
/// Preview is the RTSP feed the glasses publish at `rtsp://<ip>:554` once the
/// native bridge has started a live session — the same URL the Realtek
/// reference app plays. It is rendered with `media_kit` (libmpv), which does
/// the H.264 depacketise and decode, so nothing here needs a custom native
/// decoder.
///
/// Capture deliberately does *not* grab a preview frame: it asks the glasses
/// for a full-resolution still over the vendor channel, because the live feed
/// is downscaled and would cost the dot detector real accuracy.
class GlassCameraSource implements CameraSource {
  final GlassDeviceService _glass;

  GlassCameraSource([GlassDeviceService? glass])
      : _glass = glass ?? GlassDeviceService.instance;

  StreamSubscription<GlassLiveStream>? _streamSub;
  Player? _player;
  Future<Player>? _playerFuture;

  /// Set while the phone still has to join the glasses Wi-Fi by hand.
  String? _joinHint;
  VideoController? _videoController;

  /// Rebuilds the preview when the player is created or the URL changes.
  final ValueNotifier<int> _playerTick = ValueNotifier<int>(0);

  String? _rtspUrl;
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

    // The URL is only known once the device reports the session is up, so
    // subscribe before asking for it.
    _streamSub ??= _glass.liveStreams.listen(_onLiveStream);

    _streaming = await _glass.startLiveStream();
    if (!_streaming) {
      _lastError = _glass.lastError ?? 'Could not start the glasses live feed';
    }
    return _streaming;
  }

  void _onLiveStream(GlassLiveStream e) {
    if (e.rtspUrl.isEmpty) {
      // Station mode only: the glasses joined our hotspot but reported no IP.
      _lastError = 'Live feed has no reachable address';
      _playerTick.value++;
      return;
    }
    // AP mode with a failed automatic join: still open the player — if the
    // wearer joins the glasses Wi-Fi by hand, libmpv's retry picks it up —
    // but keep the hint visible until then.
    _joinHint = e.wifiConnected
        ? null
        : 'Join the glasses Wi-Fi "${e.ssid ?? ''}" to see the feed';
    _lastError = _joinHint;
    if (e.rtspUrl == _rtspUrl && _player != null) {
      _playerTick.value++;
      return;
    }
    _rtspUrl = e.rtspUrl;
    unawaited(_openPlayer(e.rtspUrl));
  }

  Future<void> _openPlayer(String url) async {
    try {
      // Audio off: see [createGlassPreviewPlayer]. The future is shared so two
      // quick LIVE_STREAM events cannot build two players.
      final player =
          _player ?? await (_playerFuture ??= createGlassPreviewPlayer());
      _player = player;
      _videoController ??= VideoController(player);
      await player.open(Media(url), play: true);
      // open() returning only means libmpv accepted the URL; a pending
      // Wi-Fi join is still the thing to tell the wearer about.
      _lastError = _joinHint;
    } catch (e) {
      _lastError = 'Could not play the glasses feed: $e';
      debugPrint('[GlassCameraSource] $_lastError');
    }
    _playerTick.value++;
  }

  /// Which path [captureJpeg] takes. See [GlassCaptureMode] — the default is
  /// the full-resolution original, because that is what dot detection needs.
  GlassCaptureMode captureMode = GlassCaptureMode.vendorChannel;

  /// Guards against overlapping screenshots. mpv takes its own lock for the
  /// grab, so a burst of frame-button presses queued against it is the one way
  /// this stalls playback; the second press is dropped instead.
  bool _snapshotInFlight = false;

  /// Grabs the frame libmpv is currently showing, straight from the video
  /// pipeline — no round trip to the glasses and no file transfer.
  ///
  /// Returns null when the live feed is not up: this reads the decoder's
  /// current frame, so without a playing stream there is nothing to read.
  Future<Uint8List?> captureSnapshotJpeg() async {
    final player = _player;
    if (player == null || !_streaming) {
      _lastError = 'Live feed is not running — no frame to grab';
      return null;
    }
    if (_snapshotInFlight) {
      _lastError = 'A snapshot is already being taken';
      return null;
    }
    _snapshotInFlight = true;
    try {
      final bytes = await player.screenshot(format: 'image/jpeg');
      if (bytes == null || bytes.isEmpty) {
        // Most often the stream is connected but has not decoded a frame yet.
        _lastError = 'The video feed returned no frame';
        return null;
      }
      _lastError = _joinHint;
      return bytes;
    } catch (e) {
      _lastError = 'Snapshot from the video feed failed: $e';
      debugPrint('[GlassCameraSource] $_lastError');
      return null;
    } finally {
      _snapshotInFlight = false;
    }
  }

  /// [captureSnapshotJpeg] written to a file in the app cache, returning its
  /// path. For callers that want a file rather than bytes — dataset collection,
  /// or handing the image to something that reads from disk.
  ///
  /// Cache, not the gallery: this directory is app-private and the OS may clear
  /// it. Use [GlassDeviceService.saveImageToGallery] to publish a snapshot for
  /// keeps.
  Future<String?> captureSnapshotToFile() async {
    final bytes = await captureSnapshotJpeg();
    if (bytes == null) return null;
    try {
      final dir = await getTemporaryDirectory();
      final path =
          '${dir.path}/glass_snap_${DateTime.now().millisecondsSinceEpoch}.jpg';
      await File(path).writeAsBytes(bytes, flush: true);
      return path;
    } catch (e) {
      _lastError = 'Could not write the snapshot to the cache: $e';
      debugPrint('[GlassCameraSource] $_lastError');
      return null;
    }
  }

  @override
  Future<Uint8List?> captureJpeg({
    Duration timeout = const Duration(seconds: 12),
  }) async {
    if (captureMode == GlassCaptureMode.videoSnapshot) {
      return captureSnapshotJpeg();
    }
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
      valueListenable: _playerTick,
      builder: (context, _, __) {
        final controller = _videoController;
        if (controller == null) {
          return _placeholder(
            _lastError ?? 'Starting the glasses live feed…',
            spinning: _lastError == null,
          );
        }
        return ColoredBox(
          color: Colors.black,
          child: Video(
            controller: controller,
            fit: BoxFit.contain,
            controls: NoVideoControls,
          ),
        );
      },
    );
  }

  /// Shown while the RTSP session is coming up, or when it cannot be reached.
  /// Capture works either way, so the copy says so rather than implying the
  /// glasses are unusable.
  Widget _placeholder(String message, {required bool spinning}) {
    return ColoredBox(
      color: Colors.black,
      child: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (spinning)
                const CircularProgressIndicator(color: AppTheme.primaryYellow)
              else
                const Icon(Icons.videocam_off_outlined,
                    color: AppTheme.primaryYellow, size: 40),
              const SizedBox(height: 12),
              const Text(
                'GLASSES CAMERA ACTIVE',
                style: TextStyle(
                  color: AppTheme.primaryYellow,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1.1,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                message,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.white70, fontSize: 13),
              ),
              const SizedBox(height: 4),
              const Text(
                'Capture still works.',
                style: TextStyle(color: Colors.white38, fontSize: 12),
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Future<void> dispose() async {
    await _streamSub?.cancel();
    _streamSub = null;
    _playerTick.dispose();
    // Dispose the player before telling the device to stop, so libmpv is not
    // left reading from a socket that is about to close.
    await _player?.dispose();
    _player = null;
    _playerFuture = null;
    _videoController = null;
    _rtspUrl = null;
    _joinHint = null;
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
      // A mode chosen earlier survives a reconnect; a fresh source would
      // otherwise silently revert to the vendor channel mid-session.
      if (next is GlassCameraSource && _glassCaptureMode != null) {
        next.captureMode = _glassCaptureMode!;
      }

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

  /// Which path the glasses take for a still. Setting it while the phone
  /// camera is active is remembered and applied when the glasses come back.
  GlassCaptureMode get glassCaptureMode =>
      _glassCaptureMode ?? GlassCaptureMode.vendorChannel;

  set glassCaptureMode(GlassCaptureMode mode) {
    _glassCaptureMode = mode;
    final source = _source;
    if (source is GlassCameraSource) source.captureMode = mode;
    notifyListeners();
  }

  GlassCaptureMode? _glassCaptureMode;

  /// Capture policy for Learning/Testing.
  ///
  /// Glasses + live RTSP: vendor-channel stills fail (SDK code 4), so grab
  /// frames from the feed and turn off the native frame-button shutter so
  /// one press is one Dart-handled [captureJpeg]. Phone: yellow button +
  /// phone camera as usual; restore the hardware shutter for other screens.
  Future<void> configureCapturePolicy() async {
    if (usingGlasses) {
      glassCaptureMode = GlassCaptureMode.videoSnapshot;
      await _glass.setHardwareShutter(false);
    } else {
      await _glass.setHardwareShutter(true);
    }
  }

  /// Call when leaving Learning/Testing so other screens keep default shutter.
  Future<void> restoreHardwareShutter() async {
    await _glass.setHardwareShutter(true);
  }

  /// Grabs a frame from the live feed, bypassing whatever [captureJpeg] would
  /// do. Null on the phone camera, which has no video pipeline to read.
  Future<Uint8List?> captureSnapshotJpeg() async {
    final source = _source;
    return source is GlassCameraSource ? source.captureSnapshotJpeg() : null;
  }

  /// Lightweight still for hands-free tip sampling (throttled by the caller).
  ///
  /// Glasses: RTSP video snapshot. Phone: [captureJpeg] / takePicture.
  Future<Uint8List?> captureSampleJpeg() async {
    final source = _source;
    if (source is GlassCameraSource) {
      final snap = await source.captureSnapshotJpeg();
      if (snap != null) return snap;
      return source.captureJpeg();
    }
    return source?.captureJpeg();
  }

  /// [captureSnapshotJpeg] written to the cache; returns the file path.
  Future<String?> captureSnapshotToFile() async {
    final source = _source;
    return source is GlassCameraSource ? source.captureSnapshotToFile() : null;
  }

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
