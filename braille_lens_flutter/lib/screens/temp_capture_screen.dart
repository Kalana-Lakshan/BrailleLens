import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import '../services/camera_source.dart';
import '../services/glass_device_service.dart';
import '../theme/app_theme.dart';
import '../widgets/glass_device_picker.dart';

/// TEMPORARY verification screen for dataset collection.
///
/// It answers one question the Learning and Testing screens cannot: did the
/// capture actually reach the system gallery? Every photo the glasses send is
/// shown as soon as it arrives, with the MediaStore URI the native bridge
/// returned, so a session can be checked without leaving the app and without
/// the ONNX pipeline in the way.
///
/// Run it on its own — the home screen's voice loop would otherwise compete
/// for the microphone:
///
/// ```
/// flutter run -t lib/main_temp_capture.dart
/// ```
///
/// Delete once dataset collection is finished.
class TempCaptureScreen extends StatefulWidget {
  const TempCaptureScreen({super.key});

  @override
  State<TempCaptureScreen> createState() => _TempCaptureScreenState();
}

/// One capture, as it arrived.
class _Shot {
  final DateTime at;
  final int bytes;
  final String? path;
  final String? galleryUri;
  final bool fromFrameButton;
  final GlassCaptureMode mode;

  /// Held only for the few most recent shots — a full-size original is several
  /// megabytes and the target phones have about 2 GB of RAM.
  Uint8List? jpeg;

  _Shot({
    required this.at,
    required this.bytes,
    required this.path,
    required this.galleryUri,
    required this.fromFrameButton,
    required this.mode,
    required this.jpeg,
  });

  bool get savedToGallery => galleryUri != null;

  String get sourceLabel => switch (mode) {
        GlassCaptureMode.vendorChannel => 'Full-res original',
        GlassCaptureMode.videoSnapshot => 'Video snapshot',
      };
}

class _TempCaptureScreenState extends State<TempCaptureScreen> {
  /// How many shots keep their pixels. Older entries stay in the list as rows
  /// but drop their bytes, so a long session cannot exhaust memory.
  static const int _keepDecoded = 6;

  /// A BUTTON_CLICKED closer than this to a photo is taken to be its trigger.
  static const Duration _buttonAttribution = Duration(seconds: 20);

  final GlassDeviceService _glass = GlassDeviceService.instance;
  final CameraSourceController _camera = CameraSourceController();
  final List<StreamSubscription<dynamic>> _subs = [];

  GlassCaptureMode _mode = GlassCaptureMode.vendorChannel;

  final List<_Shot> _shots = [];
  int _selected = 0;

  DateTime? _lastButtonAt;
  Timer? _captureTimeout;
  bool _capturing = false;
  String? _status;
  bool _statusIsError = false;

  @override
  void initState() {
    super.initState();
    _boot();
  }

  Future<void> _boot() async {
    // API 24–28 writes the public copy by hand and needs this; on 29+ the
    // request is a no-op because MediaStore owns the scoped entry.
    await Permission.storage.request();

    _subs.add(_glass.buttonClicks.listen((_) {
      _lastButtonAt = DateTime.now();
      _setStatus('Frame button pressed — capturing…', error: false);
      // In snapshot mode nothing else answers the press: the native shutter is
      // off, so the frame has to be grabbed here.
      if (_mode == GlassCaptureMode.videoSnapshot) unawaited(_snapshot());
    }));
    _subs.add(_glass.photos.listen(_onPhoto));
    _subs.add(_glass.events
        .where((e) => e is GlassPhotoFailed)
        .cast<GlassPhotoFailed>()
        .listen((e) {
      _captureTimeout?.cancel();
      if (mounted) setState(() => _capturing = false);
      _setStatus('Capture failed: ${e.reason}', error: true);
    }));
    _glass.addListener(_onGlassState);
    _camera.addListener(_onGlassState);

    if (!_glass.isConnected) await _connect();
    // Brings up the RTSP feed, which is both the viewfinder and the source a
    // snapshot is grabbed from.
    await _camera.initialize();
    if (mounted) setState(() {});
  }

  /// Switching modes also decides who answers the frame button: in snapshot
  /// mode the native shutter is disabled so one press is one image.
  Future<void> _setMode(GlassCaptureMode mode) async {
    if (mode == _mode) return;
    setState(() => _mode = mode);
    _camera.glassCaptureMode = mode;
    await _glass.setHardwareShutter(mode == GlassCaptureMode.vendorChannel);
    _setStatus(
      mode == GlassCaptureMode.vendorChannel
          ? 'Full-resolution original over Bluetooth — slower, sensor quality'
          : 'Frame grabbed from the live feed — instant, but downscaled',
      error: false,
    );
  }

  void _onGlassState() {
    if (mounted) setState(() {});
  }

  void _onPhoto(GlassPhotoCaptured p) {
    _record(
      jpeg: p.jpeg.isEmpty ? null : p.jpeg,
      bytes: p.jpeg.length,
      path: p.path,
      galleryUri: p.galleryUri,
      mode: GlassCaptureMode.vendorChannel,
    );
  }

  /// Grabs the frame the preview is showing and publishes it to the gallery.
  ///
  /// The bytes never touch the SD card on the glasses, so unlike the vendor
  /// path there is nothing to save unless this does it.
  Future<void> _snapshot() async {
    if (_capturing) return;
    setState(() => _capturing = true);
    _setStatus('Grabbing a frame…', error: false);

    final jpeg = await _camera.captureSnapshotJpeg();
    if (!mounted) return;
    if (jpeg == null) {
      setState(() => _capturing = false);
      _setStatus(
        _camera.source?.lastError ?? 'No frame available from the live feed',
        error: true,
      );
      return;
    }

    final uri = await _glass.saveImageToGallery(jpeg);
    if (!mounted) return;
    _record(
      jpeg: jpeg,
      bytes: jpeg.length,
      path: null,
      galleryUri: uri,
      mode: GlassCaptureMode.videoSnapshot,
    );
  }

  /// Files one capture, whichever path produced it.
  void _record({
    required Uint8List? jpeg,
    required int bytes,
    required String? path,
    required String? galleryUri,
    required GlassCaptureMode mode,
  }) {
    final now = DateTime.now();
    final fromButton = _lastButtonAt != null &&
        now.difference(_lastButtonAt!) < _buttonAttribution;
    _lastButtonAt = null;

    final shot = _Shot(
      at: now,
      bytes: bytes,
      path: path,
      galleryUri: galleryUri,
      fromFrameButton: fromButton,
      mode: mode,
      jpeg: jpeg,
    );

    if (!mounted) return;
    _captureTimeout?.cancel();
    setState(() {
      _shots.insert(0, shot);
      _selected = 0;
      _capturing = false;
      // Free the pixels of anything past the window; the row stays.
      for (var i = _keepDecoded; i < _shots.length; i++) {
        _shots[i].jpeg = null;
      }
      _status = shot.savedToGallery
          ? '${shot.sourceLabel} saved to Pictures/BrailleLens'
          : '${shot.sourceLabel} captured, but NOT saved to the gallery';
      _statusIsError = !shot.savedToGallery;
    });
  }

  void _setStatus(String message, {required bool error}) {
    if (!mounted) return;
    setState(() {
      _status = message;
      _statusIsError = error;
    });
  }

  /// Bonded devices are listed by name; the auto-pick guesses, so a session on
  /// a phone paired with several Bluetooth devices can choose by hand.
  Future<void> _pickDevice() async {
    final picked = await showGlassDevicePicker(context);
    if (picked != null) await _connect(picked.address);
  }

  Future<void> _connect([String? address]) async {
    _setStatus('Connecting…', error: false);
    final ok = await _glass.connect(address: address);
    _setStatus(
      ok ? 'Connected' : 'Connect failed: ${_glass.lastError ?? "no reason"}',
      error: !ok,
    );
  }

  /// Manual shutter. The frame button does not come through here — the native
  /// bridge starts that capture itself and the photo simply arrives.
  Future<void> _capture() async {
    if (_mode == GlassCaptureMode.videoSnapshot) return _snapshot();
    if (_capturing) return;
    setState(() => _capturing = true);
    _setStatus('Capturing…', error: false);

    // The photo arrives on the event stream, not as the return value, so the
    // spinner needs its own way out: a capture that neither lands nor fails
    // would otherwise leave the button disabled for the rest of the session.
    _captureTimeout?.cancel();
    _captureTimeout = Timer(const Duration(seconds: 20), () {
      if (!mounted || !_capturing) return;
      setState(() => _capturing = false);
      _setStatus('No photo arrived within 20 s', error: true);
    });

    final ok = await _glass.takePhoto();
    if (!ok && mounted) {
      _captureTimeout?.cancel();
      setState(() => _capturing = false);
      _setStatus(
        'Glasses refused the capture: ${_glass.lastError ?? "no reason"}',
        error: true,
      );
    }
  }

  @override
  void dispose() {
    _captureTimeout?.cancel();
    for (final s in _subs) {
      s.cancel();
    }
    _glass.removeListener(_onGlassState);
    _camera.removeListener(_onGlassState);
    _camera.dispose();
    // The native shutter is global to the bridge; leaving it off would break
    // the frame button for every other screen in the session.
    unawaited(_glass.setHardwareShutter(true));
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final connected = _glass.isConnected;
    final saved = _shots.where((s) => s.savedToGallery).length;

    return Scaffold(
      backgroundColor: AppTheme.backgroundBlack,
      appBar: AppBar(
        backgroundColor: AppTheme.backgroundBlack,
        title: const Text('Capture → Gallery check'),
        actions: [
          IconButton(
            tooltip: 'Pick glasses',
            icon: const Icon(Icons.bluetooth_searching),
            onPressed: _pickDevice,
          ),
        ],
      ),
      body: Column(
        children: [
          _connectionBar(connected, saved),
          _modeToggle(),
          if (_status != null) _statusBar(),
          if (_camera.usingGlasses) _liveStrip(),
          Expanded(child: _preview()),
          _filmstrip(),
          _captureBar(connected),
        ],
      ),
    );
  }

  Widget _connectionBar(bool connected, int saved) {
    return Container(
      width: double.infinity,
      color: Colors.white10,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      child: Row(
        children: [
          Icon(
            connected ? Icons.link : Icons.link_off,
            color: connected ? AppTheme.successCyan : AppTheme.errorCoral,
            size: 18,
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              connected
                  ? '${_glass.device?.name ?? "AI Glass"} · ${_glass.batteryLevel}%'
                  : 'Not connected',
              style: const TextStyle(color: Colors.white70, fontSize: 13),
            ),
          ),
          Text(
            '$saved/${_shots.length} in gallery',
            style: const TextStyle(
              color: AppTheme.primaryYellow,
              fontSize: 13,
              fontWeight: FontWeight.bold,
            ),
          ),
          if (!connected)
            TextButton(
              onPressed: () => _connect(),
              child: const Text('Connect'),
            ),
        ],
      ),
    );
  }

  /// The two capture paths, side by side. They are not equivalent — the
  /// subtitle says which trade each one makes, because the difference decides
  /// whether dot detection has the resolution it needs.
  Widget _modeToggle() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 10, 12, 4),
      child: SegmentedButton<GlassCaptureMode>(
        segments: const [
          ButtonSegment(
            value: GlassCaptureMode.vendorChannel,
            icon: Icon(Icons.photo_camera, size: 16),
            label: Text('Full-res', style: TextStyle(fontSize: 12)),
          ),
          ButtonSegment(
            value: GlassCaptureMode.videoSnapshot,
            icon: Icon(Icons.bolt, size: 16),
            label: Text('Snapshot', style: TextStyle(fontSize: 12)),
          ),
        ],
        selected: {_mode},
        showSelectedIcon: false,
        onSelectionChanged: (s) => _setMode(s.first),
      ),
    );
  }

  /// The live RTSP feed, kept small — it is here to frame the page, and in
  /// snapshot mode it is literally the image that gets captured.
  Widget _liveStrip() {
    return Container(
      height: 132,
      width: double.infinity,
      margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(border: Border.all(color: Colors.white12)),
      child: _camera.buildPreview(),
    );
  }

  Widget _statusBar() {
    final colour = _statusIsError ? AppTheme.errorCoral : AppTheme.successCyan;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      color: colour.withValues(alpha: 0.12),
      child: Row(
        children: [
          Icon(
            _statusIsError ? Icons.error_outline : Icons.check_circle_outline,
            color: colour,
            size: 18,
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              _status!,
              style: TextStyle(color: colour, fontSize: 13),
            ),
          ),
        ],
      ),
    );
  }

  Widget _preview() {
    if (_shots.isEmpty) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(32),
          child: Text(
            'No captures yet.\n\nPress the frame button on the glasses, '
            'or use Capture below.',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.white38, fontSize: 15),
          ),
        ),
      );
    }

    final shot = _shots[_selected];
    return Column(
      children: [
        Expanded(
          child: Container(
            width: double.infinity,
            color: Colors.black,
            child: shot.jpeg == null
                ? const Center(
                    child: Text(
                      'Pixels released to save memory.\nThe file is still on the phone.',
                      textAlign: TextAlign.center,
                      style: TextStyle(color: Colors.white30, fontSize: 13),
                    ),
                  )
                : Image.memory(
                    shot.jpeg!,
                    fit: BoxFit.contain,
                    gaplessPlayback: true,
                    errorBuilder: (_, __, ___) => const Center(
                      child: Text(
                        'Bytes arrived but could not be decoded as an image',
                        style: TextStyle(color: AppTheme.errorCoral),
                      ),
                    ),
                  ),
          ),
        ),
        _details(shot),
      ],
    );
  }

  Widget _details(_Shot shot) {
    final kb = (shot.bytes / 1024).toStringAsFixed(0);
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
      color: Colors.white10,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                shot.fromFrameButton ? Icons.radio_button_checked : Icons.touch_app,
                size: 16,
                color: AppTheme.primaryYellow,
              ),
              const SizedBox(width: 6),
              Text(
                shot.fromFrameButton ? 'Frame button' : 'On-screen button',
                style: const TextStyle(
                  color: AppTheme.primaryYellow,
                  fontSize: 13,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  shot.sourceLabel,
                  style: const TextStyle(color: Colors.white70, fontSize: 12),
                ),
              ),
              Text(
                '$kb KB · ${_clock(shot.at)}',
                style: const TextStyle(color: Colors.white54, fontSize: 12),
              ),
            ],
          ),
          const SizedBox(height: 8),
          _kv(
            'Gallery',
            shot.galleryUri ?? 'NOT SAVED',
            colour: shot.savedToGallery ? AppTheme.successCyan : AppTheme.errorCoral,
          ),
          const SizedBox(height: 4),
          _kv('Cache', shot.path ?? 'n/a — never written to disk'),
        ],
      ),
    );
  }

  Widget _kv(String label, String value, {Color colour = Colors.white54}) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
          width: 56,
          child: Text(
            label,
            style: const TextStyle(color: Colors.white38, fontSize: 11),
          ),
        ),
        Expanded(
          child: SelectableText(
            value,
            maxLines: 2,
            style: TextStyle(
              color: colour,
              fontSize: 11,
              fontFamily: 'monospace',
            ),
          ),
        ),
      ],
    );
  }

  Widget _filmstrip() {
    if (_shots.length < 2) return const SizedBox.shrink();
    return SizedBox(
      height: 76,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        itemCount: _shots.length,
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (context, i) {
          final shot = _shots[i];
          final isSelected = i == _selected;
          return GestureDetector(
            onTap: () => setState(() => _selected = i),
            child: Container(
              width: 60,
              decoration: BoxDecoration(
                border: Border.all(
                  color: isSelected ? AppTheme.primaryYellow : Colors.white24,
                  width: isSelected ? 2 : 1,
                ),
                color: Colors.black,
              ),
              child: Stack(
                fit: StackFit.expand,
                children: [
                  if (shot.jpeg != null)
                    Image.memory(shot.jpeg!, fit: BoxFit.cover)
                  else
                    const Center(
                      child: Icon(Icons.image_not_supported_outlined,
                          color: Colors.white24, size: 18),
                    ),
                  Align(
                    alignment: Alignment.bottomRight,
                    child: Icon(
                      shot.savedToGallery ? Icons.check_circle : Icons.cancel,
                      size: 14,
                      color: shot.savedToGallery
                          ? AppTheme.successCyan
                          : AppTheme.errorCoral,
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _captureBar(bool connected) {
    return SafeArea(
      top: false,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
        child: SizedBox(
          height: 56,
          child: ElevatedButton.icon(
            onPressed: connected && !_capturing ? _capture : null,
            icon: _capturing
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.camera_alt),
            label: Text(_capturing ? 'Capturing…' : 'Capture'),
          ),
        ),
      ),
    );
  }

  String _clock(DateTime t) =>
      '${t.hour.toString().padLeft(2, '0')}:'
      '${t.minute.toString().padLeft(2, '0')}:'
      '${t.second.toString().padLeft(2, '0')}';
}
