import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:media_kit/media_kit.dart';
import 'package:media_kit_video/media_kit_video.dart';
import 'package:permission_handler/permission_handler.dart';

import '../services/audio_routing_service.dart';
import '../services/camera_source.dart';
import '../services/glass_device_service.dart';
import '../theme/app_theme.dart';
import '../widgets/glass_device_picker.dart';

/// TEMPORARY bring-up screen: pings each Realtek SDK primitive the Kotlin
/// bridge exposes, with no ONNX or Braille logic in the way. Delete once the
/// Learning/Testing state machines are built on top of a verified bridge.
///
/// Run it on its own so the home screen's voice loop does not fight it for the
/// mic and TTS: `flutter run -t lib/main_hardware_debug.dart`.
class HardwareDebugScreen extends StatefulWidget {
  const HardwareDebugScreen({super.key});

  @override
  State<HardwareDebugScreen> createState() => _HardwareDebugScreenState();
}

class _HardwareDebugScreenState extends State<HardwareDebugScreen> {
  /// Where the glasses SoftAP serves RTSP in the reference app
  /// (`WifiViewModel.ipAddress`). Replaced if LIVE_STREAM reports another IP.
  static const String _defaultRtspUrl = 'rtsp://192.168.43.1:554';

  /// "Hello. This is a glasses sound test."
  static const String _sinhalaPhrase = 'ආයුබෝවන්. මෙය කණ්ණාඩි ශබ්ද පරීක්ෂණයකි.';

  static const int _maxLogLines = 40;

  final GlassDeviceService _glass = GlassDeviceService.instance;
  final AudioRoutingService _audio = AudioRoutingService.instance;
  final FlutterTts _tts = FlutterTts();

  final List<StreamSubscription<dynamic>> _subs = [];
  final List<String> _log = [];

  // 1. TTS
  bool? _sinhalaVoiceAvailable;
  String _ttsStatus = 'Idle';

  // 2. Mic
  StreamSubscription<GlassMicData>? _micSub;
  int _micChunks = 0;
  int _micBytes = 0;
  int _lastChunkLength = 0;

  // 3. RTSP
  Player? _player;
  Future<Player>? _playerFuture;
  VideoController? _videoController;
  bool _liveRequested = false;
  String _rtspUrl = _defaultRtspUrl;
  String _rtspStatus = 'Idle';
  String? _apSsid;
  String? _apPassword;
  int? _videoWidth;
  int? _videoHeight;

  // 4. Frame button
  int _buttonPresses = 0;
  String _buttonStatus = 'Waiting for a frame-button press…';
  String? _lastPhotoPath;
  String? _correctedPath;

  @override
  void initState() {
    super.initState();
    _glass.addListener(_onServiceChanged);
    _audio.addListener(_onServiceChanged);
    _subs.add(_glass.events.listen(_onGlassEvent));
    _boot();
  }

  Future<void> _boot() async {
    await Permission.microphone.request();
    await Permission.bluetoothScan.request();
    await Permission.bluetoothConnect.request();
    // Location is declared only up to Android 9 (the old Wi-Fi join reads the
    // SSID there); on newer versions this returns without a prompt.
    // Nearby-Wi-Fi covers the Station-mode hotspot on Android 13+.
    await Permission.locationWhenInUse.request();
    await Permission.nearbyWifiDevices.request();

    final sdkOk = await _glass.initialize();
    _addLog('initialize → $sdkOk');
    await _audio.refresh();

    try {
      await _tts.awaitSpeakCompletion(true);
      await _tts.setSpeechRate(0.45);
      // A missing Sinhala voice speaks nothing, which looks exactly like a
      // routing failure — so check up front and say which it is.
      _sinhalaVoiceAvailable = await _tts.isLanguageAvailable('si-LK') == true;
    } catch (e) {
      _addLog('TTS setup error: $e');
    }
    if (mounted) setState(() {});
  }

  void _onServiceChanged() {
    if (mounted) setState(() {});
  }

  void _addLog(String line) {
    final t = TimeOfDay.now();
    final s = DateTime.now().second.toString().padLeft(2, '0');
    final stamp =
        '${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}:$s';
    debugPrint('[HWDebug] $line');
    _log.insert(0, '$stamp  $line');
    if (_log.length > _maxLogLines) _log.removeLast();
    if (mounted) setState(() {});
  }

  // ── Event fan-in ───────────────────────────────────────────────────────────

  void _onGlassEvent(GlassEvent e) {
    switch (e) {
      // High-rate streams: counted in their own panels, not logged.
      case GlassMicData():
      case GlassVideoFrame():
        return;
      case GlassButtonClicked():
        _onFrameButton();
      case GlassPhotoCaptured(:final jpeg, :final path):
        final kb = (jpeg.length / 1024).toStringAsFixed(1);
        _lastPhotoPath = path;
        setState(() => _buttonStatus =
            'Photo received after press #$_buttonPresses\n'
            '${path ?? '(bridge sent no path)'}\n$kb KB');
      case GlassPhotoFailed(:final reason):
        setState(() => _buttonStatus =
            'Photo failed after press #$_buttonPresses: $reason');
      case GlassLiveStream():
        _onLiveStream(e);
      case GlassWifiApState(:final ssid, :final password):
        setState(() {
          _apSsid = ssid ?? _apSsid;
          _apPassword = password ?? _apPassword;
        });
      default:
        break;
    }
    _addLog(_describe(e));
  }

  String _describe(GlassEvent e) => switch (e) {
        GlassConnectionChanged(:final state, :final name) =>
          'CONNECTION_STATE ${state.name} ${name ?? ''}',
        GlassDeviceReady(:final success) => 'DEVICE_READY success=$success',
        GlassScanResult(:final device) =>
          'SCAN_RESULT ${device.name} ${device.address}',
        GlassButtonClicked() => 'BUTTON_CLICKED (onDeviceTriggeredTakePhoto)',
        GlassMicState(:final streaming) => 'MIC_STATE streaming=$streaming',
        GlassBattery(:final level) => 'BATTERY $level%',
        GlassDeviceAction(:final action) => 'DEVICE_ACTION $action',
        GlassWifiConnection(:final connected) =>
          'WIFI_CONNECTION connected=$connected',
        GlassWifiStaState(:final active, :final ipAddress) =>
          'WIFI_STA_STATE active=$active ip=$ipAddress',
        GlassSoftApState(:final started, :final ssid) =>
          'SOFT_AP_STATE started=$started ssid=$ssid',
        GlassPhotoCaptured(:final jpeg, :final path) =>
          'PHOTO_CAPTURED ${jpeg.length} B $path',
        GlassPhotoFailed(:final reason) => 'PHOTO_FAILED $reason',
        GlassLiveStream(
          :final channel,
          :final ssid,
          :final rtspUrl,
          :final wifiConnected,
        ) =>
          'LIVE_STREAM ch=$channel ssid=$ssid url=$rtspUrl '
              'phoneOnWifi=$wifiConnected',
        GlassWifiApState(:final ssid, :final mode) =>
          'WIFI_AP_STATE ssid=$ssid mode=$mode',
        GlassError(:final message, :final code) => 'ERROR [$code] $message',
        GlassUnknownEvent(:final raw) => 'UNKNOWN $raw',
        GlassMicData() || GlassVideoFrame() => e.runtimeType.toString(),
      };

  // ── Connection ─────────────────────────────────────────────────────────────

  /// [address] null lets the bridge auto-pick (glasses-like name, else the
  /// active Bluetooth headset).
  Future<void> _connect([String? address]) async {
    _addLog('connect(${address ?? 'auto'}) …');
    final ok = await _glass.connect(address: address);
    _addLog('connect → $ok${ok ? '' : ' (${_glass.lastError ?? 'no reason'})'}');
    await _audio.refresh();
  }

  /// Lists every paired device so the right one can be chosen by hand.
  /// Same picker the home screen uses, so both paths connect identically.
  Future<void> _pickDevice() async {
    final picked = await showGlassDevicePicker(context);
    _addLog(picked == null
        ? 'device picker: nothing connected'
        : 'device picker: connected ${picked.name} (${picked.address})');
    await _audio.refresh();
    if (mounted) setState(() {});
  }

  Future<void> _disconnect() async {
    await _stopMic();
    await _stopRtsp();
    await _glass.disconnect();
    _addLog('disconnect()');
  }

  // ── 1. TTS routing ─────────────────────────────────────────────────────────

  /// [viaSco] true forces the HFP/SCO link up first (the path prompts take
  /// while the glasses mic is live); false drops SCO so playback rides A2DP.
  Future<void> _testTts({required bool viaSco}) async {
    final routeName = viaSco ? 'SCO' : 'A2DP';
    setState(() => _ttsStatus = 'Setting up $routeName route…');

    if (viaSco) {
      final ok = await _audio.enableGlassesRoute();
      _addLog('enableGlassesRoute → $ok'
          '${ok ? '' : ' (${_audio.lastError ?? 'no reason'})'}');
    } else {
      await _audio.enablePhoneRoute();
      _addLog('enablePhoneRoute (SCO off, media follows A2DP)');
    }

    try {
      final langResult = await _tts.setLanguage('si-LK');
      setState(() => _ttsStatus = 'Speaking over $routeName…');
      _addLog('TTS setLanguage(si-LK) → $langResult, speaking');
      await _tts.speak(_sinhalaPhrase);
      setState(() => _ttsStatus =
          'Finished ($routeName). Did the glasses play it?  '
          'headset=${_audio.isHeadsetConnected} sco=${_audio.isScoActive}');
    } catch (e) {
      setState(() => _ttsStatus = 'TTS error: $e');
      _addLog('TTS error: $e');
    }
  }

  // ── 2. Mic stream ──────────────────────────────────────────────────────────

  Future<void> _toggleMic() async {
    if (_micSub != null) {
      await _stopMic();
      return;
    }

    setState(() {
      _micChunks = 0;
      _micBytes = 0;
      _lastChunkLength = 0;
    });

    // Subscribe before starting so the first chunk is not dropped by the
    // broadcast stream.
    _micSub = _glass.micData.listen((e) {
      _micChunks++;
      _micBytes += e.pcm.length;
      _lastChunkLength = e.pcm.length;
      debugPrint('[HWDebug] MIC_DATA #$_micChunks: ${e.pcm.length} bytes');
      if (mounted) setState(() {});
    });

    final ok = await _glass.startMicrophone();
    _addLog('startMic (startUserVoiceInput) → $ok');
    if (!ok) {
      await _micSub?.cancel();
      _micSub = null;
    }
    if (mounted) setState(() {});
  }

  Future<void> _stopMic() async {
    if (_micSub == null) return;
    final ok = await _glass.stopMicrophone();
    await _micSub?.cancel();
    _micSub = null;
    _addLog('stopMic → $ok  ($_micChunks chunks, $_micBytes bytes total)');
  }

  // ── 3. RTSP ────────────────────────────────────────────────────────────────

  Future<void> _startRtsp() async {
    setState(() => _rtspStatus = 'Asking glasses to start the Wi-Fi AP stream…');
    final ok = await _glass.startLiveStream(channel: GlassLiveChannel.wifiAp);
    _liveRequested = ok;
    _addLog('startLiveStream(WIFI_AP) → $ok');
    setState(() => _rtspStatus = ok
        ? 'Command sent. Waiting for LIVE_STREAM…'
        : 'startLiveStream failed: ${_glass.lastError ?? 'no reason'}');
  }

  void _onLiveStream(GlassLiveStream e) {
    setState(() {
      _apSsid = e.ssid ?? _apSsid;
      _apPassword = e.password ?? _apPassword;
      if (e.rtspUrl.isNotEmpty) _rtspUrl = e.rtspUrl;
      _rtspStatus = e.wifiConnected
          ? 'Stream up and this phone is on the glasses Wi-Fi.'
          : 'Stream up, but the automatic Wi-Fi join failed. Join the network '
              'below in Android settings, then tap "Reopen player".';
    });
    _openPlayer();
  }

  Future<void> _openPlayer() async {
    // Same audio-off player the real camera source uses, so this screen cannot
    // trigger the SCO "call ended" drop either. The future is shared so a
    // LIVE_STREAM event and a button tap cannot build two players.
    final created = await (_playerFuture ??= createGlassPreviewPlayer());
    if (!mounted) return;
    final player = _player ?? _attachPlayer(created);
    _addLog('player.open($_rtspUrl)');
    try {
      await player.open(Media(_rtspUrl), play: true);
    } catch (e) {
      _addLog('player.open failed: $e');
    }
    if (mounted) setState(() {});
  }

  /// First-time wiring for a freshly created player.
  Player _attachPlayer(Player player) {
    _player = player;
    _videoController = VideoController(player);
    _subs
      ..add(player.stream.error.listen((err) {
        _addLog('player error: $err');
        if (mounted) setState(() => _rtspStatus = 'Player error: $err');
      }))
      ..add(player.stream.width.listen((w) {
        if (mounted) setState(() => _videoWidth = w);
      }))
      ..add(player.stream.height.listen((h) {
        if (mounted) setState(() => _videoHeight = h);
      }));
    return player;
  }

  Future<void> _stopRtsp() async {
    // Close the player before the device tears the socket down.
    await _player?.dispose();
    _player = null;
    _playerFuture = null;
    _videoController = null;
    _videoWidth = null;
    _videoHeight = null;
    if (_liveRequested) {
      final ok = await _glass.stopLiveStream();
      _addLog('stopLiveStream → $ok');
      _liveRequested = false;
    }
    if (mounted) setState(() => _rtspStatus = 'Stopped');
  }

  // ── 4. Frame button ────────────────────────────────────────────────────────

  /// The bridge surfaces the press only; it does not capture on its own. Ask
  /// for the full-res original here, as the Learning screen does, so the
  /// PHOTO_CAPTURED path shows up in the status panel.
  Future<void> _onFrameButton() async {
    final n = ++_buttonPresses;
    setState(() =>
        _buttonStatus = 'Press #$n intercepted — requesting full-res photo…');
    final ok = await _glass.takePhoto();
    _addLog('takePhoto (TYPE_TAKE_ORIGIN_IMAGE) → $ok');
    if (!ok && mounted) {
      setState(() => _buttonStatus =
          'Press #$n intercepted, but takePhoto was refused: '
          '${_glass.lastError ?? 'no reason'}');
    }
  }

  /// §6 lens-distortion correction on the last received photo.
  Future<void> _correctLastPhoto() async {
    final input = _lastPhotoPath;
    if (input == null) return;
    _addLog('correctImage($input) …');
    final out = await _glass.correctImage(input);
    _addLog('correctImage → ${out ?? 'failed (${_glass.lastError})'}');
    if (mounted) setState(() => _correctedPath = out);
  }

  @override
  void dispose() {
    _glass.removeListener(_onServiceChanged);
    _audio.removeListener(_onServiceChanged);
    for (final s in _subs) {
      s.cancel();
    }
    if (_micSub != null) {
      _micSub!.cancel();
      _glass.stopMicrophone();
    }
    _player?.dispose();
    if (_liveRequested) _glass.stopLiveStream();
    _tts.stop();
    // Leave nothing hot: SCO would keep the glasses mic open.
    _audio.enablePhoneRoute();
    super.dispose();
  }

  // ── UI ─────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final connected = _glass.isConnected;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Hardware Debug (temporary)'),
        backgroundColor: AppTheme.backgroundBlack,
      ),
      body: ListView(
        padding: const EdgeInsets.all(12),
        children: [
          _Panel(
            title: 'Connection',
            lines: [
              'Glasses: ${_glass.state.name}'
                  '${_glass.device != null ? ' — ${_glass.device!.name} (${_glass.device!.address})' : ''}',
              'Battery: ${_glass.batteryLevel >= 0 ? '${_glass.batteryLevel}%' : '—'}',
              'Audio: headset=${_audio.isHeadsetConnected} '
                  'sco=${_audio.isScoActive} route=${_audio.route.name}',
              if (_glass.lastError != null)
                'Last error: [${_glass.lastErrorCode}] ${_glass.lastError}',
            ],
            actions: [
              _Btn('Connect', connected ? null : () => _connect()),
              _Btn('Pick device', connected ? null : _pickDevice),
              _Btn('Disconnect', connected ? _disconnect : null),
            ],
          ),
          _Panel(
            title: '1. Test TTS routing',
            lines: [
              'Phrase: $_sinhalaPhrase',
              'si-LK voice installed: ${_sinhalaVoiceAvailable ?? '…'}',
              _ttsStatus,
            ],
            actions: [
              _Btn('Speak via A2DP', () => _testTts(viaSco: false)),
              _Btn('Speak via SCO', () => _testTts(viaSco: true)),
            ],
          ),
          _Panel(
            title: '2. Mic stream (startUserVoiceInput)',
            lines: [
              'Service reports streaming: ${_glass.isMicStreaming}',
              'Chunks: $_micChunks   last: $_lastChunkLength B   '
                  'total: $_micBytes B',
              'Per-chunk lengths print to the debug console.',
            ],
            actions: [
              _Btn(_micSub == null ? 'Start Mic Stream' : 'Stop Mic Stream',
                  connected || _micSub != null ? _toggleMic : null),
            ],
          ),
          _Panel(
            title: '3. RTSP stream (Wi-Fi AP)',
            lines: [
              _rtspStatus,
              'URL: $_rtspUrl',
              'Glasses AP: ${_apSsid ?? '—'}'
                  '${_apPassword != null ? '  pw: $_apPassword' : ''}',
              'Phone on glasses Wi-Fi: ${_glass.isPhoneOnGlassesWifi}',
              if (_videoWidth != null && _videoHeight != null)
                'Decoding: $_videoWidth×$_videoHeight',
            ],
            actions: [
              _Btn('Start RTSP Stream', connected ? _startRtsp : null),
              _Btn('Reopen player', _openPlayer),
              _Btn('Stop', _player != null || _liveRequested ? _stopRtsp : null),
            ],
            child: _videoController == null
                ? null
                : AspectRatio(
                    aspectRatio: 16 / 9,
                    child: Video(
                      controller: _videoController!,
                      controls: NoVideoControls,
                    ),
                  ),
          ),
          _Panel(
            title: '4. Button intercept status',
            lines: [
              'Presses: $_buttonPresses',
              _buttonStatus,
              if (_correctedPath != null) 'Corrected: $_correctedPath',
            ],
            actions: [
              _Btn('Correct distortion',
                  _lastPhotoPath != null ? _correctLastPhoto : null),
            ],
          ),
          _Panel(
            title: 'Event log',
            lines: _log.isEmpty ? const ['(no events yet)'] : _log,
            monospace: true,
          ),
        ],
      ),
    );
  }
}

class _Btn {
  final String label;
  final VoidCallback? onPressed;
  const _Btn(this.label, this.onPressed);
}

class _Panel extends StatelessWidget {
  final String title;
  final List<String> lines;
  final List<_Btn> actions;
  final Widget? child;
  final bool monospace;

  const _Panel({
    required this.title,
    required this.lines,
    this.actions = const [],
    this.child,
    this.monospace = false,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      color: const Color(0xFF1E1E1E),
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: const TextStyle(
                color: AppTheme.primaryYellow,
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            for (final line in lines)
              Padding(
                padding: const EdgeInsets.only(bottom: 2),
                child: SelectableText(
                  line,
                  style: TextStyle(
                    color: Colors.white70,
                    fontSize: monospace ? 11 : 13,
                    fontFamily: monospace ? 'monospace' : null,
                  ),
                ),
              ),
            if (child != null) ...[const SizedBox(height: 8), child!],
            if (actions.isNotEmpty) ...[
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  for (final a in actions)
                    ElevatedButton(
                      onPressed: a.onPressed,
                      child: Text(a.label),
                    ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}
