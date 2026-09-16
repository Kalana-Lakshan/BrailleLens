import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Connection lifecycle of the AI Glass peripheral.
///
/// [unavailable] means the native bridge itself is missing (a build without the
/// Realtek SDK, or iOS) — distinct from [disconnected], which means the bridge
/// is present but no glasses are paired or in range.
enum GlassConnectionState {
  idle,
  scanning,
  connecting,
  connected,
  disconnected,
  unavailable,
}

/// A discovered or bonded AI Glass peripheral.
@immutable
class GlassDevice {
  final String address;
  final String name;

  const GlassDevice({required this.address, required this.name});

  factory GlassDevice.fromMap(Map<dynamic, dynamic> m) => GlassDevice(
        address: (m['address'] ?? '') as String,
        name: (m['name'] ?? 'AI Glass') as String,
      );

  @override
  bool operator ==(Object other) =>
      other is GlassDevice && other.address == address;

  @override
  int get hashCode => address.hashCode;
}

/// How the camera feed is reaching us once a live stream is up.
abstract final class GlassLiveChannel {
  /// Phone joins the SoftAP hosted by the glasses, then pulls RTSP.
  static const int wifiAp = 1;

  /// Glasses join a shared LAN or the phone hotspot, then we pull RTSP.
  static const int wifiStation = 2;

  /// H.264 chunks over the Bluetooth vendor channel.
  static const int bluetooth = 3;
}

/// `WifiApInfo.MODE_*` as verified in the Realtek SmartWear AAR.
abstract final class GlassWifiApMode {
  static const int idle = 0;
  static const int station = 1;
  static const int idleWithoutPowerDown = 2;
  static const int stationRtsp = 4;
}

// ── Event model ──────────────────────────────────────────────────────────────
// Native emits one map per event with a `type` discriminator. Modelled as a
// sealed family so a `switch` over it is exhaustive at compile time.

sealed class GlassEvent {
  const GlassEvent();

  factory GlassEvent.from(Map<dynamic, dynamic> map) {
    switch (map['type'] as String?) {
      case 'CONNECTION_STATE':
        return GlassConnectionChanged(
          state: _parseState(map['state'] as String?),
          address: map['address'] as String?,
          name: map['name'] as String?,
        );
      case 'DEVICE_READY':
        return GlassDeviceReady(map['success'] as bool? ?? false);
      case 'SCAN_RESULT':
        return GlassScanResult(GlassDevice.fromMap(map));
      case 'BUTTON_CLICKED':
        return const GlassButtonClicked();
      case 'MIC_DATA':
        return GlassMicData(map['data'] as Uint8List? ?? Uint8List(0));
      case 'MIC_STATE':
        return GlassMicState(map['streaming'] as bool? ?? false);
      case 'BATTERY':
        return GlassBattery(
          level: map['level'] as int? ?? -1,
          charging: map['charging'] as bool? ?? false,
        );
      case 'PHOTO_CAPTURED':
        return GlassPhotoCaptured(map['data'] as Uint8List? ?? Uint8List(0));
      case 'PHOTO_FAILED':
        return GlassPhotoFailed(map['reason'] as String? ?? 'unknown');
      case 'LIVE_STREAM':
        return GlassLiveStream(
          channel: map['channel'] as int? ?? GlassLiveChannel.wifiAp,
          ssid: map['ssid'] as String?,
          password: map['password'] as String?,
          rtspUrl: map['rtspUrl'] as String? ?? '',
        );
      case 'VIDEO_FRAME':
        return GlassVideoFrame(map['data'] as Uint8List? ?? Uint8List(0));
      case 'WIFI_AP_STATE':
        return GlassWifiApState(
          ssid: map['ssid'] as String?,
          password: map['password'] as String?,
          mode: map['mode'] as int? ?? GlassWifiApMode.idle,
        );
      case 'ERROR':
        return GlassError(
          map['message'] as String? ?? 'Unknown glasses error',
          code: map['code'] as String?,
        );
      default:
        return GlassUnknownEvent(map);
    }
  }

  static GlassConnectionState _parseState(String? raw) => switch (raw) {
        'scanning' => GlassConnectionState.scanning,
        'connecting' => GlassConnectionState.connecting,
        'connected' => GlassConnectionState.connected,
        'disconnected' => GlassConnectionState.disconnected,
        'unavailable' => GlassConnectionState.unavailable,
        _ => GlassConnectionState.idle,
      };
}

class GlassConnectionChanged extends GlassEvent {
  final GlassConnectionState state;
  final String? address;
  final String? name;
  const GlassConnectionChanged({required this.state, this.address, this.name});
}

class GlassDeviceReady extends GlassEvent {
  final bool success;
  const GlassDeviceReady(this.success);
}

class GlassScanResult extends GlassEvent {
  final GlassDevice device;
  const GlassScanResult(this.device);
}

/// The physical frame button was pressed.
class GlassButtonClicked extends GlassEvent {
  const GlassButtonClicked();
}

/// 16 kHz mono 16-bit PCM from the glasses microphone (~32 KB/s).
class GlassMicData extends GlassEvent {
  final Uint8List pcm;
  const GlassMicData(this.pcm);
}

class GlassMicState extends GlassEvent {
  final bool streaming;
  const GlassMicState(this.streaming);
}

class GlassBattery extends GlassEvent {
  final int level;
  final bool charging;
  const GlassBattery({required this.level, required this.charging});
}

/// A full-resolution JPEG pulled off the glasses after a capture.
class GlassPhotoCaptured extends GlassEvent {
  final Uint8List jpeg;
  const GlassPhotoCaptured(this.jpeg);
}

class GlassPhotoFailed extends GlassEvent {
  final String reason;
  const GlassPhotoFailed(this.reason);
}

class GlassLiveStream extends GlassEvent {
  final int channel;
  final String? ssid;
  final String? password;
  final String rtspUrl;
  const GlassLiveStream({
    required this.channel,
    this.ssid,
    this.password,
    required this.rtspUrl,
  });
}

/// One decoded preview frame (JPEG) from the live feed.
class GlassVideoFrame extends GlassEvent {
  final Uint8List jpeg;
  const GlassVideoFrame(this.jpeg);
}

class GlassWifiApState extends GlassEvent {
  final String? ssid;
  final String? password;
  final int mode;
  const GlassWifiApState({this.ssid, this.password, required this.mode});
}

class GlassError extends GlassEvent {
  final String message;
  final String? code;
  const GlassError(this.message, {this.code});
}

class GlassUnknownEvent extends GlassEvent {
  final Map<dynamic, dynamic> raw;
  const GlassUnknownEvent(this.raw);
}

// ── Service ──────────────────────────────────────────────────────────────────

/// Owns the Bluetooth link to the AI Glass and fans its events out to the app.
///
/// Transport note: the glasses are **not** a BLE GATT peripheral. They speak
/// Realtek Audio Connect over Bluetooth Classic (vendor/SPP), with video
/// carried separately over Wi-Fi RTSP. So this is a thin Dart face over a
/// native Android bridge wrapping `SmartWearModelClient`, rather than a
/// `flutter_blue_plus` client — there is no button characteristic to subscribe
/// to; the frame button surfaces as `onDeviceTriggeredTakePhoto()`.
///
/// A [ChangeNotifier] so screens can rebuild off connection and battery state
/// with `ListenableBuilder`, matching the plain-service style already used in
/// `lib/services/` (no Bloc/Riverpod dependency is introduced).
///
/// Every native call fails soft: a [MissingPluginException] leaves the service
/// [GlassConnectionState.unavailable] so the app falls back to phone hardware
/// instead of crashing.
class GlassDeviceService extends ChangeNotifier {
  GlassDeviceService._();

  static final GlassDeviceService instance = GlassDeviceService._();

  static const MethodChannel _control =
      MethodChannel('braille_lens/glass_control');
  static const EventChannel _eventChannel =
      EventChannel('braille_lens/glass_events');

  /// Single broadcast fan-out: however many screens listen, the native side
  /// sees exactly one onListen/onCancel pair.
  final StreamController<GlassEvent> _events =
      StreamController<GlassEvent>.broadcast();
  StreamSubscription<dynamic>? _nativeSub;

  GlassConnectionState _state = GlassConnectionState.idle;
  GlassDevice? _device;
  int _batteryLevel = -1;
  bool _charging = false;
  bool _micStreaming = false;
  bool _bridgeChecked = false;
  String? _lastError;
  String? _lastErrorCode;

  // Auto-reconnect
  Timer? _reconnectTimer;
  int _reconnectAttempt = 0;
  bool _autoReconnect = true;
  static const int _maxReconnectDelaySeconds = 30;

  GlassConnectionState get state => _state;
  GlassDevice? get device => _device;
  bool get isConnected => _state == GlassConnectionState.connected;
  bool get isAvailable => _state != GlassConnectionState.unavailable;
  int get batteryLevel => _batteryLevel;
  bool get isCharging => _charging;
  bool get isMicStreaming => _micStreaming;
  String? get lastError => _lastError;
  String? get lastErrorCode => _lastErrorCode;

  /// Every event from the glasses, in arrival order.
  Stream<GlassEvent> get events {
    _ensureNativeSubscription();
    return _events.stream;
  }

  Stream<T> _only<T extends GlassEvent>() =>
      events.where((e) => e is T).cast<T>();

  /// Frame-button presses. Wire this to the same handler as the on-screen
  /// capture button so both run one path.
  Stream<GlassButtonClicked> get buttonClicks => _only<GlassButtonClicked>();
  Stream<GlassMicData> get micData => _only<GlassMicData>();
  Stream<GlassVideoFrame> get videoFrames => _only<GlassVideoFrame>();
  Stream<GlassPhotoCaptured> get photos => _only<GlassPhotoCaptured>();
  Stream<GlassScanResult> get scanResults => _only<GlassScanResult>();
  Stream<GlassError> get errors => _only<GlassError>();

  void _ensureNativeSubscription() {
    if (_nativeSub != null) return;
    try {
      _nativeSub = _eventChannel.receiveBroadcastStream().listen(
        (dynamic raw) {
          if (raw is Map) _dispatch(GlassEvent.from(raw));
        },
        onError: (Object e) {
          _lastError = e.toString();
          _emit(GlassError(_lastError!, code: 'STREAM'));
        },
        cancelOnError: false,
      );
    } on MissingPluginException {
      _setState(GlassConnectionState.unavailable);
    }
  }

  void _emit(GlassEvent e) {
    if (!_events.isClosed) _events.add(e);
  }

  /// Folds an event into service state, then republishes it.
  void _dispatch(GlassEvent e) {
    switch (e) {
      case GlassConnectionChanged(:final state, :final address, :final name):
        if (address != null && address.isNotEmpty) {
          _device = GlassDevice(address: address, name: name ?? 'AI Glass');
        }
        if (state == GlassConnectionState.connected) {
          _reconnectAttempt = 0;
          _reconnectTimer?.cancel();
          _reconnectTimer = null;
        } else if (state == GlassConnectionState.disconnected) {
          _micStreaming = false;
          _scheduleReconnect();
        }
        _setState(state);
      case GlassBattery(:final level, :final charging):
        _batteryLevel = level;
        _charging = charging;
        notifyListeners();
      case GlassMicState(:final streaming):
        _micStreaming = streaming;
        notifyListeners();
      case GlassError(:final message, :final code):
        _lastError = message;
        _lastErrorCode = code;
        notifyListeners();
      default:
        break;
    }
    _emit(e);
  }

  void _setState(GlassConnectionState s) {
    if (_state == s) return;
    _state = s;
    notifyListeners();
  }

  // ── Commands ───────────────────────────────────────────────────────────────

  /// Brings up the native SDK. Safe to call more than once.
  Future<bool> initialize() async {
    if (_bridgeChecked && _state == GlassConnectionState.unavailable) {
      return false;
    }
    _bridgeChecked = true;
    _ensureNativeSubscription();
    final ok = await _invoke<bool>('initialize') ?? false;
    if (!ok && _state != GlassConnectionState.unavailable) {
      _setState(GlassConnectionState.disconnected);
    }
    return ok;
  }

  /// Discovers bonded or in-range glasses. Results also arrive on
  /// [scanResults].
  Future<List<GlassDevice>> scanForGlasses({
    Duration timeout = const Duration(seconds: 8),
  }) async {
    if (!await initialize()) return const [];
    _setState(GlassConnectionState.scanning);
    final raw = await _invoke<List<dynamic>>(
      'scan',
      {'timeoutMs': timeout.inMilliseconds},
    );
    final found = (raw ?? const [])
        .whereType<Map<dynamic, dynamic>>()
        .map(GlassDevice.fromMap)
        .toList(growable: false);
    if (_state == GlassConnectionState.scanning) {
      _setState(found.isEmpty
          ? GlassConnectionState.disconnected
          : GlassConnectionState.idle);
    }
    return found;
  }

  /// Connects and waits for the SDK device-ready handshake.
  ///
  /// [address] may be omitted to reconnect the last bonded pair.
  Future<bool> connect({
    String? address,
    Duration timeout = const Duration(seconds: 12),
  }) async {
    if (!await initialize()) return false;

    _autoReconnect = true;
    _setState(GlassConnectionState.connecting);

    // Subscribe *before* issuing the command: `connect` returns as soon as the
    // request is queued, while DEVICE_READY arrives asynchronously — and a
    // broadcast stream drops events that have no listener yet.
    final completer = Completer<bool>();
    final sub = _only<GlassDeviceReady>().listen((e) {
      if (!completer.isCompleted) completer.complete(e.success);
    });
    final timer = Timer(timeout, () {
      if (!completer.isCompleted) completer.complete(false);
    });

    try {
      final queued =
          await _invoke<bool>('connect', {'address': address}) ?? false;
      if (!queued) return false;
      final ready = await completer.future;
      if (ready) {
        _setState(GlassConnectionState.connected);
        unawaited(refreshBattery());
      } else {
        _setState(GlassConnectionState.disconnected);
        _scheduleReconnect();
      }
      return ready;
    } finally {
      timer.cancel();
      await sub.cancel();
    }
  }

  /// Disconnects and suppresses auto-reconnect until the next [connect].
  Future<void> disconnect() async {
    _autoReconnect = false;
    _reconnectTimer?.cancel();
    _reconnectTimer = null;
    _reconnectAttempt = 0;
    await _invoke<void>('disconnect');
    _micStreaming = false;
    _setState(GlassConnectionState.disconnected);
  }

  /// Exponential backoff, capped at [_maxReconnectDelaySeconds].
  void _scheduleReconnect() {
    if (!_autoReconnect || _reconnectTimer != null) return;
    final address = _device?.address;
    final delay = Duration(
      seconds: (1 << _reconnectAttempt).clamp(1, _maxReconnectDelaySeconds),
    );
    _reconnectAttempt = (_reconnectAttempt + 1).clamp(0, 5);
    debugPrint('[GlassDevice] reconnect in ${delay.inSeconds}s');
    _reconnectTimer = Timer(delay, () {
      _reconnectTimer = null;
      if (_autoReconnect && !isConnected) connect(address: address);
    });
  }

  /// Asks the glasses to take a full-resolution photo. The JPEG arrives on
  /// [photos] once transferred over the vendor channel.
  Future<bool> takePhoto() async => await _invoke<bool>('takePhoto') ?? false;

  /// Starts the glasses microphone PCM stream ([micData]).
  Future<bool> startMicrophone() async =>
      await _invoke<bool>('startMic') ?? false;

  Future<bool> stopMicrophone() async =>
      await _invoke<bool>('stopMic') ?? false;

  /// Brings up the RTSP live feed. Details arrive as [GlassLiveStream].
  Future<bool> startLiveStream({int channel = GlassLiveChannel.wifiAp}) async =>
      await _invoke<bool>('startLiveStream', {'channel': channel}) ?? false;

  Future<bool> stopLiveStream() async =>
      await _invoke<bool>('stopLiveStream') ?? false;

  Future<void> refreshBattery() => _invoke<void>('getBattery');

  Future<T?> _invoke<T>(String method, [Map<String, dynamic>? args]) async {
    try {
      return await _control.invokeMethod<T>(method, args);
    } on MissingPluginException {
      _lastError = 'Glasses bridge not available in this build';
      _lastErrorCode = 'NO_BRIDGE';
      _setState(GlassConnectionState.unavailable);
      return null;
    } on PlatformException catch (e) {
      _lastError = e.message ?? e.code;
      _lastErrorCode = e.code;
      debugPrint('[GlassDevice] $method failed: ${e.code} ${e.message}');
      _emit(GlassError(_lastError!, code: e.code));
      return null;
    }
  }

  @override
  void dispose() {
    _reconnectTimer?.cancel();
    _nativeSub?.cancel();
    _events.close();
    super.dispose();
  }
}
