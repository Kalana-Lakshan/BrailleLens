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

  /// The paired device's name matches a known glasses naming pattern.
  final bool likelyGlasses;

  /// Currently connected to the phone as a Bluetooth headset (A2DP/HFP).
  final bool audioConnected;

  const GlassDevice({
    required this.address,
    required this.name,
    this.likelyGlasses = false,
    this.audioConnected = false,
  });

  factory GlassDevice.fromMap(Map<dynamic, dynamic> m) => GlassDevice(
        address: (m['address'] ?? '') as String,
        name: (m['name'] ?? 'AI Glass') as String,
        likelyGlasses: m['likelyGlasses'] as bool? ?? false,
        audioConnected: m['audioConnected'] as bool? ?? false,
      );

  @override
  bool operator ==(Object other) =>
      other is GlassDevice && other.address == address;

  @override
  int get hashCode => address.hashCode;
}

/// How the camera feed is reaching us once a live stream is up.
abstract final class GlassLiveChannel {
  /// Glasses host a SoftAP; the bridge joins the phone to it and the feed is
  /// at the fixed `rtsp://192.168.43.1:554` (SmartWear guide §5.1).
  static const int wifiAp = 1;

  /// The phone hosts a hotspot and the glasses join it; the feed is at the IP
  /// the glasses report (§5.2). Refused on Android 10–12L, where the hotspot
  /// would need location access the app does not request.
  static const int wifiStation = 2;

  /// H.264 over the Bluetooth vendor channel. The bridge rejects it: the data
  /// needs the SDK's own player, which nothing on the Dart side drives.
  static const int bluetooth = 3;
}

/// What `takePhoto` asks the glasses for (§4.3 `TakePhotoType`).
enum GlassPhotoType {
  /// Full-resolution original — what the Braille pipeline needs.
  origin,

  /// Original plus a thumbnail; the bridge still returns the original.
  originAndThumbnail,

  /// Small preview only; too small for dot detection.
  thumbnail,
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
          secondaryLevel: map['secondaryLevel'] as int? ?? -1,
        );
      case 'PHOTO_CAPTURED':
        final galleryUri = map['galleryUri'] as String?;
        return GlassPhotoCaptured(
          map['data'] as Uint8List? ?? Uint8List(0),
          path: map['path'] as String?,
          // The bridge sends '' when the gallery write failed and omits the key
          // entirely on an older build; neither is a usable URI.
          galleryUri:
              (galleryUri == null || galleryUri.isEmpty) ? null : galleryUri,
        );
      case 'PHOTO_FAILED':
        return GlassPhotoFailed(map['reason'] as String? ?? 'unknown');
      case 'LIVE_STREAM':
        return GlassLiveStream(
          channel: map['channel'] as int? ?? GlassLiveChannel.wifiAp,
          ssid: map['ssid'] as String?,
          password: map['password'] as String?,
          ipAddress: map['ipAddress'] as String?,
          rtspUrl: map['rtspUrl'] as String? ?? '',
          wifiConnected: map['wifiConnected'] as bool? ?? false,
        );
      case 'WIFI_CONNECTION':
        return GlassWifiConnection(map['connected'] as bool? ?? false);
      case 'WIFI_STA_STATE':
        return GlassWifiStaState(
          active: map['active'] as bool? ?? false,
          ipAddress: map['ipAddress'] as String?,
        );
      case 'SOFT_AP_STATE':
        return GlassSoftApState(
          started: map['started'] as bool? ?? false,
          ssid: map['ssid'] as String?,
          password: map['password'] as String?,
        );
      case 'DEVICE_ACTION':
        return GlassDeviceAction(map['action'] as int? ?? -1);
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

/// §4.18 battery report. The SDK reports levels only, no charging state; for
/// a single device like the glasses only [level] is meaningful.
class GlassBattery extends GlassEvent {
  final int level;
  final int secondaryLevel;
  const GlassBattery({required this.level, this.secondaryLevel = -1});
}

/// A key-press action other than the photo button (§4.13).
class GlassDeviceAction extends GlassEvent {
  final int action;
  const GlassDeviceAction(this.action);
}

/// A full-resolution JPEG pulled off the glasses after a capture.
class GlassPhotoCaptured extends GlassEvent {
  final Uint8List jpeg;

  /// Where the SDK wrote the original on the phone, if the bridge reported it.
  /// App-private storage — no gallery app can see this one.
  final String? path;

  /// MediaStore URI of the public copy in `Pictures/BrailleLens`, or null if
  /// the gallery write failed. Null is not a capture failure: [jpeg] is still
  /// good and the pipeline runs on it either way.
  final String? galleryUri;

  const GlassPhotoCaptured(this.jpeg, {this.path, this.galleryUri});

  /// True once the image is visible to Gallery and Google Photos.
  bool get savedToGallery => galleryUri != null;
}

class GlassPhotoFailed extends GlassEvent {
  final String reason;
  const GlassPhotoFailed(this.reason);
}

class GlassLiveStream extends GlassEvent {
  final int channel;
  final String? ssid;
  final String? password;

  /// Only reported in Station mode (§5.2.3).
  final String? ipAddress;
  final String rtspUrl;

  /// Whether the phone is on the network the feed is served on. In AP mode
  /// false means the automatic join failed and the wearer's phone must join
  /// [ssid] by hand before [rtspUrl] is reachable.
  final bool wifiConnected;
  const GlassLiveStream({
    required this.channel,
    this.ssid,
    this.password,
    this.ipAddress,
    required this.rtspUrl,
    this.wifiConnected = false,
  });
}

/// Phone Wi-Fi link to the glasses AP came up or dropped (WiFi Part).
class GlassWifiConnection extends GlassEvent {
  final bool connected;
  const GlassWifiConnection(this.connected);
}

/// Glasses joined (or left) the phone hotspot in Station mode.
class GlassWifiStaState extends GlassEvent {
  final bool active;
  final String? ipAddress;
  const GlassWifiStaState({required this.active, this.ipAddress});
}

/// The phone hotspot used for Station mode started or stopped.
class GlassSoftApState extends GlassEvent {
  final bool started;
  final String? ssid;
  final String? password;
  const GlassSoftApState({required this.started, this.ssid, this.password});
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
  bool _micStreaming = false;
  bool _phoneOnGlassesWifi = false;
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
  bool get isMicStreaming => _micStreaming;

  /// True while the phone is joined to the glasses AP for the live feed.
  bool get isPhoneOnGlassesWifi => _phoneOnGlassesWifi;
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

  /// Emitted once a live session is up and the phone-side network step is
  /// done, carrying the `rtsp://…:554` URL the preview player opens.
  Stream<GlassLiveStream> get liveStreams => _only<GlassLiveStream>();
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
          _phoneOnGlassesWifi = false;
          _scheduleReconnect();
        }
        _setState(state);
      case GlassBattery(:final level):
        _batteryLevel = level;
        notifyListeners();
      case GlassWifiConnection(:final connected):
        _phoneOnGlassesWifi = connected;
        notifyListeners();
      case GlassLiveStream(:final wifiConnected):
        _phoneOnGlassesWifi = wifiConnected;
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
      if (!queued) {
        // Refused before anything was sent (e.g. no device picked); do not
        // leave the UI sitting on "connecting".
        if (_state == GlassConnectionState.connecting) {
          _setState(GlassConnectionState.disconnected);
        }
        return false;
      }
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

  /// Asks the glasses to take a photo (§4.3). The JPEG and its on-phone path
  /// arrive on [photos] once transferred over the vendor channel.
  ///
  /// [width], [height] and [quality] (0–9) are only sent when given, so the
  /// device otherwise keeps its own defaults.
  Future<bool> takePhoto({
    GlassPhotoType type = GlassPhotoType.origin,
    int? width,
    int? height,
    int? quality,
    bool playSound = true,
  }) async =>
      await _invoke<bool>('takePhoto', {
        'type': type.name,
        'width': width,
        'height': height,
        'quality': quality,
        'playSound': playSound,
      }) ??
      false;

  /// Whether a frame-button press starts a full-resolution capture in native
  /// code. Turn it off when Dart answers the press another way — a snapshot
  /// from the live video feed — so one press yields one image.
  ///
  /// [buttonClicks] fires either way.
  Future<bool> setHardwareShutter(bool enabled) async =>
      await _invoke<bool>('setHardwareShutter', {'enabled': enabled}) ?? enabled;

  /// Publishes [jpeg] to the phone's gallery at `Pictures/BrailleLens` and
  /// returns its MediaStore URI, or null if the write failed.
  ///
  /// Vendor-channel photos are published by the bridge as they arrive. This is
  /// for images that only exist in Dart — a frame grabbed from the live feed —
  /// so both capture paths land in one album.
  Future<String?> saveImageToGallery(Uint8List jpeg) async {
    if (jpeg.isEmpty) return null;
    return _invoke<String>('saveToGallery', {'data': jpeg});
  }

  /// Starts the glasses microphone PCM stream ([micData]); 16 kHz mono once
  /// the device is ready. [meeting] selects the long-form voice mode (§4.9).
  Future<bool> startMicrophone({bool meeting = false}) async =>
      await _invoke<bool>('startMic', {'mode': meeting ? 'meeting' : 'default'}) ??
      false;

  Future<bool> stopMicrophone() async =>
      await _invoke<bool>('stopMic') ?? false;

  /// Brings up the RTSP live feed (§5). Details arrive as [GlassLiveStream]
  /// once the phone is on the right network. Omitted video parameters keep
  /// the device defaults (1280×720, 30 fps, 1 Mbps).
  Future<bool> startLiveStream({
    int channel = GlassLiveChannel.wifiAp,
    int? width,
    int? height,
    int? fps,
    int? bps,
  }) async =>
      await _invoke<bool>('startLiveStream', {
        'channel': channel,
        'width': width,
        'height': height,
        'fps': fps,
        'bps': bps,
      }) ??
      false;

  Future<bool> stopLiveStream() async =>
      await _invoke<bool>('stopLiveStream') ?? false;

  Future<void> refreshBattery() => _invoke<void>('getBattery');

  /// Opens Android's Bluetooth settings so new glasses can be paired. The app
  /// can only connect to devices that are already bonded, and pairing is the
  /// system's job.
  Future<void> openBluetoothSettings() =>
      _invoke<void>('openBluetoothSettings');

  /// Removes the wide-angle lens distortion from a glasses photo (§6).
  ///
  /// Both image dimensions must be even or the SDK rejects it. Writes next to
  /// the input when [outputPath] is null. Returns the corrected file's path,
  /// or null on failure.
  Future<String?> correctImage(String inputPath, {String? outputPath}) =>
      _invoke<String>('correctImage', {
        'inputPath': inputPath,
        'outputPath': outputPath,
      });

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
