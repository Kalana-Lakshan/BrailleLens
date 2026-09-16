import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Where the app currently expects voice input and prompt output to go.
enum AudioRoute {
  /// Phone earpiece/speaker and built-in mic.
  phone,

  /// Glasses speaker over A2DP, glasses mic over HFP/SCO.
  glasses,
}

/// Forces microphone capture and TTS/earcon playback onto the AI Glass
/// Bluetooth headset, and puts them back on the phone when the glasses drop.
///
/// Android routes these two directions through different profiles, so both have
/// to be handled:
///
/// * **Input** — the glasses mic is an HFP/HSP device, reachable only while a
///   SCO link is up. `AudioManager.startBluetoothSco()` brings that link up;
///   until `ACTION_SCO_AUDIO_STATE_UPDATED` reports CONNECTED, `speech_to_text`
///   would still open the built-in mic. [enableGlassesRoute] waits for that
///   confirmation rather than assuming it.
/// * **Output** — TTS and earcons follow A2DP automatically once the headset is
///   connected, *except* while SCO is active, when everything collapses to the
///   narrowband SCO link. That is acceptable for prompts and is what lets the
///   wearer hear them in the same earpiece.
///
/// The Realtek glasses expose a standard Bluetooth headset to Android, so this
/// is plain `AudioManager` work on the native side and deliberately does not go
/// through the Realtek SDK — it stays useful even if the vendor channel drops.
///
/// Fails soft in the same style as the rest of `lib/services/`: if the native
/// bridge is missing the service reports [AudioRoute.phone] and the app keeps
/// working on built-in hardware.
class AudioRoutingService extends ChangeNotifier {
  AudioRoutingService._();

  static final AudioRoutingService instance = AudioRoutingService._();

  static const MethodChannel _channel =
      MethodChannel('braille_lens/audio_routing');
  static const EventChannel _eventChannel =
      EventChannel('braille_lens/audio_routing_events');

  StreamSubscription<dynamic>? _sub;
  AudioRoute _route = AudioRoute.phone;
  bool _scoActive = false;
  bool _headsetConnected = false;
  bool _available = true;
  String? _lastError;

  AudioRoute get route => _route;
  bool get isOnGlasses => _route == AudioRoute.glasses;

  /// True once the SCO link is actually up — only then is the glasses mic the
  /// one `speech_to_text` will open.
  bool get isScoActive => _scoActive;

  /// True while Android reports a connected Bluetooth headset profile.
  bool get isHeadsetConnected => _headsetConnected;
  bool get isAvailable => _available;
  String? get lastError => _lastError;

  void _listen() {
    if (_sub != null) return;
    try {
      _sub = _eventChannel.receiveBroadcastStream().listen(
        (dynamic raw) {
          if (raw is! Map) return;
          switch (raw['type'] as String?) {
            case 'SCO_STATE':
              _scoActive = raw['active'] as bool? ?? false;
              if (!_scoActive && _route == AudioRoute.glasses) {
                // SCO dropped under us (glasses out of range, call took the
                // link). Reflect reality so callers stop believing the wearer
                // can hear prompts.
                _route = AudioRoute.phone;
              }
              notifyListeners();
            case 'HEADSET_STATE':
              _headsetConnected = raw['connected'] as bool? ?? false;
              if (!_headsetConnected) {
                _scoActive = false;
                _route = AudioRoute.phone;
              }
              notifyListeners();
          }
        },
        onError: (Object e) => debugPrint('[AudioRouting] stream error: $e'),
        cancelOnError: false,
      );
    } on MissingPluginException {
      _available = false;
    }
  }

  /// Refreshes [isHeadsetConnected] from the platform.
  Future<bool> refresh() async {
    _listen();
    final connected = await _invoke<bool>('isHeadsetConnected') ?? false;
    if (connected != _headsetConnected) {
      _headsetConnected = connected;
      notifyListeners();
    }
    return connected;
  }

  /// Routes mic capture and prompt playback to the glasses.
  ///
  /// Returns false if no headset is connected or the SCO link never came up
  /// within [timeout]; the caller should then keep using phone hardware.
  Future<bool> enableGlassesRoute({
    Duration timeout = const Duration(seconds: 5),
  }) async {
    _listen();
    if (!await refresh()) {
      _lastError = 'No Bluetooth headset connected';
      return false;
    }

    final ok = await _invoke<bool>('startSco', {
      'timeoutMs': timeout.inMilliseconds,
    });
    if (ok != true) {
      _lastError = 'SCO link did not come up';
      _route = AudioRoute.phone;
      notifyListeners();
      return false;
    }

    _scoActive = true;
    _route = AudioRoute.glasses;
    _lastError = null;
    notifyListeners();
    debugPrint('[AudioRouting] input+output on glasses');
    return true;
  }

  /// Tears the SCO link down and returns audio to the phone.
  Future<void> enablePhoneRoute() async {
    _listen();
    await _invoke<void>('stopSco');
    _scoActive = false;
    if (_route != AudioRoute.phone) {
      _route = AudioRoute.phone;
      notifyListeners();
    }
    debugPrint('[AudioRouting] input+output on phone');
  }

  /// Picks the best available route: glasses when a headset is connected,
  /// phone otherwise. Call after a connect/disconnect on
  /// `GlassDeviceService`.
  Future<AudioRoute> syncToGlasses({required bool glassesConnected}) async {
    if (glassesConnected) {
      final ok = await enableGlassesRoute();
      if (ok) return AudioRoute.glasses;
    }
    await enablePhoneRoute();
    return AudioRoute.phone;
  }

  Future<T?> _invoke<T>(String method, [Map<String, dynamic>? args]) async {
    try {
      return await _channel.invokeMethod<T>(method, args);
    } on MissingPluginException {
      _available = false;
      _lastError = 'Audio routing bridge not available in this build';
      return null;
    } on PlatformException catch (e) {
      _lastError = e.message ?? e.code;
      debugPrint('[AudioRouting] $method failed: ${e.code} ${e.message}');
      return null;
    }
  }

  @override
  void dispose() {
    _sub?.cancel();
    super.dispose();
  }
}
