import 'dart:async';

import 'package:flutter/foundation.dart';

import 'glass_device_service.dart';

/// Connection states surfaced to the home-screen indicator.
enum BluetoothConnectionState { checking, connected, notFound, unavailable }

/// A discovered BrailleLens wearable device.
class BrailleDevice {
  final String id;
  final String name;
  const BrailleDevice({required this.id, required this.name});
}

/// Thin adapter that keeps the existing HomeScreen indicator API while the real
/// work happens in [GlassDeviceService].
///
/// The mock scan this class used to perform is gone: [scanForGlasses] now does
/// a real discovery-and-connect against the AI Glass over the Realtek vendor
/// channel. Kept as a separate type so `home_screen.dart` did not have to
/// change shape; new code should talk to [GlassDeviceService] directly and read
/// its richer state (battery, mic, live feed).
class BluetoothService {
  final GlassDeviceService _glass;
  StreamSubscription<GlassEvent>? _sub;

  /// Fired whenever the underlying connection state changes, so a screen can
  /// refresh its indicator without polling.
  final ValueNotifier<BluetoothConnectionState> stateListenable =
      ValueNotifier<BluetoothConnectionState>(BluetoothConnectionState.checking);

  BluetoothService({GlassDeviceService? glass})
      : _glass = glass ?? GlassDeviceService.instance {
    _sub = _glass.events.listen((e) {
      if (e is GlassConnectionChanged) stateListenable.value = state;
    });
  }

  BluetoothConnectionState get state => switch (_glass.state) {
        GlassConnectionState.connected => BluetoothConnectionState.connected,
        GlassConnectionState.scanning ||
        GlassConnectionState.connecting ||
        GlassConnectionState.idle =>
          BluetoothConnectionState.checking,
        GlassConnectionState.unavailable =>
          BluetoothConnectionState.unavailable,
        GlassConnectionState.disconnected => BluetoothConnectionState.notFound,
      };

  BrailleDevice? get connectedDevice {
    final d = _glass.device;
    if (d == null || !_glass.isConnected) return null;
    return BrailleDevice(id: d.address, name: d.name);
  }

  bool get isConnected => _glass.isConnected;

  /// Battery percentage of the connected glasses, or -1 when unknown.
  int get batteryLevel => _glass.batteryLevel;

  /// Scans for AI Glass and connects to the first match.
  ///
  /// [scanDuration] bounds discovery only; the connect handshake gets its own
  /// timeout inside [GlassDeviceService.connect].
  Future<bool> scanForGlasses({
    Duration scanDuration = const Duration(seconds: 8),
  }) async {
    debugPrint('[BluetoothService] scanning for AI Glass…');

    if (!await _glass.initialize()) {
      debugPrint('[BluetoothService] glasses bridge unavailable');
      stateListenable.value = state;
      return false;
    }

    // The scan lists every paired device (glasses-like names first, then the
    // active Bluetooth headset), so only auto-connect to one of those — never
    // to arbitrary paired earbuds or a car kit.
    final found = await _glass.scanForGlasses(timeout: scanDuration);
    final GlassDevice? pick = found
            .where((d) => d.likelyGlasses)
            .firstOrNull ??
        found.where((d) => d.audioConnected).firstOrNull;
    if (pick == null) {
      debugPrint('[BluetoothService] no glasses found');
      stateListenable.value = state;
      return false;
    }

    final ok = await _glass.connect(address: pick.address);
    debugPrint(
      '[BluetoothService] connect to ${pick.name}: '
      '${ok ? 'connected' : 'failed'}',
    );
    stateListenable.value = state;
    return ok;
  }

  Future<void> disconnect() async {
    await _glass.disconnect();
    stateListenable.value = state;
  }

  void dispose() {
    _sub?.cancel();
    _sub = null;
    stateListenable.dispose();
  }
}
