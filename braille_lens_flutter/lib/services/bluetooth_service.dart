import 'dart:async';
import 'package:flutter/foundation.dart';

enum BluetoothConnectionState { checking, connected, notFound, unavailable }

/// Represents a discovered BrailleLens wearable device.
class BrailleDevice {
  final String id;
  final String name;
  const BrailleDevice({required this.id, required this.name});
}

/// Mock Bluetooth service that simulates a hardware scan for BrailleLens glasses.
///
/// Integration Hook: Replace [scanForGlasses] with real BLE scanning using
/// `flutter_blue_plus` or `flutter_bluetooth_serial` when physical hardware
/// is available. The rest of the app (HomeScreen BT indicator) will work
/// unchanged because it only reads [state] and [connectedDevice].
class BluetoothService {
  BluetoothConnectionState _state = BluetoothConnectionState.checking;
  BrailleDevice? _connectedDevice;

  BluetoothConnectionState get state => _state;
  BrailleDevice? get connectedDevice => _connectedDevice;
  bool get isConnected => _state == BluetoothConnectionState.connected;

  /// Simulates a Bluetooth LE scan for BrailleLens glasses.
  ///
  /// Current behavior (mock): always returns [BluetoothConnectionState.notFound]
  /// after a short simulated scan delay.
  ///
  /// To simulate a successful connection during development, change the block
  /// marked "MOCK" below.
  Future<bool> scanForGlasses({
    Duration scanDuration = const Duration(seconds: 3),
  }) async {
    _state = BluetoothConnectionState.checking;
    debugPrint('[BluetoothService] Scanning for BrailleLens glasses...');

    // Simulate scan network delay
    await Future.delayed(scanDuration);

    // ── MOCK BLOCK ────────────────────────────────────────────────────────────
    // Change _state = BluetoothConnectionState.connected and set _connectedDevice
    // to simulate a paired device. Leave as-is for frontend-only mode.
    _state = BluetoothConnectionState.notFound;
    _connectedDevice = null;

    // Example of what a real connection result would look like:
    // _state = BluetoothConnectionState.connected;
    // _connectedDevice = const BrailleDevice(
    //   id: 'BL:00:11:22:33:44',
    //   name: 'BrailleLens Glasses v1',
    // );
    // ── END MOCK BLOCK ────────────────────────────────────────────────────────

    debugPrint(
      '[BluetoothService] Scan complete — state: ${_state.name}',
    );
    return isConnected;
  }

  Future<void> disconnect() async {
    _state = BluetoothConnectionState.notFound;
    _connectedDevice = null;
    debugPrint('[BluetoothService] Disconnected.');
  }

  void dispose() {
    disconnect();
  }
}
