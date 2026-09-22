import 'package:flutter/material.dart';

import '../services/glass_device_service.dart';
import '../theme/app_theme.dart';

/// Lets the wearer pick which paired Bluetooth device is the AI Glass.
///
/// Auto-connect guesses from the device name and from which device is the
/// phone's active headset, but product names vary by vendor, so the guess can
/// be wrong or find nothing. This lists every paired device — the same thing
/// the Realtek reference app does — and connects to the one that is tapped.
///
/// Returns the device that was connected, or null if the sheet was dismissed
/// or the connection failed.
Future<GlassDevice?> showGlassDevicePicker(BuildContext context) async {
  final glass = GlassDeviceService.instance;

  // Listed before the sheet opens: this is a fast read of the bonded list,
  // not a radio scan, so there is nothing to wait through.
  final devices = await glass.scanForGlasses();
  if (!context.mounted) return null;

  final picked = await showModalBottomSheet<GlassDevice>(
    context: context,
    backgroundColor: const Color(0xFF1E1E1E),
    isScrollControlled: true,
    builder: (ctx) => SafeArea(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Padding(
            padding: EdgeInsets.fromLTRB(20, 18, 20, 6),
            child: Text(
              'SELECT YOUR GLASSES',
              style: TextStyle(
                color: AppTheme.primaryYellow,
                fontWeight: FontWeight.bold,
                letterSpacing: 1.2,
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Text(
              devices.isEmpty
                  ? 'No paired devices found. Pair the glasses in Android '
                      'Bluetooth settings first, then come back.'
                  : 'Paired devices on this phone. Tap your AI Glass.',
              style: const TextStyle(color: Colors.white70, fontSize: 13),
            ),
          ),
          const SizedBox(height: 8),
          Flexible(
            child: ListView.builder(
              shrinkWrap: true,
              itemCount: devices.length,
              itemBuilder: (_, i) {
                final d = devices[i];
                // Why this device is a likely match, so the wearer is not
                // guessing between half a dozen identical-looking rows.
                final hints = <String>[
                  if (d.likelyGlasses) 'name looks like glasses',
                  if (d.audioConnected) 'connected as headset',
                ];
                return ListTile(
                  leading: Icon(
                    d.likelyGlasses || d.audioConnected
                        ? Icons.visibility
                        : Icons.bluetooth,
                    color: d.likelyGlasses || d.audioConnected
                        ? AppTheme.primaryYellow
                        : Colors.white38,
                  ),
                  title: Text(
                    d.name,
                    style: const TextStyle(color: Colors.white),
                  ),
                  subtitle: Text(
                    hints.isEmpty ? d.address : '${d.address} · ${hints.join(' · ')}',
                    style: const TextStyle(color: Colors.white54, fontSize: 12),
                  ),
                  onTap: () => Navigator.pop(ctx, d),
                );
              },
            ),
          ),
          const Divider(height: 1, color: Colors.white12),
          ListTile(
            leading: const Icon(Icons.settings_bluetooth, color: Colors.white70),
            title: const Text(
              'Pair new glasses in Bluetooth settings',
              style: TextStyle(color: Colors.white70, fontSize: 14),
            ),
            onTap: () {
              // Pairing itself is Android's job; the app can only connect to
              // devices that are already bonded.
              GlassDeviceService.instance.openBluetoothSettings();
              Navigator.pop(ctx);
            },
          ),
          const SizedBox(height: 8),
        ],
      ),
    ),
  );

  if (picked == null) return null;
  final ok = await glass.connect(address: picked.address);
  return ok ? picked : null;
}
