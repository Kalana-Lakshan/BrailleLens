import 'package:flutter/material.dart';
import 'package:media_kit/media_kit.dart';

import 'screens/temp_capture_screen.dart';
import 'theme/app_theme.dart';

/// TEMPORARY entry point for dataset collection. Boots straight into
/// [TempCaptureScreen], skipping the home screen's voice-command loop so it
/// cannot grab the mic while photos are being collected.
///
///   flutter run -t lib/main_temp_capture.dart
void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // Not used by this screen, but GlassDeviceService is shared with the live
  // preview path and constructing a Player without it crashes.
  MediaKit.ensureInitialized();
  runApp(
    MaterialApp(
      title: 'BrailleLens Capture Check',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.darkHighContrastTheme,
      home: const TempCaptureScreen(),
    ),
  );
}
