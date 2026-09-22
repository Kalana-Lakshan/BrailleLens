import 'package:flutter/material.dart';
import 'package:media_kit/media_kit.dart';

import 'screens/hardware_debug_screen.dart';
import 'theme/app_theme.dart';

/// TEMPORARY entry point for AI Glass bring-up. Boots straight into
/// [HardwareDebugScreen], skipping the home screen's voice-command loop so it
/// cannot grab the mic or talk over the TTS test.
///
///   flutter run -t lib/main_hardware_debug.dart
void main() {
  WidgetsFlutterBinding.ensureInitialized();
  MediaKit.ensureInitialized();
  runApp(
    MaterialApp(
      title: 'BrailleLens HW Debug',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.darkHighContrastTheme,
      home: const HardwareDebugScreen(),
    ),
  );
}
