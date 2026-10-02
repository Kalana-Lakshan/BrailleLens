import 'package:flutter/material.dart';

import 'screens/stt_test_screen.dart';
import 'theme/app_theme.dart';

/// TEMPORARY entry point for checking the Sinhala STT model on a phone.
/// Boots straight into [SttTestScreen], skipping the home screen's
/// voice-command loop so it cannot hold the mic.
///
///   flutter run -t lib/main_stt_test.dart
void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(
    MaterialApp(
      title: 'BrailleLens STT Test',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.darkHighContrastTheme,
      home: const SttTestScreen(),
    ),
  );
}
