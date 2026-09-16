import 'package:flutter/material.dart';
import 'package:media_kit/media_kit.dart';
import 'config/app_config.dart';
import 'screens/home_screen.dart';
import 'services/prescan_bridge.dart';
import 'theme/app_theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // Loads libmpv, used for the AI Glass RTSP preview. Must run before any
  // Player is constructed.
  MediaKit.ensureInitialized();
  PrescanBridge.prescanServerUrl = AppConfig.prescanServerUrl;
  runApp(const BrailleLensApp());
}

class BrailleLensApp extends StatelessWidget {
  const BrailleLensApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'BrailleLens Offline',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.darkHighContrastTheme,
      home: const HomeScreen(),
    );
  }
}
