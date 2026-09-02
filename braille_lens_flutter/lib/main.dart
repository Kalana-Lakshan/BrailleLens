import 'package:flutter/material.dart';
import 'config/app_config.dart';
import 'screens/home_screen.dart';
import 'services/prescan_bridge.dart';
import 'theme/app_theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
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
