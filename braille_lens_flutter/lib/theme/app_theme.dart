import 'package:flutter/material.dart';

class AppTheme {
  // Ultra High-Contrast Palette
  static const Color backgroundBlack = Color(0xFF121212);
  static const Color primaryYellow = Color(0xFFFFD700); // WCAG AAA Contrast
  static const Color learningZoneBlue = Color(0xFF1E3A8A);
  static const Color testingZoneTeal = Color(0xFF064E3B);
  
  // Feedback Colors
  static const Color successCyan = Color(0xFF00E5FF);
  static const Color errorCoral = Color(0xFFFF1744);

  static ThemeData get darkHighContrastTheme {
    return ThemeData(
      brightness: Brightness.dark,
      scaffoldBackgroundColor: backgroundBlack,
      primaryColor: primaryYellow,
      colorScheme: const ColorScheme.dark(
        primary: primaryYellow,
        secondary: successCyan,
        error: errorCoral,
        surface: backgroundBlack,
      ),
      textTheme: const TextTheme(
        displayLarge: TextStyle(fontSize: 32, fontWeight: FontWeight.bold, color: primaryYellow),
        bodyLarge: TextStyle(fontSize: 22, fontWeight: FontWeight.w600, color: Colors.white),
      ),
    );
  }
}
