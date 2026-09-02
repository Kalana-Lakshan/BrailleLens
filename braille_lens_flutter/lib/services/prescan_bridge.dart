import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

import '../models/braille_cell.dart';
import 'prescan_onnx_service.dart';

/// Stage-1 prescan: detect all Braille cells and label each crop.
///
/// Order:
/// 1. On-device ONNX (`braille_model.onnx` + dot grid) — **default for phone**
/// 2. Native MethodChannel `prescanPage`
/// 3. HTTP PC server (`prescan_server.py`) if [prescanServerUrl] is set
class PrescanBridge {
  static const MethodChannel _channel = MethodChannel('com.braillelens/vision');

  static String? prescanServerUrl;
  static final PrescanOnnxService _onDevice = PrescanOnnxService();

  static Future<bool> ensureOnDeviceReady() => _onDevice.initialize();

  Future<CellMap> prescanPage(
    Uint8List jpegBytes, {
    String lang = 'si',
    String backend = 'dnn',
    void Function(int done, int total)? onProgress,
  }) async {
    // 1) On-device CNN + dot grid
    try {
      final map = await _onDevice.prescanPage(
        jpegBytes,
        onProgress: onProgress,
      );
      if (map.cells.isNotEmpty) {
        debugPrint('[PrescanBridge] on-device: ${map.cells.length} cells');
        return map;
      }
    } catch (e) {
      debugPrint('[PrescanBridge] on-device failed: $e');
      // fall through to native / HTTP
    }

    // 2) Native plugin
    try {
      final dynamic raw = await _channel.invokeMethod<dynamic>(
        'prescanPage',
        {'jpeg': jpegBytes, 'lang': lang, 'backend': backend},
      );
      if (raw != null) {
        final map = raw is String
            ? jsonDecode(raw) as Map<String, dynamic>
            : Map<String, dynamic>.from(raw as Map);
        final result = CellMap.fromJson(map);
        if (result.cells.isNotEmpty) {
          debugPrint('[PrescanBridge] native: ${result.cells.length} cells');
          return result;
        }
      }
    } on MissingPluginException {
      debugPrint('[PrescanBridge] no native prescan plugin');
    } on PlatformException catch (e) {
      debugPrint('[PrescanBridge] native error: ${e.message}');
    }

    // 3) HTTP dev server
    final base = prescanServerUrl;
    if (base != null && base.isNotEmpty) {
      final uri = Uri.parse('$base/prescan?lang=$lang&backend=$backend');
      final resp = await http
          .post(
            uri,
            headers: {'Content-Type': 'image/jpeg'},
            body: jpegBytes,
          )
          .timeout(const Duration(seconds: 120));
      if (resp.statusCode == 200) {
        final result = CellMap.fromJson(
          jsonDecode(resp.body) as Map<String, dynamic>,
        );
        debugPrint('[PrescanBridge] HTTP: ${result.cells.length} cells');
        return result;
      }
      throw Exception('Prescan server ${resp.statusCode}: ${resp.body}');
    }

    throw PrescanUnavailableException(
      'On-device prescan failed. Check lighting and that braille_model.onnx is bundled. '
      'Optional: set PrescanBridge.prescanServerUrl for PC server.',
    );
  }
}

class PrescanUnavailableException implements Exception {
  final String message;
  PrescanUnavailableException(this.message);
  @override
  String toString() => message;
}
