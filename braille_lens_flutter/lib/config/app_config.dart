/// App configuration — edit before testing on a physical phone.
class AppConfig {
  AppConfig._();

  /// Stage-1 prescan server on your PC (same WiFi as phone).
  ///
  /// 1. Run: `finger_cell_track\.venv\Scripts\python.exe braille_lens_flutter\tools\prescan_server.py`
  /// 2. Replace with your PC's LAN IP (ipconfig → IPv4), e.g. `http://192.168.1.5:8765`
  ///
  /// Set to `null` only if you implement native `prescanPage` MethodChannel on Android.
  static const String? prescanServerUrl = null; // e.g. 'http://192.168.1.5:8765'

  /// 26-class cell CNN (28×28 grayscale). Bundled as a single ONNX file.
  static const String brailleCnnAsset = 'assets/models/braille_model.onnx';

  /// Fingertip YOLO26n — UINT8 quantized first (phone CPU), FP32 fallback.
  static const String fingertipOnnxAsset =
      'assets/models/fingertip_braille_yolo26n_mobile.onnx';

  static const String fingertipOnnxFallbackAsset =
      'assets/models/fingertip_braille_yolo26n.onnx';
}
