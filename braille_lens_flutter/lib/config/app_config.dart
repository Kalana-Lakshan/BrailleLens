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

  /// 64-class Sinhala Braille cell CNN (64×64 grayscale, class index == the
  /// 6-dot cell code). Exported by `braille_cnn/export_onnx.py` from
  /// `braille_cnn/checkpoints/braille_cnn_gold_finetuned.pt` — trained
  /// directly on the Sri Lankan Sinhala Braille chart, unlike the previous
  /// `braille_model.onnx` (a 26-class English Grade-1 letter model that
  /// needed `SinhalaBrailleService`'s English-letter round-trip hack to
  /// produce Sinhala output at all).
  static const String brailleCnnAsset = 'assets/models/braille_cnn.onnx';

  /// Per-code {code, si, en, dots} rows, generated alongside [brailleCnnAsset].
  static const String brailleLabelsAsset = 'assets/models/braille_labels.json';

  /// Fingertip YOLO26n — UINT8 quantized first (phone CPU), FP32 fallback.
  static const String fingertipOnnxAsset =
      'assets/models/fingertip_braille_yolo26n_mobile.onnx';

  static const String fingertipOnnxFallbackAsset =
      'assets/models/fingertip_braille_yolo26n.onnx';

  /// Multi-cell page detector — single-class YOLO26n (`braille_cell`),
  /// 1280×1280 input, up to 800 boxes per page. Exported by
  /// `cell_detect/export_to_onnx.py` from `cell_detect/weights/
  /// braille_cell_best.pt`. UINT8 quantized first (phone CPU), FP32 fallback
  /// — same dual-asset pattern as [fingertipOnnxAsset].
  static const String cellDetectorOnnxAsset =
      'assets/models/braille_cell_yolo26n_mobile.onnx';

  static const String cellDetectorOnnxFallbackAsset =
      'assets/models/braille_cell_yolo26n.onnx';
}
