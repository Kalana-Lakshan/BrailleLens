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
  /// braille_cell_gold_degraded.pt` — fine-tuned with an added synthetic
  /// camera-degradation augmentation to close the domain gap with the AiSee
  /// glasses camera (see `cell_detect/CAMERA_DEGRADATION.md`). Consistent
  /// mAP50 gains on held-out Gold pages across high/low/degraded quality
  /// variants; mixed on 2 real unseen AiSee photos (fewer merged boxes on
  /// one, lower recall on faint dots on the other) -- see
  /// `braille_cell_yolo26n_degraded_meta.json`'s provenance_notes. UINT8
  /// quantized first (phone CPU), FP32 fallback — same dual-asset pattern as
  /// [fingertipOnnxAsset]. Previous default: `braille_cell_yolo26n[_mobile].onnx`
  /// (from `braille_cell_best.pt`), still bundled as an asset if reverting.
  static const String cellDetectorOnnxAsset =
      'assets/models/braille_cell_yolo26n_degraded_mobile.onnx';

  static const String cellDetectorOnnxFallbackAsset =
      'assets/models/braille_cell_yolo26n_degraded.onnx';

  /// Minimum box score for [cellDetectorOnnxAsset]. Belongs with the model,
  /// not the detector code: scores are calibrated per export.
  ///
  /// The degraded model's raw head scores lower than the old end-to-end one,
  /// so the old 0.50 threw most cells away — on live phone frames all of them
  /// (page scan found 0 cells in both modes). On the 24 annotated Gold pages
  /// (5,875 cells): 0.50 → P 0.96 / R 0.41; 0.25 → P 0.74 / R 0.75 (F1 0.747,
  /// vs 0.52 at best for the old model). 0.25 is also what the model's own
  /// meta.json specifies, and phone frames score lower still, so recall
  /// matters more than the last few points of precision.
  /// If reverting to `braille_cell_yolo26n[_mobile].onnx`, set this to 0.50.
  static const double cellDetectorConfThreshold = 0.25;
}
