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
  ///
  /// Active: `fingertip_robust_yolo26n[_mobile].onnx` (from
  /// `finger_cell_track/weights/yolo26n_fingertip_robust_best.pt`,
  /// `finger_cell_track/fingertip_robust/BrailleLens_Fingertip_Robust_Colab.ipynb`)
  /// — trained on public fingertips plus oversampled Braille photos and
  /// glasses-camera frames, with lighting augmentation and no-fingertip
  /// negatives. Previous default: `fingertip_braille_yolo26n[_mobile].onnx`,
  /// still bundled for reverting (with threshold 0.25).
  static const String fingertipOnnxAsset =
      'assets/models/fingertip_robust_yolo26n_mobile.onnx';

  static const String fingertipOnnxFallbackAsset =
      'assets/models/fingertip_robust_yolo26n.onnx';

  /// Minimum score of the best fingertip row; below it there is no fingertip.
  /// Chosen on validation data that includes no-fingertip images
  /// (`fingertip_robust_yolo26n_meta.json`).
  static const double fingertipConfThreshold = 0.35;

  /// Multi-cell page detector — single-class YOLO26n (`braille_cell`),
  /// 1280×1280 input, up to 800 boxes per page.
  ///
  /// Lineage: `braille_cell_best.pt` (DSBI+Angelina) -> gold fine-tune ->
  /// `braille_cell_gold_degraded.pt` (camera-degradation augmentation, see
  /// `cell_detect/CAMERA_DEGRADATION.md`) -> `braille_cell_lighting.pt`
  /// (teammate's lighting/Braille-free-negatives fine-tune, see
  /// `lighting_robust/results/`) -> **active**: `braille_cell_lighting_negatives.pt`
  /// -- the lighting checkpoint further fine-tuned on 39 of our own
  /// hard-negative images (barcodes/QR codes, dense code/text, shop clutter
  /// -- see `cell_detect/make_synthetic_negatives.py` and the "Non-braille
  /// images" folder), same proven 12-image positive set, no other changes
  /// (`cell_detect/gold_finetune_2.py`). Held-out test mAP50 0.926 vs 0.940
  /// for the lighting checkpoint alone (negligible difference); full pipeline
  /// (detect+classify+reconstruct text, 6 Gold text pages) 89.9% letters-only
  /// vs 88.2% -- a real gain on 4 of 6 pages, the best full-pipeline result
  /// measured all project -- see `reports/eval/gold_text.md`'s "hard-negatives
  /// fine-tuned on top of the teammate's lighting-robust checkpoint" section.
  /// Also confirmed clean (0 false positives) on a real AiSee photo with
  /// background clutter (person, laptop, signage), same as the parent.
  ///
  /// A separate hard-negative fine-tune off an earlier point in this lineage
  /// (`braille_cell_yolo26n_negatives[_mobile].onnx`, from
  /// `braille_cell_gold_negatives.pt` -- built on `braille_cell_gold.pt`,
  /// before the degradation/lighting work) independently found and fixed the
  /// same zero-non-braille-exposure problem first; superseded by the active
  /// checkpoint above once `braille_cell_lighting.pt` was tracked, since
  /// stacking the same hard negatives on the stronger lighting-robust base
  /// won on every metric. Still bundled, kept for history/comparison.
  ///
  /// UINT8 quantized first (phone CPU), FP32 fallback — same dual-asset
  /// pattern as [fingertipOnnxAsset]. Every earlier stage in the lineage
  /// above also still bundled if reverting.
  static const String cellDetectorOnnxAsset =
      'assets/models/braille_cell_yolo26n_lighting_negatives_mobile.onnx';

  static const String cellDetectorOnnxFallbackAsset =
      'assets/models/braille_cell_yolo26n_lighting_negatives.onnx';

  /// Minimum box score for [cellDetectorOnnxAsset]. Belongs with the model,
  /// not the detector code: scores are calibrated per export.
  ///
  /// Carried over from `braille_cell_lighting.pt` (0.40, its own sweep
  /// against held-out Gold pages and Braille-free images -- see
  /// `lighting_robust/results/`), not independently re-swept for this
  /// specific derived checkpoint -- a live-detection threshold sweep
  /// matching that same methodology hasn't been done here. A `.val()`-based
  /// sweep (0.15-0.50) on held-out Gold pages showed a flat F1 (~0.926)
  /// across the whole range, so this is a reasonable default, not a
  /// rigorously re-calibrated one.
  ///
  /// The degraded model's raw head scores lower than the old end-to-end one,
  /// so the old 0.50 threw most cells away — on live phone frames all of them
  /// (page scan found 0 cells in both modes). On the 24 annotated Gold pages
  /// (5,875 cells): 0.50 → P 0.96 / R 0.41; 0.25 → P 0.74 / R 0.75 (F1 0.747,
  /// vs 0.52 at best for the old model). 0.25 is also what the model's own
  /// meta.json specifies, and phone frames score lower still, so recall
  /// matters more than the last few points of precision.
  /// If reverting to `braille_cell_yolo26n[_mobile].onnx`, set this to 0.50.
  ///
  /// Lighting model: 0.40, from a sweep of the UINT8 file run like this
  /// service (letterbox, NMS 0.40) on held-out Gold pages 10/11 and
  /// Braille-free images. F1 peaks at 0.40-0.45 (0.944-0.945); 0.40 keeps more
  /// recall for phone frames. vs the degraded model at 0.25: F1 0.944 vs
  /// 0.875, real low-light recall 0.978 vs 0.940, false boxes on Braille-free
  /// images 5.9/img vs 58.2. Details: `lighting_robust/results/`.
  /// If reverting to the degraded model, set this back to 0.25.
  static const double cellDetectorConfThreshold = 0.40;
}
