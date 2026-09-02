# Phone test — both ONNX models on-device

## Models wired

| Model | Asset | Stage | Role |
|-------|-------|-------|------|
| `braille_model.onnx` | `assets/models/braille_model.onnx` | Photo 1 | Classify each cell crop (26 English letters a–z) |
| `fingertip_braille_yolo26n_mobile.onnx` | `assets/models/..._mobile.onnx` | Photo 2 | Fingertip box → hit-test → letter from photo 1 map |

**No PC server required** for basic testing.

---

## Step 1 — Verify models load

1. Build and install on phone:
   ```powershell
   cd braille_lens_flutter
   flutter pub get
   flutter run
   ```
   Or release APK:
   ```powershell
   flutter build apk --release
   ```
   Install: `build\app\outputs\flutter-apk\app-release.apk`

2. On home screen tap **ONNX** (top right).

3. You should see:
   - ✓ `CNN  braille_model.onnx`
   - ✓ `YOLO  fingertip_braille_yolo26n_mobile.onnx`

4. Tap **CNN · 26 samples** — runs bundled test images (may not be 100% accurate; model is a prototype).

5. Tap **YOLO · camera** — point at finger on Braille page; yellow/cyan box should appear.

---

## Step 2 — Full two-photo flow

1. Home → **Learning Mode**
2. Bottom should show: `CNN: braille_model.onnx · YOLO: fingertip_braille_yolo26n_mobile.onnx`
3. **Stage 1:** Page only, no finger → **Capture page**
   - Wait for “Classifying cells N/M…”
   - Yellow boxes appear on frozen image
4. **Stage 2:** Finger on one cell → **Capture finger on cell**
   - `👆 FINGER TRACKED` + letter at bottom
   - TTS speaks the character

**Keep phone still** between photo 1 and photo 2.

---

## Known limits (honest)

- **Letters are English a–z** from `braille_model.onnx`, not Sinhala (needs 64-class model later).
- **Cell boxes** use dot-grid detection in Dart, not YOLO — count may differ from your 234-cell PC run.
- **Accuracy** depends on lighting and holding the phone steady.

---

## If something fails

| Symptom | Fix |
|---------|-----|
| CNN FAILED on ONNX screen | Run `flutter clean` then `flutter pub get`; confirm `assets/models/braille_model.onnx` exists |
| YOLO FAILED | Confirm `fingertip_braille_yolo26n_mobile.onnx` in assets and pubspec |
| No cells on page capture | More even light; fill frame with page; avoid shadows |
| Wrong letter under finger | Rescan page; keep same angle for both photos |
| YOLO no box | Retake finger photo; or tap fingertip when prompted |
