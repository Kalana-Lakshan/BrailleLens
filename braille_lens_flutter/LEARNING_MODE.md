# Learning Mode — two-stage covered character

## Flow

1. **Stage 1** — Capture hand-free page photo → prescan builds `CellMap` (yellow boxes + Sinhala labels per cell).
2. **Stage 2** — Capture finger on page → fingertip position → **geometry hit-test** against `CellMap` → show covered Sinhala letter.

Stage 2 does **not** run a CNN on the finger photo. The character comes from stage 1 only.

## Prescan (stage 1)

Choose one:

### A) PC dev server (recommended for now)

```powershell
cd BrailleLens
finger_cell_track\.venv\Scripts\python.exe braille_lens_flutter\tools\prescan_server.py
```

In `lib/main.dart` before `runApp`:

```dart
import 'services/prescan_bridge.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  PrescanBridge.prescanServerUrl = 'http://192.168.x.x:8765'; // your PC IP
  runApp(const BrailleLensApp());
}
```

Phone and PC must be on the same WiFi.

### B) Native MethodChannel

Implement `prescanPage` on Android/iOS (`com.braillelens/vision`) returning JSON:

```json
{"width": 720, "height": 1280, "cells": [{"id": 0, "x0": ..., "char": "ක", "code": 19, ...}]}
```

Wire your existing prescan code here if you already have 234-cell detection on device.

## Fingertip (stage 2)

Optional ONNX: copy `finger_cell_track/yolo_domain_specific/fingertip_braille_yolo26n_mobile.onnx`
to `assets/models/` and add to `pubspec.yaml` assets.

If the model is missing, the app prompts you to **tap the fingertip contact point** on the photo.

## Bug fix: `#0 space · no dots`

This happened when the fingertip was hit-tested in the **wrong coordinate frame** (or on code-0 gap cells).

Fixed in:

- `coordinate_mapper.dart` — map finger photo → prescan pixels
- `cell_hit_test.dart` — prefer non-empty cells; nearest centre on overlap
- `covered_cell_service.dart` — contact point at bottom of fingertip box
- `image_fit.dart` — overlays aligned with displayed image

## Key files

| File | Role |
|------|------|
| `lib/services/covered_cell_service.dart` | Covered character lookup |
| `lib/services/cell_hit_test.dart` | Geometry hit-test |
| `lib/services/coordinate_mapper.dart` | Two-photo coordinate map |
| `lib/screens/learning_screen.dart` | Two-stage UI |
