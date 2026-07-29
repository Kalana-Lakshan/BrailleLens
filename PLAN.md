# BrailleLens — Live Camera Sinhala Output: Implementation Plan

## Goal

Live camera feed focused on a Braille page → classified Sinhala sentence printed in the terminal, refreshing in real time.

---

## What Is Already Done

| Component | File(s) | Status |
|---|---|---|
| CNN model | `cnn.py` | Done |
| Synthetic training | `train.py` | Done — 100% on synthetic test set |
| DBSI fine-tuning | `finetune_dbsi.py` | Done — 98.44% on real flatbed scans |
| Static image inference | `infer_page.py` | Done — works on a saved photo |
| Dot detection & cell clustering | `dot_detect.py` | Done |
| English label decoding | `labels.py` | Done |
| Sinhala label table | `labels.py` | **Partial** — only 24/64 codes, some duplicates |

---

## Remaining Work — 3 Branches

### Branch Order

```
Branch 1: fix/sinhala-labels        ← fix the lookup table (correctness first, no model risk)
     ↓
Branch 2: feat/camera-capture       ← build the live camera loop
     ↓
Branch 3: feat/live-sinhala-output  ← assemble and display the Sinhala sentence
```

---

## Branch 1: `fix/sinhala-labels`

**Goal:** Complete and verify the Sinhala Braille label table before anything else depends on it.

**Files touched:** `braille_cnn/labels.py`

**What to do:**
- Cross-check every entry in `CODE_TO_SINHALA` against the "සිංහල බ්‍රේල් අක්ෂර මාලාව" reference chart
- Fill in all missing codes for vowels, consonants, and vowel signs visible in the chart
- Resolve the three duplicate `"ව"` entries (codes 37, 39, 58)
- Add combining vowel sign codes as separate entries where the chart shows them

**Commits:**
1. `fix(labels): correct duplicate ව entries and add missing consonants from chart`
2. `feat(labels): add combining vowel sign codes from Sinhala Braille chart`
3. `test(labels): add sanity-check script that prints all 64 code mappings`

---

## Branch 2: `feat/camera-capture`

**Goal:** Open the webcam, stream frames into the existing `run_auto` inference pipeline.

**Files touched:** `braille_cnn/camera.py` (new), `braille_cnn/infer_page.py` (refactor)

**What to do:**
- Refactor `_classify` in `infer_page.py` so the model is loaded **once at startup**, not on every call (current Bug 4)
- Create `braille_cnn/camera.py` using `cv2.VideoCapture` to open the webcam
- Convert each captured frame to a grayscale PIL Image (same format `infer_page.py` already expects)
- Add a **frame-stability check**: compute the pixel difference between consecutive frames; skip inference while the camera is moving (prevents flickering results)
- Add a `--camera` flag to the entry point that starts the capture loop

**Commits:**
1. `refactor(infer_page): load model once at startup, pass into _classify as argument`
2. `feat(camera): add VideoCapture loop with grayscale frame conversion`
3. `feat(camera): add frame-stability check to suppress inference during motion`
4. `feat(camera): wire stable camera frames into existing run_auto inference pipeline`

---

## Branch 3: `feat/live-sinhala-output`

**Goal:** Assemble classified cells into a clean Sinhala sentence and print it in the terminal, refreshing in place.

**Files touched:** `braille_cnn/camera.py`, `braille_cnn/infer_page.py`

**What to do:**
- Set `--lang si` as the default for camera mode
- Replace the per-cell confidence-score output with a single assembled Sinhala sentence string
- Add a **confidence threshold** (suggested: `0.6`) — cells below the threshold print `_` as a placeholder so the user knows a cell was detected but uncertain
- Use `\r` (carriage return) or `curses` to overwrite the previous terminal line on each stable frame, so the output updates in place rather than scrolling
- Confirm `_assemble_line_text` word-spacing logic works correctly with Sinhala Unicode (Sinhala is left-to-right, no special handling needed)

**Commits:**
1. `feat(output): assemble classified cells into a clean Sinhala sentence string`
2. `feat(output): add confidence threshold with _ placeholder for uncertain cells`
3. `feat(output): in-place terminal refresh — each stable frame overwrites the last`
4. `feat(camera): set --lang si as default for camera mode`

---

## Known Risks & Notes

| Risk | Mitigation |
|---|---|
| Model does not generalise to handheld camera photos (documented in RESULTS.md) | May need to collect real phone-camera crops and fine-tune again after Branch 2 is working |
| DBSI fine-tuned checkpoint has catastrophic forgetting on synthetic data | Always use `braille_cnn_dbsi_finetuned.pt` for real images — never the synthetic-only checkpoint |
| `torch.load` missing `weights_only=True` will break on PyTorch 2.6+ | Add `weights_only=True` to all `torch.load` calls when touching those files |
| Sinhala label table was hand-transcribed and may have errors | Branch 1 must be reviewed by a native Sinhala reader before TTS use |
