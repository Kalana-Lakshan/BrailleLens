# Cycle 2 field sheet — what to record on the phone

Fill this in during the device session. Every block names the exact placeholder in
the Master Test Plan Report it unblocks, so nothing is collected that the report
has no slot for.

Hand the whole filled file back and the report edits can be written from it.

---

## 0. Session header

> Fills: *Test environment record (cycle 1)*, *Revision History* new row, *§5 Risks*
> row "No phone / APK for MD cases".

| Field | Value |
| --- | --- |
| Date of session | |
| Testers present | |
| Commit SHA under test | |
| APK type | debug / release |
| Build command used | |
| APK size | |
| Phone make + model | |
| Android version / API level | |
| Phone RAM | |
| Glasses model + firmware | |
| Glasses battery at start / end | |
| Android SDK version on build machine | |
| TalkBack on or off | |

The report currently says "Android SDK: Not installed on the test machine this
cycle" — that line changes once you build the APK, so record the SDK version.

---

## 1. The 24 MD cases

> Fills: *Headline results* row "Manual / device cases 0/24 executed",
> *Appendix B* (needs a new **Result** column), *§4.2 coverage* row
> "Device / config / load / perf 0%", *entry/exit criteria* row
> "Physical phone + APK — Not met".

Copy this row once per case:

```
MD-xx | tester | PASS / FAIL / BLOCKED | spoken output (verbatim) | expected | DEF-id if fail | notes
```

**Record the spoken output word for word.** For a blind-user product the exact
sentence is the evidence; "it worked" cannot be written into the report.

Extra numbers per case, beyond pass/fail:

| Case | Also record |
| --- | --- |
| MD-01 | Seconds from tap-icon to welcome finishing. Which permission dialogs appeared, in order. |
| MD-02 | Deny camera and mic **separately**, two runs. For each: was anything spoken, or only on-screen? (This is the DEF-02 verdict.) |
| MD-03 | Say each of learning / testing / latest / contest **10 times**. Count how many were accepted. Also count false accepts from unrelated speech. Gives the STT % for *Reliability STT ≥90%* and DEF-08. |
| MD-04 | Taps that registered vs taps made. |
| MD-05 | Run on Gold pg-1…pg-6. Record cells spoken per page. Compare to the GT char counts already in the report (290, 238, 276, 290, 264, 287). |
| MD-06 | The failure sentence, verbatim. |
| MD-08 | 10 known cells per page: expected Sinhala char vs heard Sinhala char, one line each. This is the on-device number the report has never had. Also note which alignment tier was announced, if any. |
| MD-10 | The refusal sentence, verbatim. |
| MD-11 / MD-12 | Shift the page, then reframe. Record whether the answer stayed correct, and whether the degraded path was **spoken** (DEF-03 verdict). |
| MD-14 / MD-15 | 10 right-cell and 10 wrong-cell trials. Earcon heard? Correct scoring? |
| MD-17 / MD-18 | See §3 below — the glasses block. |
| MD-19 | Cover the screen. Seconds to complete the first full loop (budget <15 s). Could the loop be completed by audio alone — yes/no. |
| MD-21 | 30 minutes, 50 lookups. Memory at start / at 15 min / at end. Any ANR or crash. Lookup time for lookups 1–5 vs 46–50. |
| MD-22 | Airplane mode on: does the full loop still work? Storage diff before/after a session — any leftover image or audio file, with paths. |
| MD-24 | Native Sinhala reader, 20 characters: glyph correct yes/no, pronunciation acceptable yes/no, per character. |

### Before you run these — a gap in Appendix B

Appendix B claims MD-01…24 but scripts only 20 IDs. **MD-07, 09, 13, 16, 20 and
23 have no rows.** MD-13 is referenced in Appendix C ("FR6 Fingertip — need
MD-08/13") but never defined. Either write those six cases first, or change every
"0/24" in the report to "0/20". Decide which before the session, so the count in
the results matches the count in the appendix.

---

## 2. Performance numbers

> Fills: *§3.1.4 Performance Profiling* — currently "Not executed on phone", which
> is the largest empty section in the report.

Twenty or more repetitions each; report **median and p95**, not a single reading.

| Measure | SRS budget | Median | p95 |
| --- | --- | --- | --- |
| Earcon latency (trigger → sound) | ≤150 ms | | |
| Speech start (trigger → first phoneme) | ≤300 ms | | |
| Capture → character spoken, end to end | <500 ms (written for streaming) | | |
| Preview frame rate | ≥24 fps | | |
| Peak RAM | <200 MB | | |
| Average CPU | <25% | | |

How to capture them:

```powershell
adb logcat -v time > perf.log          # timestamps give the latency deltas
adb shell dumpsys meminfo com.braillelens.braille_lens_flutter
adb shell dumpsys gfxinfo com.braillelens.braille_lens_flutter
```

A `flutter build apk --profile` build plus DevTools gives cleaner frame and CPU
numbers than logcat. The report already notes the <500 ms budget was written for
a streaming design while the app takes two deliberate captures — record the two
stage times separately so that budget can be formally revised with evidence
rather than marked failed.

---

## 3. Glasses block — and the two live bugs

> Fills: *§3.1.7 Failover and Recovery* ("Hardware failover not run"),
> MD-17/18, *Appendix C* rows FR2 / FR6 / "Reliability camera fallback".

Build the debug entry point, not the normal app:

```powershell
flutter run --release -t lib/main_hardware_debug.dart
```

That boots straight into the hardware debug screen, which prints every event from
the glasses. Copy its log verbatim.

For the frame button, the single most important line is what appears on a press:

| What the log shows | What it means |
| --- | --- |
| `BUTTON_CLICKED (onDeviceTriggeredTakePhoto)` | The press arrives. The bug is downstream, in capture. |
| `DEVICE_ACTION <n>` | The press arrives as a different action id, not as take-photo. Fixable — record `<n>`. |
| nothing at all | The vendor channel never delivered it. Check `DEVICE_READY success=true` appeared first, and that the official AIGlass app is force-stopped. |
| `BUTTON_CLICKED` then `PHOTO_FAILED <reason>` | Press fine, capture failed. Record the reason string. |

Also record:

- `DEVICE_READY success=` value, and the seconds from connect to that line
- Bytes reported on `PHOTO_CAPTURED`
- Milliseconds from `BUTTON_CLICKED` to `PHOTO_CAPTURED`
- On `startMic`: does `MIC_STATE streaming=true` appear, and do mic chunks count up?
- Disconnect the glasses mid-session: seconds until the phone camera takes over, and whether the switch was **spoken** (the ≤500 ms fallback claim)

For the voice input, record separately:

- Whether the microphone permission dialog appeared at all on first launch
- Whether anything is spoken when no command is recognised, or only silence
- The phone's system language setting
- Whether it behaves differently with the glasses disconnected

That last pair matters: `_speech.listen()` is called without a `localeId`, so it
follows the phone's locale — if the phone is set to Sinhala, the English command
words cannot match. And the first attempt asks for `onDevice: true` recognition,
which many phones do not have installed. Both are cheap to confirm on the day and
both would present exactly as "voice input is not getting".

---

## 4. Defect verdicts

> Fills: *Findings log* DEF-01 … DEF-10, *§4.2 exit criteria* row "No High DEF
> open for demo", *Appendix C* row "Usability 3 full feedback".

| ID | Question the session answers | Verdict |
| --- | --- | --- |
| DEF-01 | Did any failure show only on screen, with nothing spoken? Which one. | |
| DEF-02 | Denying a permission — was the consequence spoken? | |
| DEF-03 | When alignment dropped to the scale tier, was it announced? | |
| DEF-07 | Does dwell behave as the SRS describes? | |
| DEF-08 | The MD-03 accept rate — does it clear 90%? | |

DEF-04, DEF-05, DEF-09 and DEF-10 are code and provenance gaps. No amount of
device testing closes them; they need a code change or a re-export, so leave them
Open unless someone does that work.

---

## 5. Also worth fixing while editing

The report as it stands carries three internal inconsistencies:

- The cover says **Version 1.0**, §4 Deliverables says **v1.3**, §4.2 says **Cycle 1 (v1.3)**.
- The cover says executed **17 September 2026**; §4.1.1 and the environment record say the automated re-run was **20 September 2026**.
- The Revision History gives commit **6d7db9f**; Headline results and Appendix A give **b5a828c**.

Pick one of each before adding cycle 2, or the new numbers inherit the confusion.
