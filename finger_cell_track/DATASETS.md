# Fingertip datasets for BrailleLens

Local folder (gitignored): `finger_cell_track/datasets/`

Suggested layout after you download manually:

```text
finger_cell_track/datasets/
  TI1K/                    # already present
  fingertip256/            # from mobiofp Drive zip
  roboflow_finger_tip/     # Roboflow YOLO export
  roboflow_tip_detect/     # Roboflow export
  roboflow_fingertip_*/    # other Roboflow FingerTip sets
  SCUT_Ego_Finger/         # after HCII approval
  SCUT_Ego_Gesture/        # after HCII approval
  CASIA_EgoGesture/        # after signed agreement
```

---

## Best match for tip-only / YOLO boxes

| Dataset | Size | Labels | Access | Fit for BrailleLens | Links |
|---------|------|--------|--------|---------------------|-------|
| **TI1K** | 1,000 images | Hand box + thumb & index tip `(x,y)` | Free on GitHub | Strong — tip points, egocentric-ish | [GitHub repo](https://github.com/MahmudulAlam/TI1K-Dataset) · [ZIP (master)](https://github.com/MahmudulAlam/TI1K-Dataset/archive/refs/heads/master.zip) |
| **mobiofp fingertip256** | 256 images | YOLO boxes on fingertip | Google Drive in repo | Strong — Ultralytics format; small | [mobiofp repo](https://github.com/rotiroti/mobiofp) · [fingertip256obj.zip](https://drive.google.com/file/d/15akG23eTbT2TZv78kJHRImCWeKS1XL2X) · [YOLOv8n weights amd64](https://drive.google.com/file/d/1THsT9OcTbjl_Qadw_4WqvdOEoJxVYuDW) · [weights arm64](https://drive.google.com/file/d/1ia2Vkf4UfRI6Q_SIV1k30_WiorrlRd3K) |
| **Roboflow Finger Tip Detection** | ~538 images | Object detection (finger tips) | Roboflow export (YOLO) | Good starter | [Universe project](https://universe.roboflow.com/first-leo0f/finger-tip-detection-i4xqf) |
| **Roboflow tip_detect** | ~14.5k images | Tip keypoints | Roboflow | Good — keypoints, not Braille domain | [Universe project](https://universe.roboflow.com/pinpoint/tip_detect) |
| **Roboflow FingerTip (several)** | ~300–900 each | Tip / finger classes | Roboflow Universe | Mixed quality; check before use | [Search: fingertip](https://universe.roboflow.com/search?q=fingertip) · [Search: FingerTip](https://universe.roboflow.com/search?q=FingerTip) |

### Manual steps (tip-only sets)

1. **TI1K** — clone or download ZIP above → extract to `datasets/TI1K/`  
   *(Already downloaded locally: 900 train + 100 test.)*
2. **fingertip256** — open Drive link → save `fingertip256obj.zip` → extract to `datasets/fingertip256/`
3. **Roboflow** — create free account → open project → **Download Dataset** → format **YOLOv8** → unzip under `datasets/roboflow_*`

---

## Large egocentric fingertip datasets (application needed)

| Dataset | Size | Labels | Access | Fit | Links |
|---------|------|--------|--------|-----|-------|
| **SCUT-Ego-Finger / EgoGesture (HCII)** | ~93k frames (24 subjects) | Hand + fingertips (YOLSE paper) | Lab form / email | Best research set; password / approval | [HCII data center](http://www.hcii-lab.net/data/) · [SCUTEgoFinger index](http://www.hcii-lab.net/data/SCUTEgoFinger/index.htm) · [GitHub docs](https://github.com/hyichao/EgoFinger.HCII.SCUT) · Contact: hyichao@foxmail.com |
| **SCUT-Ego-Gesture** | ~29k RGB (8 gesture classes used in papers) | Gestures + fingertips | Same HCII source + form | Good; used in fingertip papers | [HCII data](http://www.hcii-lab.net/data/) · [Partition notes (MahmudulAlam)](https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection/tree/master/dataset) |
| **CASIA EgoGesture** | Large RGB-D videos | Gesture clips (not tip boxes) | Signed agreement | Weak for tip YOLO — gestures, not tip boxes | [Dataset page](https://nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html) · [Paper PDF](https://nlpr.ia.ac.cn/iva/yfzhang/datasets/EgoGesture.pdf) |

### Manual steps (large egocentric)

1. **SCUT-Ego-Finger** — visit HCII pages → follow their application / email process → place extracted data in `datasets/SCUT_Ego_Finger/`
2. **SCUT-Ego-Gesture** — same HCII approval channel; see MahmudulAlam dataset folder for train/val/test file lists after you have images
3. **CASIA EgoGesture** — download & sign agreement from the CASIA page → email as instructed → download within their time window → `datasets/CASIA_EgoGesture/`

None of these include **Braille paper**. Use them to pretrain a tip detector, then fine-tune on your own IP Webcam Braille frames.

---

## Local status

| Dataset | Path | Status |
|---------|------|--------|
| TI1K | `datasets/TI1K/` | **OK** — 900 train + 100 test + `annotation/` |
| mobiofp repo (docs) | `datasets/_mobiofp_repo/` | README only; zip is manual Drive download |
| Others | — | Download manually using links above |

---

## Suggested order for BrailleLens

1. **TI1K** (local) — convert tip points → YOLO boxes, smoke-train **YOLO26n**  
2. **fingertip256** (Drive) — add real YOLO tip boxes  
3. Optional Roboflow sets for more variety  
4. Optional **SCUT-Ego-Finger** for scale  
5. **Your Braille IP Webcam images** — required for domain fit  
