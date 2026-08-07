# Fingertip dataset download links

Manual downloads for BrailleLens tip YOLO training.  
Save archives under `finger_cell_track/datasets/` (gitignored).

---

## Tip-only / YOLO-friendly (download these first)

### TI1K (1,000 images — tip points)
- Repo: https://github.com/MahmudulAlam/TI1K-Dataset  
- ZIP: https://github.com/MahmudulAlam/TI1K-Dataset/archive/refs/heads/master.zip  
- Local (if already cloned): `datasets/TI1K/`

### mobiofp fingertip256 (256 images — YOLO boxes)

**Important:** The GitHub repo does **not** contain the images. Data lives only on the author’s Google Drive. Those Drive files are often **not public** (“Request access”), so automated / anonymous download fails.

- Repo (code + notebooks only): https://github.com/rotiroti/mobiofp  
- Dataset zip (may need access request): https://drive.google.com/file/d/15akG23eTbT2TZv78kJHRImCWeKS1XL2X  
- Optional YOLOv8n tip weights amd64: https://drive.google.com/file/d/1THsT9OcTbjl_Qadw_4WqvdOEoJxVYuDW  
- Optional YOLOv8n tip weights arm64: https://drive.google.com/file/d/1ia2Vkf4UfRI6Q_SIV1k30_WiorrlRd3K  
- Extract to: `datasets/fingertip256/`

**If Drive asks for permission:**

1. Click **Request access** on Drive (author must approve), **or**  
2. Open a GitHub issue on [rotiroti/mobiofp](https://github.com/rotiroti/mobiofp/issues) asking for public/shared links, **or**  
3. **Skip mobiofp for now** — use **TI1K** (already local) + **Roboflow** + your own Braille tip photos.  

Note: fingertip256 is built from **IIITD ISPFDv1** fingerphotos. The full IIITD set also needs an institute license: https://iab-rubric.org/index.php/cmbd → email `databases@iab-rubric.org`.

### Roboflow Finger Tip Detection (~538 images)
- https://universe.roboflow.com/first-leo0f/finger-tip-detection-i4xqf  
- Export: prefer **YOLO26** if listed; otherwise **YOLOv8** (same Ultralytics TXT + `data.yaml` — works with YOLO26 training)  
- Unzip → `datasets/roboflow_finger_tip/`

### Roboflow tip_detect (~14.5k images — keypoints) — SKIP
- https://universe.roboflow.com/pinpoint/tip_detect  
- **Not usable for BrailleLens.** Labels are **pen / stylus tips** (“pinpoint”), not human fingertips.  
- Do not download or train on this for finger-on-Braille detection.

### Other Roboflow FingerTip sets (mixed quality)
- Search fingertip: https://universe.roboflow.com/search?q=fingertip  
- Search FingerTip: https://universe.roboflow.com/search?q=FingerTip  
- Open sample images first; skip anything that is pens/stylus/darts, not fingers.  

**Note:** YOLO26 and YOLOv8 use the **same** box label layout (`class xc yc w h` normalized). Choosing YOLO26 on Roboflow is ideal naming; YOLOv8 export is still valid for `YOLO("yolo26n.pt").train(...)`.  

---

## Large egocentric (application / agreement required)

### SCUT-Ego-Finger (~93k frames)
- HCII data center: http://www.hcii-lab.net/data/  
- Dataset index: http://www.hcii-lab.net/data/SCUTEgoFinger/index.htm  
- GitHub docs: https://github.com/hyichao/EgoFinger.HCII.SCUT  
- Contact: hyichao@foxmail.com  
- Extract to: `datasets/SCUT_Ego_Finger/`

### SCUT-Ego-Gesture (~29k RGB used in papers)
- HCII data: http://www.hcii-lab.net/data/  
- Train/val/test file lists: https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection/tree/master/dataset  
- Extract to: `datasets/SCUT_Ego_Gesture/`

### CASIA EgoGesture (RGB-D videos — weak for tip YOLO boxes)
- Page: https://nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html  
- Paper: https://nlpr.ia.ac.cn/iva/yfzhang/datasets/EgoGesture.pdf  
- Extract to: `datasets/CASIA_EgoGesture/`

---

## Also collect yourself (required for Braille domain)

Use IP Webcam over a Braille page, tip-only close-ups:

- Target: **200–500+** frames  
- Save to: `datasets/braille_tip_own/`  
- Label one box per image: `fingertip`

---

## Suggested download order

1. TI1K (GitHub ZIP) — already local under `datasets/TI1K-Dataset-master/`  
2. Roboflow Finger Tip Detection — already local as `datasets/Finger Tip Detection.v1i.yolo26/`  
3. **Skip** mobiofp if Drive is locked; **skip** `tip_detect` (pen tips)  
4. Your own Braille tip photos  

### Train (Kaggle GPU — not TPU)

```powershell
finger_cell_track\.venv\Scripts\python.exe finger_cell_track/prepare_tip_yolo_dataset.py --clean
# Zip datasets/fingertip_yolo26 and upload to Kaggle
# Open BrailleLens_Fingertip_YOLO26_Kaggle.ipynb with Accelerator = GPU
```

More notes: see [DATASETS.md](DATASETS.md).
