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
- Repo: https://github.com/rotiroti/mobiofp  
- Dataset zip: https://drive.google.com/file/d/15akG23eTbT2TZv78kJHRImCWeKS1XL2X  
- Optional YOLOv8n tip weights (amd64): https://drive.google.com/file/d/1THsT9OcTbjl_Qadw_4WqvdOEoJxVYuDW  
- Optional YOLOv8n tip weights (arm64): https://drive.google.com/file/d/1ia2Vkf4UfRI6Q_SIV1k30_WiorrlRd3K  
- Extract to: `datasets/fingertip256/`

### Roboflow Finger Tip Detection (~538 images)
- https://universe.roboflow.com/first-leo0f/finger-tip-detection-i4xqf  
- Export as **YOLOv8** → `datasets/roboflow_finger_tip/`

### Roboflow tip_detect (~14.5k images — keypoints)
- https://universe.roboflow.com/pinpoint/tip_detect  
- Export as YOLO / keypoint format → `datasets/roboflow_tip_detect/`

### Other Roboflow FingerTip sets (mixed quality)
- Search fingertip: https://universe.roboflow.com/search?q=fingertip  
- Search FingerTip: https://universe.roboflow.com/search?q=FingerTip  

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

1. TI1K (GitHub ZIP)  
2. mobiofp fingertip256 (Google Drive)  
3. Roboflow Finger Tip Detection  
4. Your own Braille tip photos  
5. SCUT-Ego-Finger (after lab approval)  

More notes: see [DATASETS.md](DATASETS.md).
