# Fingertip Annotation — Braille_fingertip (LabelMe)

Label **60 photos** in `Gold Dataset/Braille_fingertip/` for YOLO26 domain fine-tuning.

---

## 1. Setup

```powershell
py -3.11 -m pip install labelme
py -3.11 -m labelme "Gold Dataset/Braille_fingertip"
```

Work through every `IMG_*.JPG` in the folder.

---

## 2. Output files

For every `IMG_E6076.JPG`, save **`IMG_E6076.json`** in the **same folder**, same basename.

No separate `labels/` folder — LabelMe JSON lives next to the image.

---

## 3. What to draw

| Setting | Value |
|---------|-------|
| Tool | Rectangle (`Ctrl+R`) |
| Label text | `fingertip` (lowercase, exact) |
| Count | **One box per image** (or zero if no finger) |

### Box placement

- Tight around the **distal pad** of the reading finger (usually index).
- Include from the last knuckle through the nail — where skin meets the Braille page.
- Exclude wrist, bracelet, and other fingers when possible.
- Box center should sit on the pad, **not** on blank page background.

### No finger visible

Save the JSON with **zero shapes** (empty). The converter will write an empty `.txt` label.

---

## 4. Do not label

- Braille cells or dots (that is `Gold Dataset/High quality dataset/` — different task).
- Polygons or circles for YOLO (boxes only). Optional `tip_contact` points are for future SkinContour eval — skip for now.

---

## 5. Checklist per image

- [ ] One rectangle labeled `fingertip`, or zero shapes if no tip
- [ ] Box on the reading finger pad, not background
- [ ] `IMG_*.json` saved next to `IMG_*.JPG`
- [ ] No duplicate boxes on the same finger

---

## 6. After all 60 are done

From the BrailleLens repo root:

```powershell
$py = "finger_cell_track\.venv\Scripts\python.exe"
& $py finger_cell_track/yolo_domain_specific/build_dataset.py
& $py finger_cell_track/yolo_domain_specific/pack_for_colab.py
```

Then upload `colab_upload/braille_fingertip_yolo.zip` to Google Drive and open the Colab notebook (see `README.md`).
