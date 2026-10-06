"""Pack everything BrailleLens_LightingRobust_Colab.ipynb needs into one zip.

    finger_cell_track\\.venv\\Scripts\\python.exe lighting_robust/pack_for_colab.py

Writes colab_upload/lighting_robust_bundle.zip (git-ignored). Upload it to
MyDrive/BrailleLens_LightingRobust/ and run the notebook.

Bundle layout:
  gold/high/pg-N.jpeg + pg-N.json     Gold pages, normal capture (LabelMe boxes,
  gold/low/pg-N.jpeg  + pg-N.json     label = dot numbers e.g. "2456")
  weights/braille_cell_gold_degraded.pt   detector currently in the app
  weights/braille_cnn_gold_finetuned.pt   classifier currently in the app
  code/labels.py                       code -> Sinhala label table for export
  real_unlabeled/*.jpg                 raw glasses-camera frames (no labels):
                                       qualitative check + real negatives
  negatives_extra/*                    optional: any photos with NO Braille you
                                       drop into lighting_robust/negatives_extra/
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab_upload" / "lighting_robust_bundle.zip"
GOLD = {"high": ROOT / "Gold Dataset" / "High quality dataset",
        "low": ROOT / "Gold Dataset" / "Low quality dataset"}
WEIGHTS = [ROOT / "cell_detect" / "weights" / "braille_cell_gold_degraded.pt",
           ROOT / "braille_cnn" / "checkpoints" / "braille_cnn_gold_finetuned.pt"]
LABELS_PY = ROOT / "braille_cnn" / "labels.py"
RAW_VIDEOS = [ROOT / "Manual_Tests" / "glass_camera" / "glass camera feed sample.mp4"]
NEG_DIR = Path(__file__).resolve().parent / "negatives_extra"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _video_frames(path: Path, n: int) -> list[tuple[str, bytes]]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * total / n))
        ok, frame = cap.read()
        if not ok:
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            out.append((f"{path.stem.replace(' ', '_')}_{k:03d}.jpg", buf.tobytes()))
    cap.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-per-video", type=int, default=40)
    args = ap.parse_args()

    for p in [*GOLD.values(), *WEIGHTS, LABELS_PY]:
        if not p.exists():
            raise SystemExit(f"Missing: {p}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    counts = {"gold": 0, "real_unlabeled": 0, "negatives_extra": 0}
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for variant, folder in GOLD.items():
            for js in sorted(folder.glob("pg-*.json")):
                img = next((js.with_suffix(e) for e in (".jpeg", ".jpg", ".png")
                            if js.with_suffix(e).exists()), None)
                if img is None:
                    print(f"  skip {js.name}: no image")
                    continue
                z.write(js, f"gold/{variant}/{js.name}")
                z.write(img, f"gold/{variant}/{img.name}")
                counts["gold"] += 1
        for w in WEIGHTS:
            z.write(w, f"weights/{w.name}")
        z.write(LABELS_PY, "code/labels.py")
        for vid in RAW_VIDEOS:
            if vid.exists():
                for name, data in _video_frames(vid, args.frames_per_video):
                    z.writestr(f"real_unlabeled/{name}", data)
                    counts["real_unlabeled"] += 1
        if NEG_DIR.is_dir():
            for f in sorted(NEG_DIR.iterdir()):
                if f.suffix.lower() in IMG_EXT:
                    z.write(f, f"negatives_extra/{f.name}")
                    counts["negatives_extra"] += 1

    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)  {counts}")
    print("Upload to: MyDrive/BrailleLens_LightingRobust/lighting_robust_bundle.zip")


if __name__ == "__main__":
    main()
