"""Pack the Braille-domain data BrailleLens_Fingertip_Robust_Colab.ipynb needs.

    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/fingertip_robust/pack_for_colab.py

Writes colab_upload/fingertip_robust_bundle.zip (git-ignored) and
colab_upload/fingertip_robust_review.jpg (look at it before uploading: it shows
the glasses-video pseudo-labels and the no-fingertip frames).

The public fingertip data (TI1K + Roboflow, ~12.5k images) is NOT in this
bundle: the notebook reads it from Drive (fingertip_combined_yolo26.zip or the
unzipped MyDrive/BrailleLens/fingertip_yolo26 folder).

Bundle layout:
  braille/images|labels/{train,val,test}   60 Braille fingertip photos (48/6/6),
                                           resized to 1600 px long side
  gold_pages/{train,val,test}/*.jpg        Gold Braille page photos, no hand
                                           (pages 1-8 / 9,12 / 10,11, high + low)
  glasses/train_pos/images|labels          raw glasses-camera frames with a fingertip,
  glasses/eval_pos/images|labels             boxes from the current Braille model
                                             (confident frames + short-gap interpolation)
  glasses/train_neg/images                 glasses frames with no hand in view
  glasses/eval_neg/images
  glasses/ambiguous/images                 hand holding/reaching the page: visual check only
  weights/current_braille.pt               fingertip model in the app now
  weights/combined.pt                      TI1K + Roboflow model trained 3 Oct
  negatives_extra/*                        optional photos with NO fingertip you drop in
                                           finger_cell_track/fingertip_robust/negatives_extra/
  bundle_meta.json                         frame ranges + pseudo-label stats

Train/eval glasses frames come from separate time ranges of the video so the
evaluation frames are never near-duplicates of training frames.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "colab_upload" / "fingertip_robust_bundle.zip"
REVIEW = ROOT / "colab_upload" / "fingertip_robust_review.jpg"
BRAILLE = ROOT / "finger_cell_track" / "yolo_domain_specific" / "datasets" / "braille_fingertip_yolo"
GOLD = {"high": ROOT / "Gold Dataset" / "High quality dataset",
        "low": ROOT / "Gold Dataset" / "Low quality dataset"}
GOLD_SPLIT = {"train": range(1, 9), "val": (9, 12), "test": (10, 11)}
GOLD_EXCLUDE = {("low", 9), ("low", 10), ("low", 12)}  # a thumb holds the page edge: not a clean "no fingertip" image
VIDEO = ROOT / "Manual_Tests" / "glass_camera" / "glass camera feed sample.mp4"
TEACHER = ROOT / "braille_lens_flutter" / "assets" / "models" / "fingertip_braille_yolo26n.onnx"
WEIGHTS = {"current_braille.pt": ROOT / "finger_cell_track" / "weights" / "yolo26n_fingertip_braille_best.pt",
           "combined.pt": ROOT / "finger_cell_track" / "weights" / "yolo26n_fingertip_combined_best.pt"}
NEG_DIR = HERE / "negatives_extra"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Segments of the glasses video, checked by eye on a contact sheet (frame numbers).
POS_TRAIN, POS_EVAL = (610, 880), (900, 1160)          # index fingertip on the page
NEG_TRAIN = [(100, 400), (1185, 1260)]                 # page only, no hand
NEG_EVAL = [(420, 580), (1270, 1330), (1385, 1425)]
AMBIGUOUS = (1340, 1489)                               # hand holding / reaching the page
PSEUDO_CONF = 0.5
MAX_GAP = 6


def _jpg(img, q=92) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    assert ok
    return buf.tobytes()


def _resize_long(img, long_side):
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    return img if s >= 1 else cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


class Teacher:
    """Current app model; full frame + two square crops so small glasses-camera tips are not missed."""

    def __init__(self, path: Path):
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def _run(self, bgr):
        h, w = bgr.shape[:2]
        sc = min(640 / w, 640 / h)
        nw, nh = round(w * sc), round(h * sc)
        px, py = (640 - nw) // 2, (640 - nh) // 2
        canvas = np.full((640, 640, 3), 114, np.uint8)
        canvas[py:py + nh, px:px + nw] = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (nw, nh))
        out = self.sess.run(None, {"images": canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255})[0][0]
        r = out[out[:, 4].argmax()]
        return np.array([(r[0] - px) / sc, (r[1] - py) / sc, (r[2] - px) / sc, (r[3] - py) / sc]), float(r[4])

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        best = self._run(bgr)
        if w > h:
            for x0 in (0, w - h):
                b, s = self._run(bgr[:, x0:x0 + h])
                if s > best[1]:
                    best = (b + np.array([x0, 0, x0, 0]), s)
        return best


def _read_frames(cap, lo, hi, step):
    for i in range(lo, hi + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            yield i, f


def _pseudo_labels(cap, teacher, lo, hi):
    """Teacher box per frame: confident frames, plus linear interpolation across gaps <= MAX_GAP."""
    det = {}
    for i, f in _read_frames(cap, lo, hi, 1):
        b, s = teacher(f)
        if s >= PSEUDO_CONF:
            det[i] = b
    keys = sorted(det)
    labels = dict(det)
    for a, b in zip(keys, keys[1:]):
        if 1 < b - a <= MAX_GAP:
            wa = det[a][2] - det[a][0]
            ca, cb = (det[a][:2] + det[a][2:]) / 2, (det[b][:2] + det[b][2:]) / 2
            if np.linalg.norm(ca - cb) <= 2 * wa:
                for i in range(a + 1, b):
                    t = (i - a) / (b - a)
                    labels[i] = det[a] * (1 - t) + det[b] * t
    return labels, len(det)


def _yolo_line(box, w, h):
    x0, y0, x1, y1 = np.clip(box, 0, [w, h, w, h])
    return f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--braille-long-side", type=int, default=1600)
    args = ap.parse_args()
    for p in [BRAILLE, *GOLD.values(), VIDEO, TEACHER, *WEIGHTS.values()]:
        if not p.exists():
            raise SystemExit(f"Missing: {p}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    review: list[np.ndarray] = []
    add = lambda k, n=1: counts.__setitem__(k, counts.get(k, 0) + n)

    def tile(img, box=None, text="", col=(0, 255, 0)):
        vis = img.copy()
        if box is not None:
            cv2.rectangle(vis, tuple(map(int, box[:2])), tuple(map(int, box[2:])), col, 4)
        cv2.putText(vis, text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        s = min(400 / vis.shape[1], 300 / vis.shape[0])
        vis = cv2.resize(vis, (round(vis.shape[1] * s), round(vis.shape[0] * s)))
        canvas = np.zeros((300, 400, 3), np.uint8)
        canvas[:vis.shape[0], :vis.shape[1]] = vis
        return canvas

    teacher = Teacher(TEACHER)
    cap = cv2.VideoCapture(str(VIDEO))
    meta = {"video": VIDEO.name, "pos_train": POS_TRAIN, "pos_eval": POS_EVAL, "neg_train": NEG_TRAIN,
            "neg_eval": NEG_EVAL, "ambiguous": AMBIGUOUS, "pseudo_conf": PSEUDO_CONF, "max_gap": MAX_GAP}

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for split in ("train", "val", "test"):
            for ip in sorted((BRAILLE / "images" / split).iterdir()):
                if ip.suffix.lower() not in IMG_EXT:
                    continue
                img = _resize_long(cv2.imread(str(ip)), args.braille_long_side)
                z.writestr(f"braille/images/{split}/{ip.stem}.jpg", _jpg(img, 93))
                lp = BRAILLE / "labels" / split / f"{ip.stem}.txt"
                z.writestr(f"braille/labels/{split}/{ip.stem}.txt", lp.read_text() if lp.exists() else "")
                add(f"braille_{split}")

        for variant, folder in GOLD.items():
            for split, pages in GOLD_SPLIT.items():
                for n in pages:
                    ip = next((folder / f"pg-{n}{e}" for e in (".jpeg", ".jpg", ".png") if (folder / f"pg-{n}{e}").exists()), None)
                    if ip is None or (variant, n) in GOLD_EXCLUDE:
                        continue
                    img = cv2.imread(str(ip))
                    z.writestr(f"gold_pages/{split}/{variant}_pg{n}.jpg", _jpg(img))
                    add(f"gold_{split}")
                    if n in (1, 10) and variant == "low":
                        review.append(tile(img, text=f"neg gold {variant} {n}"))

        for name, (lo, hi) in (("train_pos", POS_TRAIN), ("eval_pos", POS_EVAL)):
            labels, n_conf = _pseudo_labels(cap, teacher, lo, hi)
            meta[f"{name}_confident_frames"] = n_conf
            meta[f"{name}_labelled_frames"] = len(labels)
            meta[f"{name}_frames"] = hi - lo + 1
            for i, f in _read_frames(cap, lo, hi, 2):
                h, w = f.shape[:2]
                if name == "train_pos" and i not in labels:
                    continue
                z.writestr(f"glasses/{name}/images/f{i:05d}.jpg", _jpg(f))
                z.writestr(f"glasses/{name}/labels/f{i:05d}.txt", _yolo_line(labels[i], w, h) if i in labels else "")
                add(name)
                if i % 40 == 0:
                    review.append(tile(f, labels.get(i), f"{name} {i}" + ("" if i in labels else " (no box)")))

        for name, ranges in (("train_neg", NEG_TRAIN), ("eval_neg", NEG_EVAL)):
            for lo, hi in ranges:
                for i, f in _read_frames(cap, lo, hi, 4):
                    z.writestr(f"glasses/{name}/images/f{i:05d}.jpg", _jpg(f))
                    add(name)
                    if i % 100 == 0:
                        review.append(tile(f, text=f"{name} {i}"))
        for i, f in _read_frames(cap, *AMBIGUOUS, 6):
            z.writestr(f"glasses/ambiguous/images/f{i:05d}.jpg", _jpg(f))
            add("ambiguous")

        for arc, p in WEIGHTS.items():
            z.write(p, f"weights/{arc}")
        if NEG_DIR.is_dir():
            for f in sorted(NEG_DIR.iterdir()):
                if f.suffix.lower() in IMG_EXT:
                    z.write(f, f"negatives_extra/{f.name}")
                    add("negatives_extra")
        meta["counts"] = counts
        z.writestr("bundle_meta.json", json.dumps(meta, indent=2))

    cols = 6
    review += [np.zeros_like(review[0])] * (-len(review) % cols)
    cv2.imwrite(str(REVIEW), np.vstack([np.hstack(review[r:r + cols]) for r in range(0, len(review), cols)]))
    print(json.dumps(meta, indent=2))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    print(f"Review sheet: {REVIEW}")
    print("Upload to: MyDrive/BrailleLens_Fingertip_Domain/fingertip_robust_bundle.zip")


if __name__ == "__main__":
    main()
