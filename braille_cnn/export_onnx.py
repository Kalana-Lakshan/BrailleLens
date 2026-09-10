"""Export the fine-tuned SimpleBrailleCNN to ONNX for on-device (Flutter) use.

    py -3.11 -m braille_cnn.export_onnx

Outputs (into the Flutter app's asset folder):
    braille_lens_flutter/assets/models/braille_cnn.onnx     float32 [1,1,64,64] -> [1,64]
    braille_lens_flutter/assets/models/braille_labels.json  64 rows: {code, si, en, dots}

The Dart side must preprocess each cell crop exactly like braille_cnn/normalize.py:
    grayscale -> 64x64 (bicubic) -> (px - mean) / (max(std, 10) * 4) + 0.5 -> clip[0,1]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from braille_cnn.cnn import SimpleBrailleCNN
from braille_cnn.labels import code_to_label
from braille_cnn.normalize import normalize_crop

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "braille_cnn" / "checkpoints" / "braille_cnn_gold_finetuned.pt"
ASSET_DIR = ROOT / "braille_lens_flutter" / "assets" / "models"
IMG_SIZE = 64
NUM_CLASSES = 64


def _load(ckpt: Path) -> SimpleBrailleCNN:
    sd = torch.load(str(ckpt), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd and not any(
        k.startswith(("features", "classifier")) for k in sd
    ):
        sd = sd["model"]
    model = SimpleBrailleCNN(NUM_CLASSES)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def _dot_string(code: int) -> str:
    dots = [str(d) for d in range(1, 7) if code & (1 << (d - 1))]
    return "no dots" if not dots else "dot " + ", ".join(dots) if len(dots) == 1 \
        else "dots " + ", ".join(dots)


def _write_labels(path: Path) -> None:
    rows = []
    for code in range(NUM_CLASSES):
        rows.append({
            "code": code,
            "si": code_to_label(code, lang="si"),
            "en": code_to_label(code, lang="en"),
            "dots": _dot_string(code),
        })
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _parity(model: SimpleBrailleCNN, onnx_path: Path) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    max_abs = 0.0
    agree = 0
    n = 40
    for _ in range(n):
        raw = rng.integers(0, 255, size=(IMG_SIZE, IMG_SIZE)).astype(np.float32)
        norm = normalize_crop(raw)
        x = norm[None, None].astype(np.float32)
        with torch.no_grad():
            t = model(torch.from_numpy(x)).numpy()
        o = sess.run(None, {"input": x})[0]
        max_abs = max(max_abs, float(np.abs(t - o).max()))
        agree += int(t.argmax() == o.argmax())
    print(f"  parity: argmax agree {agree}/{n}, max|logit delta| {max_abs:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = ASSET_DIR / "braille_cnn.onnx"
    labels_path = ASSET_DIR / "braille_labels.json"

    print(f"loading {args.checkpoint.name}")
    model = _load(args.checkpoint)

    dummy = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    # Fold any external-data sidecar back into one file, and clamp the ONNX IR
    # version to 9 -- the onnxruntime build bundled in the Flutter `onnxruntime`
    # package rejects IR 10 ("Unsupported model IR version: 10, max supported
    # IR version: 9"). opset 17 is well within IR 9's range, so this is safe.
    try:
        import onnx

        m = onnx.load(str(onnx_path))
        if m.ir_version > 9:
            m.ir_version = 9
        onnx.save_model(m, str(onnx_path), save_as_external_data=False)
        sidecar = onnx_path.with_suffix(".onnx.data")
        if sidecar.exists():
            sidecar.unlink()
    except Exception as exc:  # noqa: BLE001
        print(f"  (skipped external-data fold / ir clamp: {exc})")

    _write_labels(labels_path)
    size_kb = onnx_path.stat().st_size / 1024
    print(f"wrote {onnx_path}  ({size_kb:.0f} KB)")
    print(f"wrote {labels_path}")
    _parity(model, onnx_path)


if __name__ == "__main__":
    main()
