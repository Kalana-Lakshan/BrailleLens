"""Export the trained Braille *cell* detector to ONNX and bundle it straight
into the Flutter app's assets — same pattern as
`finger_cell_track/yolo_domain_specific/export_to_onnx.py`, except this
writes directly into `braille_lens_flutter/assets/models/` instead of
leaving a manual-copy step, and the imgsz is 1280 not 640 (see
`cell_detect/configs/cells.yaml`: a cell is ~36x58px on a 1704x2340 DSBI
page — at 640 it shrinks to ~14px and detection suffers; 1280 keeps it
near 27px).

No retraining — converts existing .pt weights only.

Usage (from repo root)::

    .venv\\Scripts\\python.exe cell_detect/export_to_onnx.py

Outputs, written directly into braille_lens_flutter/assets/models/:
  braille_cell_yolo26n.onnx          — FP32, fixed 1280x1280 (max compatibility)
  braille_cell_yolo26n_mobile.onnx   — dynamic-quantized UINT8 (smaller/faster CPU)
  braille_cell_yolo26n_meta.json     — input/output shapes + provenance for Flutter
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_DEFAULT_WEIGHTS = _HERE / "weights" / "braille_cell_best.pt"
_ASSET_DIR = _ROOT / "braille_lens_flutter" / "assets" / "models"
_OUT_FP32 = _ASSET_DIR / "braille_cell_yolo26n.onnx"
_OUT_MOBILE = _ASSET_DIR / "braille_cell_yolo26n_mobile.onnx"
_OUT_META = _ASSET_DIR / "braille_cell_yolo26n_meta.json"

IMGSZ = 1280
CLASS_NAMES = ["braille_cell"]
CONF_THRESH = 0.25
IOU_THRESH = 0.45
# cells.yaml's training config caps at 800 (measured max 623 cells/page, 95th
# pct 566) -- match it here so a dense real page isn't silently truncated at
# Ultralytics' default export top-k of 300.
MAX_DET = 800


def _sha256_16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def export_onnx(weights: Path, imgsz: int) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    # Fixed input shape, simplified graph, opset 12 = broad onnxruntime mobile
    # support -- same flags as the fingertip export. nms=False: YOLO26 uses an
    # end-to-end (dual-assignment) head, so the exported output is already a
    # fixed-size, de-duplicated detection list without a separate NMS op; the
    # flag just controls whether Ultralytics appends its own NMS graph node,
    # which mobile onnxruntime builds don't reliably support anyway.
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        opset=12,
        dynamic=False,
        half=False,
        nms=False,
        batch=1,
        device="cpu",
        max_det=MAX_DET,
    )
    return Path(exported)


def clamp_ir_version(onnx_path: Path) -> int:
    """The onnxruntime build bundled in the Flutter `onnxruntime` package
    rejects ONNX IR version 10+ ("Unsupported model IR version: 10, max
    supported IR version: 9"). Ultralytics' opset-12 export hasn't needed
    this in practice, but check and clamp defensively anyway -- opset 12 is
    well within IR 9's range. Returns the final IR version."""
    import onnx

    m = onnx.load(str(onnx_path))
    if m.ir_version > 9:
        m.ir_version = 9
        onnx.save_model(m, str(onnx_path), save_as_external_data=False)
    return m.ir_version


def quantize_mobile(fp32_path: Path, out_path: Path) -> bool:
    """Dynamic UINT8 quantization for smaller/faster CPU mobile inference."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        print("onnxruntime quantization unavailable — skipping mobile quant")
        return False

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(out_path),
        weight_type=QuantType.QUInt8,
    )
    return out_path.exists()


def write_metadata(onnx_path: Path, imgsz: int, weights: Path, ir_version: int) -> None:
    import onnx

    model = onnx.load(str(onnx_path))
    inputs = []
    for inp in model.graph.input:
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.dim_value else d.dim_param or "?")
        inputs.append({"name": inp.name, "shape": shape, "dtype": "float32"})

    outputs = []
    for out in model.graph.output:
        shape = []
        for d in out.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.dim_value else d.dim_param or "?")
        outputs.append({"name": out.name, "shape": shape})

    meta = {
        "model": onnx_path.name,
        "task": "detect",
        "class_names": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "imgsz": imgsz,
        "input_layout": "NCHW",
        "color_format": "RGB",
        "normalize": {"scale": 1.0 / 255.0, "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        "conf_threshold": CONF_THRESH,
        "iou_threshold": IOU_THRESH,
        "max_det": MAX_DET,
        "nms_in_app": True,
        "ir_version": ir_version,
        "source_weights": weights.name,
        "source_weights_sha256_16": _sha256_16(weights),
        "inputs": inputs,
        "outputs": outputs,
        "flutter_notes": (
            "Preprocess: resize letterbox to 1280x1280, RGB, divide by 255. "
            "Postprocess: YOLO26 detection head, output already de-duplicated "
            "(end-to-end head) -- keep every row above conf_threshold (a page "
            "has many cells, unlike the fingertip model which only keeps the "
            "single best row). Box centers are the cell centroids."
        ),
    }
    _OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Export the Braille cell detector to ONNX")
    p.add_argument("--weights", type=Path, default=_DEFAULT_WEIGHTS)
    p.add_argument("--imgsz", type=int, default=IMGSZ)
    p.add_argument("--skip-quant", action="store_true", help="Skip UINT8 mobile variant")
    args = p.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}")

    _ASSET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {args.weights} -> ONNX (imgsz={args.imgsz})")
    print(f"  source sha256[:16] = {_sha256_16(args.weights)}")
    exported = export_onnx(args.weights, args.imgsz)
    if exported.resolve() != _OUT_FP32.resolve():
        import shutil

        shutil.copy2(exported, _OUT_FP32)
    print(f"FP32 ONNX: {_OUT_FP32} ({_OUT_FP32.stat().st_size / 1e6:.1f} MB)")

    ir_version = clamp_ir_version(_OUT_FP32)
    print(f"  IR version: {ir_version}")

    write_metadata(_OUT_FP32, args.imgsz, args.weights, ir_version)
    print(f"Metadata:  {_OUT_META}")

    if not args.skip_quant:
        if quantize_mobile(_OUT_FP32, _OUT_MOBILE):
            print(
                f"Mobile ONNX (UINT8): {_OUT_MOBILE} "
                f"({_OUT_MOBILE.stat().st_size / 1e6:.1f} MB)"
            )
        else:
            print("Mobile quant skipped — use FP32 file for Flutter")

    print("\nDone. Assets already written into:")
    print("  braille_lens_flutter/assets/models/")


if __name__ == "__main__":
    main()
