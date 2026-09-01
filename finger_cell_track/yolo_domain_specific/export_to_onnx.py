"""Export Braille fingertip YOLO26n to mobile-friendly ONNX.

No retraining — converts existing .pt weights only.

Usage (from repo root)::

    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/yolo_domain_specific/export_to_onnx.py

Outputs in this folder:
  fingertip_braille_yolo26n.onnx          — FP32, fixed 640×640 (max compatibility)
  fingertip_braille_yolo26n_mobile.onnx   — dynamic-quantized UINT8 (smaller/faster CPU)
  fingertip_braille_yolo26n_meta.json     — input/output names + shapes for Flutter
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FCT = _HERE.parent
_DEFAULT_WEIGHTS = _FCT / "weights" / "yolo26n_fingertip_braille_best.pt"
_OUT_FP32 = _HERE / "fingertip_braille_yolo26n.onnx"
_OUT_MOBILE = _HERE / "fingertip_braille_yolo26n_mobile.onnx"
_OUT_META = _HERE / "fingertip_braille_yolo26n_meta.json"

IMGSZ = 640
CLASS_NAMES = ["fingertip"]
CONF_THRESH = 0.25
IOU_THRESH = 0.45


def export_onnx(weights: Path, imgsz: int) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    # Fixed input shape, simplified graph, opset 12 = broad onnxruntime mobile support.
    # nms=False: run NMS in Flutter (standard for YOLO mobile pipelines).
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
    )
    return Path(exported)


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


def write_metadata(onnx_path: Path, imgsz: int) -> None:
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
        "nms_in_app": True,
        "inputs": inputs,
        "outputs": outputs,
        "flutter_notes": (
            "Preprocess: resize letterbox to 640x640, RGB, divide by 255. "
            "Postprocess: YOLO26 detection head — decode boxes, NMS, take highest-conf fingertip. "
            "Tip point = box center."
        ),
    }
    _OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Export fingertip YOLO to mobile ONNX")
    p.add_argument("--weights", type=Path, default=_DEFAULT_WEIGHTS)
    p.add_argument("--imgsz", type=int, default=IMGSZ)
    p.add_argument("--skip-quant", action="store_true", help="Skip UINT8 mobile variant")
    args = p.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}")

    print(f"Exporting {args.weights} -> ONNX (imgsz={args.imgsz})")
    exported = export_onnx(args.weights, args.imgsz)
    shutil.copy2(exported, _OUT_FP32)
    print(f"FP32 ONNX: {_OUT_FP32} ({_OUT_FP32.stat().st_size / 1e6:.1f} MB)")

    write_metadata(_OUT_FP32, args.imgsz)
    print(f"Metadata:  {_OUT_META}")

    if not args.skip_quant:
        if quantize_mobile(_OUT_FP32, _OUT_MOBILE):
            print(
                f"Mobile ONNX (UINT8): {_OUT_MOBILE} "
                f"({_OUT_MOBILE.stat().st_size / 1e6:.1f} MB)"
            )
        else:
            print("Mobile quant skipped — use FP32 file for Flutter")

    print("\nDone. Copy to Flutter:")
    print("  braille_lens_flutter/assets/models/fingertip_braille_yolo26n.onnx")


if __name__ == "__main__":
    main()
