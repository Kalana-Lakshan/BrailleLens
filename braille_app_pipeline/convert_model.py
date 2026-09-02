import os
import torch
import torch.nn as nn
from model import BrailleCNN

def convert():
    model_dir = os.path.dirname(__file__)
    pth_path = os.path.join(model_dir, "braille_model.pth")
    
    # Target assets folder in Flutter project
    assets_dir = os.path.abspath(os.path.join(model_dir, "..", "braille_lens_flutter", "assets"))
    models_assets_dir = os.path.join(assets_dir, "models")
    os.makedirs(models_assets_dir, exist_ok=True)
    
    onnx_path = os.path.join(models_assets_dir, "braille_model.onnx")
    labels_path = os.path.join(assets_dir, "labels.txt")
    
    print(f"Loading weights from {pth_path}...")
    model = BrailleCNN()
    if os.path.exists(pth_path):
        model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    else:
        print("Warning: braille_model.pth not found, exporting uninitialized BrailleCNN structure.")
    
    model.eval()
    
    # Export to ONNX
    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Successfully exported ONNX model to {onnx_path}")

    # The onnxruntime build bundled in the Flutter `onnxruntime` package
    # rejects ONNX IR version 10+ ("Unsupported model IR version: 10, max
    # supported IR version: 9"). Recent torch.onnx.export builds (the
    # dynamo-based exporter) emit IR 10 regardless of the requested
    # opset_version above -- clamp it back down. opset 12 is well within
    # IR 9's range, so this is safe. Same fix as braille_cnn/export_onnx.py.
    import onnx
    m = onnx.load(onnx_path)
    if m.ir_version > 9:
        original_ir_version = m.ir_version
        m.ir_version = 9
        onnx.save_model(m, onnx_path, save_as_external_data=False)
        print(f"Clamped ONNX IR version to 9 (was {original_ir_version})")
    
    # Generate labels.txt
    labels = [chr(ord('a') + i) for i in range(26)]
    with open(labels_path, "w") as f:
        f.write("\n".join(labels))
    print(f"Successfully exported labels to {labels_path}")

if __name__ == "__main__":
    convert()
