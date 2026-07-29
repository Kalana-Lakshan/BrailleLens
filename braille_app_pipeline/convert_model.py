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
    
    # Generate labels.txt
    labels = [chr(ord('a') + i) for i in range(26)]
    with open(labels_path, "w") as f:
        f.write("\n".join(labels))
    print(f"Successfully exported labels to {labels_path}")

if __name__ == "__main__":
    convert()
