import os
import sys
import torch
import torch.nn as nn
from dataset_loader import KaggleBrailleDataset
from model import BrailleCNN, predict_image
from audio_engine import speak
from torch.utils.data import DataLoader

def verify():
    print("=== BrailleLens Verification Script ===\n")
    results = {}
    
    dataset_path = r"d:\BrailleLens\archive\Braille Dataset\Braille Dataset"
    model_save_path = os.path.join(os.path.dirname(__file__), "braille_model.pth")
    
    # 1. Dataset loading
    try:
        dataset = KaggleBrailleDataset(root_dir=dataset_path)
        if len(dataset) > 0:
            sample_img, sample_label = dataset[0]
            if sample_img.shape == (1, 28, 28) and isinstance(sample_label, int):
                results['Dataset Loading'] = "PASS"
            else:
                results['Dataset Loading'] = f"FAIL (Bad shape/label: {sample_img.shape}, {sample_label})"
        else:
             results['Dataset Loading'] = "FAIL (No images)"
    except Exception as e:
        results['Dataset Loading'] = f"FAIL ({e})"

    # 2. Model instantiation
    try:
        model = BrailleCNN()
        dummy_tensor = torch.randn(1, 1, 28, 28)
        out = model(dummy_tensor)
        if out.shape == (1, 26):
            results['Model Instantiation'] = "PASS"
        else:
            results['Model Instantiation'] = f"FAIL (Bad output shape: {out.shape})"
    except Exception as e:
         results['Model Instantiation'] = f"FAIL ({e})"

    # 3. Quick 1-epoch training (on a tiny subset)
    try:
        if results.get('Dataset Loading') == "PASS":
            device = torch.device("cpu")
            tiny_dataset = torch.utils.data.Subset(dataset, range(10))
            loader = DataLoader(tiny_dataset, batch_size=2)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for images, labels in loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                break # Just 1 batch
            
            torch.save(model.state_dict(), model_save_path)
            results['Quick Training'] = "PASS"
        else:
            results['Quick Training'] = "SKIP (Dataset failed)"
    except Exception as e:
        results['Quick Training'] = f"FAIL ({e})"

    # 4. Single-image inference
    try:
        if results.get('Quick Training') == "PASS":
            sample_path = dataset.samples[0][0]
            char, conf = predict_image(sample_path, model_save_path)
            if 'a' <= char <= 'z' and 0 <= conf <= 1:
                results['Inference'] = "PASS"
            else:
                results['Inference'] = f"FAIL (Bad output: {char}, {conf})"
        else:
            results['Inference'] = "SKIP (Training failed)"
    except Exception as e:
        results['Inference'] = f"FAIL ({e})"

    # 5. TTS smoke test
    try:
        speak("Pipeline verification complete")
        results['TTS Audio'] = "PASS"
    except Exception as e:
        results['TTS Audio'] = f"FAIL ({e})"

    # Summary
    print("\n--- Summary ---")
    all_pass = True
    for k, v in results.items():
        print(f"{k:20}: {v}")
        if "FAIL" in v:
            all_pass = False
            
    if all_pass:
        print("\nAll checks passed successfully!")
        sys.exit(0)
    else:
        print("\nSome checks failed.")
        sys.exit(1)

if __name__ == "__main__":
    verify()
