import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class BrailleCNN(nn.Module):
    def __init__(self):
        super(BrailleCNN, self).__init__()
        # Input: (1, 28, 28)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # (32, 14, 14)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # (64, 7, 7)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2) # (128, 3, 3)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 26)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_model(model_path, device='cpu'):
    model = BrailleCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    predicted_idx = predicted.item()
    predicted_char = chr(predicted_idx + ord('a'))
    return predicted_char, confidence.item()
