import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class KaggleBrailleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        # Find all jpg files in root_dir and subdirectories
        self.image_paths = []
        for ext in ('*.jpg', '*.JPG', '**/*.jpg', '**/*.JPG'):
            self.image_paths.extend(glob.glob(os.path.join(root_dir, ext), recursive=True))
            
        # Remove duplicates if any
        self.image_paths = list(set(self.image_paths))
            
        # Filter and parse labels
        self.samples = []
        for path in self.image_paths:
            filename = os.path.basename(path)
            label_char = filename[0].lower()
            if 'a' <= label_char <= 'z':
                label_idx = ord(label_char) - ord('a')
                self.samples.append((path, label_idx))
                
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
