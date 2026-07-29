import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset_loader import KaggleBrailleDataset
from model import BrailleCNN

def main():
    parser = argparse.ArgumentParser(description="Train BrailleCNN Model")
    parser.add_argument("--dataset_path", type=str, default=r"d:\BrailleLens\archive\Braille Dataset\Braille Dataset", help="Path to Kaggle dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = KaggleBrailleDataset(root_dir=args.dataset_path)
    if len(dataset) == 0:
        print("Error: No images found in the dataset path!")
        return

    # Train / Val Split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Dataset split: {train_size} train, {val_size} val")

    model = BrailleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    model_save_path = os.path.join(os.path.dirname(__file__), "braille_model.pth")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / train_size
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = correct / total
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model with accuracy: {val_acc:.4f}")
            
        # In case we have a very small dataset or only 1 epoch, make sure we save something
        if epoch == args.epochs - 1 and not os.path.exists(model_save_path):
            torch.save(model.state_dict(), model_save_path)

    print(f"Training complete. Best model saved to {model_save_path}")

if __name__ == "__main__":
    main()
