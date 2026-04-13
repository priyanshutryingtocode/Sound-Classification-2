import os
import torch
import torch.nn as nn
import torch.optim as optim
import config
from dataset import get_dataloaders
from model import ResNetAudio

def train_model():
    print(f"Using device: {config.DEVICE}")
    print("Loading data...")
    
    # 1. Setup Data
    train_loader, test_loader, train_folds = get_dataloaders()
    print(f"Starting training on Folds {train_folds} | Testing on Fold {config.TEST_FOLD}")
    print("-" * 50)

    # 2. Setup Model, Loss, and Optimizer
    model = ResNetAudio().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_acc = 0.0

    # 3. Training Loop
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        avg_loss = running_loss / len(train_loader)

        # 4. Validation Loop
        model.eval()
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        scheduler.step(val_acc)

        print(f"Epoch [{epoch+1:2d}/{config.EPOCHS}] | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # 5. Save the best version
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(config.SAVE_DIR, f"resnet_best_fold_{config.TEST_FOLD}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"   -> New best model saved! ({best_acc:.2f}%)")

    print("-" * 50)
    print(f"Training complete. Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    # Safety check before starting
    if not os.path.exists(config.DATA_DIR):
        print(f"CRITICAL ERROR: Cannot find the dataset at {config.DATA_DIR}")
    else:
        train_model()