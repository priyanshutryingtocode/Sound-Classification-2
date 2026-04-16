import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm 
import config
from dataset import get_dataloaders
from model import ResNetAudio

def train_model():
    print(f"Using device: {config.DEVICE}")
    
    train_loader, test_loader, train_folds = get_dataloaders()
    print(f"Starting training on Folds {train_folds} | Testing on Fold {config.TEST_FOLD}")

    model = ResNetAudio().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_acc = 0.0

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1:2d}/{config.EPOCHS}] Train", leave=False)
        
        for inputs, labels in train_loop:
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

            train_loop.set_postfix(loss=loss.item())

        train_acc = 100. * correct / total
        avg_loss = running_loss / len(train_loader)

        model.eval()
        val_correct, val_total = 0, 0
        
        val_loop = tqdm(test_loader, desc=f"Epoch [{epoch+1:2d}/{config.EPOCHS}] Valid", leave=False)
        
        with torch.no_grad():
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        scheduler.step(val_acc)

        print(f"Epoch [{epoch+1:2d}/{config.EPOCHS}] | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(config.SAVE_DIR, f"resnet_best_fold_{config.TEST_FOLD}_iter2.pth")
            torch.save(model.state_dict(), save_path)
            print(f"   -> New best model saved! ({best_acc:.2f}%)")

    print(f"Training complete. Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    if not os.path.exists(config.DATA_DIR):
        print(f"CRITICAL ERROR: Cannot find the dataset at {config.DATA_DIR}")
    else:
        train_model()