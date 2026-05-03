import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import your custom modules from the vit folder
import config
from dataset import UrbanSoundDataset
from model import ViTAudio

def train_single_fold():
    # ---------------------------------------------------------
    # 1. SETUP
    # ---------------------------------------------------------
    TEST_FOLD = 1
    EPOCHS = 30
    
    print(f"\n{'='*50}")
    print(f" VISION TRANSFORMER TEST RUN: FOLD {TEST_FOLD}")
    print(f" Total Epochs: {EPOCHS} | Device: {config.DEVICE}")
    print(f"{'='*50}\n")

    # The training data is every fold EXCEPT fold 1
    train_folds = [f for f in range(1, 11) if f != TEST_FOLD]
    
    print("Loading Training Data...")
    train_ds = UrbanSoundDataset(config.DATA_DIR, folds=train_folds)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)

    print("Loading Validation Data...")
    val_ds = UrbanSoundDataset(config.DATA_DIR, folds=[TEST_FOLD])
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # ---------------------------------------------------------
    # 2. INITIALIZE MODEL
    # ---------------------------------------------------------
    print("Downloading/Loading ViT Weights...")
    model = ViTAudio(num_classes=10).to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    # Using a slightly lower learning rate is standard for pre-trained Transformers
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 

    best_val_acc = 0.0

    # ---------------------------------------------------------
    # 3. TRAINING LOOP
    # ---------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        # --- TRAIN PHASE ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)

        # --- PROGRESS PRINTING ---
        print(f"Epoch [{epoch}/{EPOCHS}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}")

        # --- SAVE BEST MODEL ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(config.SAVE_DIR, f"vit_best_fold_{TEST_FOLD}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"   -> ⭐ New best accuracy! Model saved to {save_path}")

    print(f"\n{'='*50}")
    print(f" TEST RUN COMPLETE! Peak Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    train_single_fold()