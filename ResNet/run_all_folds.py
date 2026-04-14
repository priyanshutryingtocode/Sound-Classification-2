import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import config
from dataset import UrbanSoundDataset
from model import ResNetAudio

def run_cross_validation():
    print(f"\n{'='*60}")
    print(" STARTING AUTOMATED 10-FOLD CROSS-VALIDATION")
    print(f" Device: {config.DEVICE} | Epochs per Fold: {config.EPOCHS}")
    print(f"{'='*60}\n")

    # To store the absolute best score from each fold
    fold_accuracies = []

    # Loop exactly 10 times, changing the test fold each time
    for test_fold in range(1, 11):
        print(f"\n[{'-'*15} FOLD {test_fold}/10 {'-'*15}]")
        
        # Determine which 9 folds are for training
        train_folds = [f for f in range(1, 11) if f != test_fold]

        # 1. Load Data for this specific rotation
        train_ds = UrbanSoundDataset(config.DATA_DIR, folds=train_folds)
        test_ds = UrbanSoundDataset(config.DATA_DIR, folds=[test_fold])

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

        # 2. CRITICAL: Initialize a completely fresh, blank ResNet brain for every fold
        model = ResNetAudio().to(config.DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        best_fold_acc = 0.0

        # 3. The Training Loop for this specific fold
        for epoch in range(config.EPOCHS):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            
            # Using leave=False so the progress bars delete themselves to keep the terminal clean
            train_loop = tqdm(train_loader, desc=f"  Epoch [{epoch+1:2d}/{config.EPOCHS}] Train", leave=False)
            
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

            # Validation Phase
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

            # Save the best model for this specific fold
            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                save_path = os.path.join(config.SAVE_DIR, f"resnet_best_fold_{test_fold}.pth")
                torch.save(model.state_dict(), save_path)
                
            # Optional: Print epoch stats (Comment this out if you only want to see the final fold results)
            print(f"  Epoch [{epoch+1:2d}/{config.EPOCHS}] | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% (Best: {best_fold_acc:.1f}%)")

        # 4. Record the final best accuracy for this fold
        fold_accuracies.append(best_fold_acc)
        print(f"--> Fold {test_fold} Completed! Peak Accuracy: {best_fold_acc:.2f}%")

    # 5. Calculate and print the grand totals
    print(f"\n{'-'*60}")
    print(" FINAL 10-FOLD CROSS-VALIDATION METRICS\n")

    
    for i, acc in enumerate(fold_accuracies, 1):
        print(f" Fold {i:2d}: {acc:.2f}%")
        
    avg_acc = np.mean(fold_accuracies)
    std_dev = np.std(fold_accuracies)

    print("-" * 60)
    print(f" AVERAGE ACCURACY: {avg_acc:.2f}% (+/- {std_dev:.2f}%)\n")
    print(f"{'-'*60}\n")

if __name__ == "__main__":
    if not os.path.exists(config.DATA_DIR):
        print(f"CRITICAL ERROR: Cannot find the dataset at {config.DATA_DIR}")
    else:
        run_cross_validation()