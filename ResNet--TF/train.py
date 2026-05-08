import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
import torchaudio.transforms as T # type: ignore
from tqdm import tqdm

import config as cfg
from dataset import UrbanSoundDataset
from model import ResNetTF

def run_10_fold_cv():
    print(f"\n{'='*60}")
    print(" INITIATING 10-FOLD CROSS VALIDATION: RESNET-TF")
    print(f"{'='*60}\n")

    fold_results = []

    for test_fold in range(1, 11):
        print(f"\n{'*'*40}")
        print(f" STARTING FOLD {test_fold} / 10")
        print(f"{'*'*40}")

        train_folds = [f for f in range(1, 11) if f != test_fold]
        
        train_ds = UrbanSoundDataset(cfg.DATA_DIR, folds=train_folds)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)

        val_ds = UrbanSoundDataset(cfg.DATA_DIR, folds=[test_fold])
        val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

        model = ResNetTF(num_classes=cfg.NUM_CLASSES).to(cfg.DEVICE)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY) 
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=5)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS - 5)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])

        freq_masking = T.FrequencyMasking(freq_mask_param=cfg.FREQ_MASK_PARAM).to(cfg.DEVICE)
        time_masking = T.TimeMasking(time_mask_param=cfg.TIME_MASK_PARAM).to(cfg.DEVICE)

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(1, cfg.EPOCHS + 1):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{cfg.EPOCHS}] Train", leave=False)

            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                inputs = freq_masking(inputs)
                inputs = time_masking(inputs)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                current_lr = scheduler.get_last_lr()[0]
                train_pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}")

            train_acc = 100 * correct / total
            scheduler.step()

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch}/{cfg.EPOCHS}] Val  ", leave=False)

            with torch.no_grad():
                for inputs, labels in val_pbar:
                    inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    val_pbar.set_postfix(loss=f"{loss.item():.4f}")

            val_acc = 100 * val_correct / val_total
            val_loss = val_loss / len(val_loader)

            print(f"Epoch [{epoch}/{cfg.EPOCHS}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0 
                save_path = os.path.join(cfg.SAVE_DIR, f"resnet_tf_best_fold_{test_fold}.pth")
                torch.save(model.state_dict(), save_path)
                print(f"   -> ⭐ New best for Fold {test_fold}! Saved to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= cfg.PATIENCE:
                    print(f"\n🛑 EARLY STOPPING! Fold {test_fold} hasn't improved in {cfg.PATIENCE} epochs.")
                    break 

        print(f"\n--- Fold {test_fold} Complete! Peak Accuracy: {best_val_acc:.2f}% ---")
        fold_results.append(best_val_acc)

        del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n\n{'='*60}")
    print(" 10-FOLD CROSS VALIDATION FINAL RESULTS")
    print(f"{'='*60}")
    for idx, acc in enumerate(fold_results):
        print(f" Fold {idx+1}: {acc:.2f}%")
    
    final_avg = np.mean(fold_results)
    final_std = np.std(fold_results)
    print(f"{'-'*60}")
    print(f" OVERALL ACCURACY: {final_avg:.2f}% (+/- {final_std:.2f}%)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_10_fold_cv()