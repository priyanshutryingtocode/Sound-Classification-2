import os
import matplotlib
matplotlib.use('Agg') 

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

import config as cfg
from dataset import UrbanSoundDataset
from model import ResNetTF

def plot_master_confusion_matrix():
    classes = [
        'Air Conditioner', 'Car Horn', 'Children Playing', 'Dog Bark',
        'Drilling', 'Engine Idling', 'Gun Shot', 'Jackhammer',
        'Siren', 'Street Music'
    ]

    print(f"\n{'='*60}")
    print(" GENERATING MASTER CONFUSION MATRIX (10-FOLD)")
    print(f"{'='*60}\n")
    print("Aggregating predictions across all 10 folds... This will take a moment.")

    all_preds = []
    all_labels = []

    for fold in range(1, 11):
        val_ds = UrbanSoundDataset(cfg.DATA_DIR, folds=[fold])
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        model = ResNetTF(num_classes=cfg.NUM_CLASSES).to(cfg.DEVICE)
        weights_path = os.path.join(cfg.SAVE_DIR, f"resnet_tf_best_fold_{fold}.pth")
        
        try:
            model.load_state_dict(torch.load(weights_path, map_location=cfg.DEVICE, weights_only=True))
            print(f"Loaded weights for Fold {fold}...")
        except FileNotFoundError:
            print(f"Warning: Could not find {weights_path}. Skipping...")
            continue

        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(cfg.DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

    print(f"\nData collected! Total predictions loaded: {len(all_preds)}")
    if len(all_preds) == 0:
        print("🛑 ERROR: No predictions were generated! The script did not load your .pth files.")
        print(f"Please check if your weights actually exist at: {cfg.SAVE_DIR}")
        return

    print("Generating heatmap image...")

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.set_theme(style="white") 

    ax = sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", 
                     xticklabels=classes, yticklabels=classes, 
                     cbar_kws={'label': 'Prediction Accuracy'})

    plt.title("Master Confusion Matrix: ResNet-TF (10-Fold CV)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("True Sound Class", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Sound Class", fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()

    save_path = os.path.join(cfg.SAVE_DIR, "master_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 
    
    print(f"SUCCESS! Heatmap successfully generated and saved to: {save_path}")

if __name__ == "__main__":
    plot_master_confusion_matrix()