import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

import config
from dataset import UrbanSoundDataset
from model import ResNetAudio

CLASSES = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
    'siren', 'street_music'
]

def generate_matrix(fold_to_test, output_dir):
    print(f"\n[{'-'*10} Testing Fold {fold_to_test} {'-'*10}]")

    test_ds = UrbanSoundDataset(config.DATA_DIR, folds=[fold_to_test])
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    model = ResNetAudio().to(config.DEVICE)
    model_path = os.path.join(config.SAVE_DIR, f"resnet_best_fold_{fold_to_test}.pth")
    
    if not os.path.exists(model_path):
        print(f"CRITICAL ERROR: Could not find model weights at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(config.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate raw counts
    cm = confusion_matrix(all_labels, all_preds)
    
    # NEW: Calculate the score out of 1 (normalized by row)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # NEW: Create custom text annotations combining raw count and decimal score
    annot_labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Example format: "42\n(0.85)"
            annot_labels[i, j] = f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})"

    plt.figure(figsize=(12, 10))
    
    # NEW: Feed the normalized matrix for coloring, but use our custom text for the labels
    sns.heatmap(cm_normalized, annot=annot_labels, fmt='', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES,
                cbar_kws={'label': 'Prediction Accuracy (Score out of 1)'})
    
    plt.title(f'ResNet Confusion Matrix (Fold {fold_to_test})', fontsize=16, pad=20)
    plt.ylabel('Actual, True Sound', fontsize=14)
    plt.xlabel('What the Model Guessed', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"confusion_matrix_fold_{fold_to_test}.png")
    plt.savefig(save_path, dpi=300)
    print(f"SUCCESS! Matrix saved as: {save_path}")
    
    plt.close()

if __name__ == "__main__":

    OUTPUT_FOLDER = os.path.join(config.BASE_DIR, "confusion_matrices")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print(f"Generating matrices for all 10 folds. Saving to: {OUTPUT_FOLDER}")
    
    for i in range(1, 11):
        generate_matrix(fold_to_test=i, output_dir=OUTPUT_FOLDER)
        
    print(f"\nAll 10 matrices have been successfully generated and saved in '{OUTPUT_FOLDER}'!")