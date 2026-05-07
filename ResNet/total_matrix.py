import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
matplotlib.use('Agg')

import config
from dataset import UrbanSoundDataset
from model import ResNetAudio

CLASSES = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
    'siren', 'street_music'
]

def generate_cumulative_matrix():

    global_preds = []
    global_labels = []

    for fold_num in range(1, 11):
        print(f"Listening to Fold {fold_num}/10...")

        test_ds = UrbanSoundDataset(config.DATA_DIR, folds=[fold_num])
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

        model = ResNetAudio().to(config.DEVICE)
        model_path = os.path.join(config.SAVE_DIR, f"resnet_best_fold_{fold_num}.pth")
            
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE, weights_only=True))
        model.eval()

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(config.DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                global_preds.extend(preds.cpu().numpy())
                global_labels.extend(labels.numpy())

    cm = confusion_matrix(global_labels, global_preds)

    total_correct = np.trace(cm)
    total_guesses = np.sum(cm)
    overall_accuracy = (total_correct / total_guesses) * 100

    # Calculate the score out of 1 (normalized by row)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(14, 12))
    
    # We pass the normalized matrix directly to Seaborn and tell it to format as a 2-decimal float ('.2f')
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES,
                cbar_kws={'label': 'Prediction Accuracy (Score out of 1)'})
    
    plt.title(f'ResNet Confusion Matrix\nOverall Accuracy: {overall_accuracy:.2f}%', fontsize=18, pad=20)
    plt.ylabel('Actual Sound', fontsize=14)
    plt.xlabel('Model Prediction', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    save_path = os.path.join(config.BASE_DIR, "cumulative_resnet_matrix.png")
    plt.savefig(save_path, dpi=300)
    print(f"\nSUCCESS! Master Cumulative Matrix saved as: {save_path}")

if __name__ == "__main__":
    generate_cumulative_matrix()