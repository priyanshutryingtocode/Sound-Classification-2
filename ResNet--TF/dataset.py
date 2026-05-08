import os
import torch
from torch.utils.data import Dataset

class UrbanSoundDataset(Dataset):
    """
    Loads pre-processed Mel-Spectrogram tensors (.pt files) for UrbanSound8K.
    """
    def __init__(self, data_dir, folds):
        self.file_paths = []
        self.labels = []
        for fold_num in folds:
            fold_path = os.path.join(data_dir, f"fold{fold_num}")
            if os.path.exists(fold_path):
                for file_name in os.listdir(fold_path):
                    if file_name.endswith('.pt'):
                        self.file_paths.append(os.path.join(fold_path, file_name))
                        self.labels.append(int(file_name.split('-')[1]))

    def __len__(self): 
        return len(self.file_paths)
        
    def __getitem__(self, index):
        try: 
            return torch.load(self.file_paths[index], weights_only=True), self.labels[index]
        except Exception: 
            return torch.zeros(1, 64, 172), self.labels[index]