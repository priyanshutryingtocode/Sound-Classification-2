import os
import torch
from torch.utils.data import Dataset, DataLoader
import config

class UrbanSoundDataset(Dataset):
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
            spectrogram = torch.load(self.file_paths[index], weights_only=True)
            return spectrogram, self.labels[index]
        except Exception:
            return torch.zeros(1, 64, 172), self.labels[index]

def get_dataloaders():

    train_folds = [f for f in range(1, 11) if f != config.TEST_FOLD]
    
    train_ds = UrbanSoundDataset(config.DATA_DIR, folds=train_folds)
    test_ds = UrbanSoundDataset(config.DATA_DIR, folds=[config.TEST_FOLD])

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, train_folds