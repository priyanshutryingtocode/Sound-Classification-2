import os
import torch

# Paths 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../Dataset/UrbanSound8K_processed_audio"))
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 24  
EPOCHS = 50      
PATIENCE = 20    
NUM_CLASSES = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Augmentation Parameters
FREQ_MASK_PARAM = 8
TIME_MASK_PARAM = 10