import os
import torch

# DIRECTORIES
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../Dataset/UrbanSound8K_processed_audio"))
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(SAVE_DIR, exist_ok=True)

# HYPERPARAMETERS
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
TEST_FOLD = 10 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")