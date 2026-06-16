import os
import random

import numpy as np
import torch

SEED = 42
IMG_SIZE = 256
NUM_CLASSES = 4
CLASS_NAMES = ["Background", "Muscle", "Endometrium", "Cavity"]

# Directories
DATA_DIR = "/Users/postgres/Desktop/phd/Project/adenomyosis/png_data/images"
MASK_DIR = "/Users/postgres/Desktop/phd/Project/adenomyosis/png_data/masks"
PLOTS_DIR = "plots"
RESULTS_DIR = "results"
MODELS_DIR = "models"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Training hyperparameters
BATCH_SIZE = 4
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-3
DROPOUT = 0.4
EARLY_STOPPING_PATIENCE = 25

SPLIT = (0.7, 0.15, 0.15)

# 4 or 8slices in each fifti images, so will convert them to 2D png first, shuffle then load
# splits to 4 differnet 2D images

# resize to 256

[0.5, 2.0, 10.0, 20.0]
CLASS_WEIGHTS = [0.5, 0.5, 0.5, 0.5] //INCREASE AND DECREASE THE VALUES HERE ALSO

# Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def setup_reproducibility():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
