# U-Net Medical Image Segmentation

A PyTorch implementation of U-Net for multi-class medical image segmentation with support for NIfTI format and advanced data augmentation.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- 🏥 **Medical imaging focused**: Built for NIfTI format with patient-level data splitting
- 🎯 **Flexible architecture**: Support for multi-class and multi-binary segmentation modes
- 🔄 **Advanced augmentation**: Elastic deformation, rotation, brightness/contrast, flips
- ⚖️ **Class imbalance handling**: Weighted loss functions and adaptive thresholding
- 🚀 **GPU accelerated**: Automatic CUDA/MPS/CPU detection
- 📊 **Comprehensive metrics**: Dice score and IoU evaluation

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Architecture](#architecture)

---

## Installation

### Requirements

- Python 3.8 or higher
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/U-Net-Image-Segmentation.git
cd U-Net-Image-Segmentation
```

2. **Create virtual environment:**

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

3. **Install dependencies:**

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install numpy scipy scikit-image matplotlib pillow tqdm nibabel
```

For CPU-only installation:
```bash
pip install torch torchvision torchaudio
pip install numpy scipy scikit-image matplotlib pillow tqdm nibabel
```

---

## Quick Start

### 5-Minute Setup

```bash
# 1. Prepare your data (see Dataset Preparation section)
python check_pairs.py  # Verify data integrity

# 2. Configure for your dataset
# Edit config.py: NUM_CLASSES, CLASS_NAMES, data paths

# 3. Train the model
python run.py

# 4. Evaluate performance
python evaluation.py

# 5. Run inference on new data
python inference_example.py  # (see Inference section)
```

---

## Dataset Preparation

### Input Format

The model requires **paired images and segmentation masks**.

### Supported Formats

**NIfTI Format** (`.nii` or `.nii.gz`):
- 3D medical imaging volumes
- Automatically extracted to 2D slices for training

**PNG Format** (`.png`):
- 2D image slices
- Pre-extracted from 3D volumes

### Directory Structure

```
your_project/
├── png_data/
│   ├── images/
│   │   ├── patient001_slice0.png
│   │   ├── patient001_slice1.png
│   │   ├── patient002_slice0.png
│   │   └── ...
│   └── masks/
│       ├── patient001_slice0.png
│       ├── patient001_slice1.png
│       ├── patient002_slice0.png
│       └── ...
```

### Naming Convention

**CRITICAL:** Files must follow this pattern for patient-level splitting:

```
{patient_id}_slice{N}.png
```

Examples:
- ✅ `patient001_slice0.png`
- ✅ `scan_abc_slice5.png`
- ❌ `slice0_patient001.png` (wrong order)
- ❌ `patient001.png` (missing slice number)

This ensures all slices from one patient stay together during train/val/test splitting.

### Mask Labels

Masks should be **single-channel grayscale** images with integer class labels:

```
0 = Background
1 = Class 1 (e.g., muscle)
2 = Class 2 (e.g., endometrium)
3 = Class 3 (e.g., cavity)
```

### Converting NIfTI to PNG

If your data is in NIfTI format:

1. **Edit `preprocess.py`:**

```python
IMG_DIR = "path/to/nifti/images"
MASK_DIR = "path/to/nifti/masks"
OUT_IMG_DIR = "png_data/images"
OUT_MASK_DIR = "png_data/masks"
```

2. **Run preprocessing:**

```bash
python preprocess.py
```

### Verify Data Pairing

```bash
python check_pairs.py
```

This checks:
- All images have corresponding masks
- Correct naming convention
- No missing or duplicate files

---

## Configuration

Edit `config.py` to customize for your dataset:

### Essential Settings

```python
# Number of segmentation classes (including background)
NUM_CLASSES = 4  # 3 structures + background

# Class names (excluding background)
CLASS_NAMES = ["Muscle", "Endometrium", "Cavity"]

# Segmentation mode
MODE = "multibinary"  # or "multiclass"
# - multibinary: Better for class imbalance, overlapping structures
# - multiclass: Traditional softmax, mutually exclusive classes

# Data paths
DATA_DIR = "png_data/images"
MASK_DIR = "png_data/masks"

# Training settings
BATCH_SIZE = 4
EPOCHS = 300
LR = 1e-4
IMG_SIZE = 256  # Images resized to 256x256
```

### Advanced Settings

```python
# Class weights (handle imbalance)
CLASS_WEIGHTS = [0.5, 2.0, 10.0, 20.0]  # [bg, class1, class2, class3]
# Higher weight = more importance

# Multi-binary settings
BINARY_CLASS_WEIGHTS = [2.0, 10.0, 15.0]  # Per-class positive weights
THRESHOLDS = [0.5, 0.4, 0.3]  # Confidence thresholds per class

# Data split (train/val/test)
SPLIT = (0.7, 0.15, 0.15)

# Regularization
DROPOUT = 0.5
WEIGHT_DECAY = 1e-3
```

---

## Training

### Start Training

```bash
python run.py
```

This will:
1. Load and split the dataset (patient-level)
2. Apply data augmentation during training
3. Train for specified epochs
4. Save best model to `models/best_model.pth`
5. Generate training curves and sample predictions

### Monitor Progress

Watch the console output:

```
Epoch 50/300 | Train Loss: 0.4523 | Val Loss: 0.4012
  Dice: Muscle: 0.812 | Endometrium: 0.743 | Cavity: 0.689
  Mean Dice (excl. bg): 0.748

Best mean Dice: 0.748
```

**Key metrics:**
- **Loss:** Should decrease (lower is better)
- **Dice Score:** Should increase (max = 1.0, good > 0.7)
- **Mean Dice:** Average across all non-background classes

### Output Files

```
models/
└── best_model.pth         # Best model weights

plots/
├── training_curves.png    # Loss and Dice over time
└── segmentation_results.png  # Sample predictions
```

### Training Tips

**GPU out of memory?**
```python
# In config.py
BATCH_SIZE = 1
IMG_SIZE = 128
```

**Training too slow?**
```python
EPOCHS = 100  # Reduce epochs for testing
```

**Poor results?**
1. Train longer (more epochs)
2. Increase class weights for small classes
3. Check data quality (`python view_ground_truth.py 10`)
4. Ensure correct mask labels (0, 1, 2, 3...)

---

## Evaluation

### Evaluate on Test Set

```bash
python evaluation.py
```

**Output:**
```
=== Test Set Evaluation ===
Class         | Dice  | IoU
--------------------------------
Background    | 0.982 | 0.964
Muscle        | 0.856 | 0.748
Endometrium   | 0.792 | 0.655
Cavity        | 0.734 | 0.580
--------------------------------
Mean (excl bg)| 0.794 | 0.661
```

**Interpreting Results:**
- **Dice > 0.9:** Excellent
- **Dice 0.7-0.9:** Good
- **Dice 0.5-0.7:** Moderate
- **Dice < 0.5:** Poor (needs improvement)

---

## Inference

### Create Inference Script

```python
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

import config
import model as models

# Load trained model
net = models.new_model()
net.load_state_dict(torch.load("models/best_model.pth", map_location=config.DEVICE))
net.eval()

# Prepare image transform
transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
])

# Load image
img_path = "path/to/test/image.png"
img = Image.open(img_path).convert('L')  # Grayscale
img_tensor = transform(img).unsqueeze(0)

# Run inference
with torch.no_grad():
    output = net(img_tensor.to(config.DEVICE))
    
    if config.MODE == "multibinary":
        # Multi-binary mode
        probs = torch.sigmoid(output[0]).cpu().numpy()
        pred_mask = np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.uint8)
        for i, thresh in enumerate(config.THRESHOLDS):
            pred_mask[probs[i] > thresh] = i + 1
    else:
        # Multi-class mode
        pred_mask = output.argmax(dim=1)[0].cpu().numpy()

# Visualize
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(pred_mask, cmap='tab10')
plt.title("Prediction")
plt.axis('off')
plt.tight_layout()
plt.savefig("prediction.png")
plt.show()
```

---

## Architecture

### U-Net Overview

```
Input (1 channel, 256×256)
    ↓
Encoder (6 levels: 64→128→256→512→512→1024)
    ↓
Bottleneck (1024 channels)
    ↓
Decoder (6 levels with skip connections)
    ↓
Output (NUM_CLASSES channels, 256×256)
```

### Key Features

- **Deep architecture:** 6 encoder/decoder levels
- **Skip connections:** Preserve spatial information
- **Dropout:** Progressive regularization (0.125× to 1×)
- **Flexible output:** Multi-class (softmax) or multi-binary (sigmoid) heads

### Model Specifications

```python
Parameters: ~31M (default configuration)
Input size: 256×256×1 (grayscale)
Output size: 256×256×NUM_CLASSES
Memory: ~4GB VRAM (batch_size=4)
```

---

## Troubleshooting

### Common Issues

#### "CUDA out of memory"

**Solution:**
```python
# config.py
BATCH_SIZE = 1
IMG_SIZE = 128
```

Or use CPU:
```python
DEVICE = torch.device("cpu")
```

#### "No module named 'torch'"

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install PyTorch
pip install torch torchvision
```

#### Poor Segmentation Results

**Checklist:**
- [ ] Masks have correct labels (0, 1, 2, 3...)?
- [ ] Enough training data (>50 images minimum)?
- [ ] Images and masks properly paired?
- [ ] Adjusted class weights for imbalance?
- [ ] Trained long enough (loss still decreasing)?

**Debug steps:**
```bash
# 1. Visualize ground truth
python view_ground_truth.py 10

# 2. Check data pairing
python check_pairs.py

# 3. Monitor training curves
# Check plots/training_curves.png for overfitting
```

#### Model Not Improving

**Solutions:**

1. **Adjust learning rate:**
```python
LR = 1e-5  # Lower if loss oscillates
LR = 1e-3  # Higher if loss barely moves
```

2. **Increase class weights for small classes:**
```python
CLASS_WEIGHTS = [0.5, 5.0, 20.0, 30.0]
```

3. **Add more augmentation** (edit `dataset.py`)

4. **Train longer:**
```python
EPOCHS = 500
```

---

## Project Structure

```
U-Net-Image-Segmentation/
├── config.py              # Configuration settings
├── model.py               # U-Net architecture
├── dataset.py             # Data loader and augmentation
├── training.py            # Training logic and losses
├── evaluation.py          # Metrics calculation
├── run.py                 # Main training script
├── preprocess.py          # NIfTI to PNG converter
├── check_pairs.py         # Data integrity checker
├── view_ground_truth.py   # Visualization tool
├── search_patient.py      # Search utility
├── combinedFolder.py      # Data merging utility
├── combinedMasks.py       # Mask merging utility
├── add_new_data.py        # Dataset expansion utility
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Data Augmentation

The following augmentations are applied during training:

- **Geometric:**
  - Horizontal flip (50%)
  - Vertical flip (50%)
  - Random rotation ±15° (40%)
  
- **Intensity:**
  - Brightness adjustment ±30% (50%)
  - Contrast adjustment ±30% (50%)
  
- **Elastic deformation:** (30%)
  - Simulates tissue deformation in medical images

All augmentations maintain image-mask correspondence.

---

## Advanced Usage

### Transfer Learning

Load pretrained weights:

```python
net = models.new_model()
net.load_state_dict(torch.load("pretrained_model.pth"), strict=False)

# Optionally freeze encoder
for param in net.backbone.parameters():
    param.requires_grad = False
```

### Custom Loss Functions

Edit `training.py` to add custom losses:

```python
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
    
    def forward(self, pred, target):
        return self.dice(pred, target) + 0.5 * self.focal(pred, target)
```

### Multi-GPU Training

Enable DataParallel:

`

## Acknowledgments

- U-Net architecture based on [Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597)
- Built with PyTorch and NiBabel for medical imaging support

---

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com]

---

**Happy segmenting! 🎯**
