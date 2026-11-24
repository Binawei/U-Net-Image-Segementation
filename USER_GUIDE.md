# U-Net Image Segmentation - User Guide

A complete guide for using this U-Net model on your own medical imaging dataset.

## Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Dataset Preparation](#dataset-preparation)
5. [Configuration](#configuration)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Inference](#inference)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This U-Net implementation performs multi-class semantic segmentation on medical images. It was originally developed for segmenting anatomical structures (muscle, endometrium, cavity) in transvaginal ultrasound images.

**Key Features:**
- Deep U-Net architecture (6 encoder levels)
- Two modes: multi-class or multi-binary segmentation
- Patient-level train/validation/test splitting (prevents data leakage)
- Data augmentation (flips, rotation, brightness, elastic deformation)
- Automatic GPU/CPU detection
- Support for NIfTI medical imaging format

---

## Requirements

### Hardware
- **Minimum:** 8GB RAM, CPU
- **Recommended:** 16GB+ RAM, NVIDIA GPU with 4GB+ VRAM
- **Storage:** ~5-10GB for model training

### Software
- Python 3.8 or higher
- CUDA 11.8+ (if using GPU)

---

## Installation

### Step 1: Clone or Download the Code

```bash
# If you received a zip file, extract it
# Or clone from repository
git clone <repository-url>
cd U-Net-Image-Segementation
```

### Step 2: Create Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (GPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy scipy scikit-image matplotlib pillow tqdm nibabel
```

**For CPU-only:**
```bash
pip install torch torchvision torchaudio
```

### Step 4: Verify Installation

```bash
python test_device.py
```

You should see confirmation that PyTorch is installed and GPU detected (if available).

---

## Dataset Preparation

### Input Format

The model expects **paired images and masks**:
- **Images:** Grayscale medical images
- **Masks:** Segmentation labels with integer class values

### Supported Formats

**Option A: NIfTI Format (.nii or .nii.gz)**
- 3D medical imaging format
- Each file contains multiple 2D slices
- Use this if your data is already in NIfTI format

**Option B: PNG Format (.png)**
- 2D image slices
- Already extracted from 3D volumes
- Faster for training

### Directory Structure

```
your_project/
├── data/
│   ├── nifti/                    # Option A: NIfTI files
│   │   ├── images/
│   │   │   ├── patient001.nii.gz
│   │   │   ├── patient002.nii.gz
│   │   │   └── ...
│   │   └── masks/
│   │       ├── patient001.nii.gz
│   │       ├── patient002.nii.gz
│   │       └── ...
│   │
│   └── png_data/                 # Option B: PNG slices
│       ├── images/
│       │   ├── patient001_slice0.png
│       │   ├── patient001_slice1.png
│       │   └── ...
│       └── masks/
│           ├── patient001_slice0.png
│           ├── patient001_slice1.png
│           └── ...
```

### Naming Convention

**IMPORTANT:** Files must follow this naming pattern:
```
{patient_id}_slice{N}.png
```

Examples:
- `patient001_slice0.png`
- `scan_abc_slice5.png`
- `case123_slice2.png`

This ensures the patient-level splitting works correctly (all slices from one patient stay together in train/val/test).

### Mask Label Format

Masks should be **single-channel grayscale** images with integer values representing classes:

```
0 = Background
1 = Class 1 (e.g., muscle)
2 = Class 2 (e.g., endometrium)
3 = Class 3 (e.g., cavity)
...
```

**Example:** For 3 classes + background, pixel values should be: 0, 1, 2, or 3.

---

## Configuration

### Step 1: Convert NIfTI to PNG (If Needed)

If your data is in NIfTI format, convert it first:

1. **Edit `preprocess.py`:**

```python
# Update these paths to match your data
IMG_DIR = "path/to/your/nifti/images"
MASK_DIR = "path/to/your/nifti/masks"
OUT_IMG_DIR = "png_data/images"
OUT_MASK_DIR = "png_data/masks"
```

2. **Run preprocessing:**

```bash
python preprocess.py
```

This will extract 2D slices and save them as PNG files.

### Step 2: Configure Training Settings

**Edit `config.py`:**

```python
# 1. Set number of classes (including background)
NUM_CLASSES = 4  # Change this: 4 = 3 classes + background

# 2. Update class names
CLASS_NAMES = ["Muscle", "Endometrium", "Cavity"]  # Your class names (excluding background)

# 3. Choose segmentation mode
MODE = "multibinary"  # or "multiclass"
# - multibinary: Better for class imbalance, independent predictions per class
# - multiclass: Traditional softmax, mutually exclusive classes

# 4. Update binary class names (if using multibinary mode)
BINARY_CLASS_NAMES = ["Muscle", "Endometrium", "Cavity"]

# 5. Set data directories
DATA_DIR = "png_data/images"
MASK_DIR = "png_data/masks"

# 6. Training hyperparameters
BATCH_SIZE = 4          # Reduce if GPU memory issues
EPOCHS = 300            # Number of training epochs
LR = 1e-4              # Learning rate
IMG_SIZE = 256          # Input image size (256x256)

# 7. Class weights (for handling class imbalance)
# Higher weights = more importance to that class
CLASS_WEIGHTS = [0.5, 2.0, 10.0, 20.0]  # [background, class1, class2, class3]
BINARY_CLASS_WEIGHTS = [2.0, 10.0, 15.0]  # For multibinary mode

# 8. Thresholds (for multibinary mode)
THRESHOLDS = [0.5, 0.4, 0.3]  # Confidence thresholds per class
```

### Step 3: Verify Data Pairing

Check that images and masks are properly paired:

```bash
python check_pairs.py
```

This will verify:
- All images have corresponding masks
- No missing or extra files
- Correct naming convention

---

## Training

### Step 1: Start Training

```bash
python run.py
```

This will:
1. Load and split the dataset (70% train, 15% val, 15% test)
2. Train the model for the specified number of epochs
3. Save the best model to `models/best_model.pth`
4. Generate training curves in `plots/training_curves.png`
5. Create visualization samples in `plots/segmentation_results.png`

### Step 2: Monitor Training

Watch the console output:

```
Epoch 1/300 | Train Loss: 0.8234 | Val Loss: 0.7156
  Dice: Muscle: 0.654 | Endometrium: 0.432 | Cavity: 0.321
  Mean Dice (excl. bg): 0.469

Epoch 10/300 | Train Loss: 0.4523 | Val Loss: 0.4012
  Dice: Muscle: 0.812 | Endometrium: 0.743 | Cavity: 0.689
  Mean Dice (excl. bg): 0.748

Best mean Dice: 0.748
```

**What to look for:**
- **Loss decreasing:** Both train and val loss should go down
- **Dice score increasing:** Higher is better (max = 1.0)
- **Overfitting check:** If train loss << val loss, you're overfitting

### Step 3: Check Output

After training completes, check:
- `models/best_model.pth` - Trained model weights
- `plots/training_curves.png` - Loss and Dice curves
- `plots/segmentation_results.png` - Sample predictions

### Training Tips

**If training is too slow:**
```python
# In config.py
BATCH_SIZE = 2  # Reduce batch size
EPOCHS = 100    # Fewer epochs for testing
```

**If GPU runs out of memory:**
```python
BATCH_SIZE = 1  # Minimum batch size
IMG_SIZE = 128  # Smaller image size
```

**If results are poor:**
1. Train longer (more epochs)
2. Adjust class weights (increase weight for small/important classes)
3. Add more data augmentation
4. Check data quality (labels correct?)

---

## Evaluation

### Evaluate on Test Set

```bash
python evaluation.py
```

This will:
- Load the best model
- Run predictions on the test set
- Calculate metrics: Dice score, IoU
- Print per-class and average performance

**Example output:**
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

---

## Inference

### Inference on New Data

#### Option 1: Inference on PNG Images

Create a custom inference script based on `infer_milan.py`:

```python
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

import config
import model as models

# Load model
net = models.new_model()
net.load_state_dict(torch.load("models/best_model.pth", map_location=config.DEVICE))
net.eval()

# Prepare transform
transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
])

# Load and predict on a single image
img_path = "path/to/your/test/image.png"
img = Image.open(img_path).convert('L')  # Grayscale
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    output = net(img_tensor.to(config.DEVICE))
    
    if config.MODE == "multibinary":
        # Multi-binary: apply sigmoid and thresholds
        probs = torch.sigmoid(output[0]).cpu().numpy()  # (3, H, W)
        pred_mask = np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.uint8)
        for i, thresh in enumerate(config.THRESHOLDS):
            pred_mask[probs[i] > thresh] = i + 1
    else:
        # Multi-class: argmax
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
plt.show()
```

#### Option 2: Inference on NIfTI Volumes

Use the Milan inference script as a template - modify paths in `infer_milan.py`:

```python
# In infer_milan.py, update paths:
MILAN_IMAGES_DIR = "path/to/your/nifti/images"
RESULTS_DIR = "path/to/output/results"
```

Then run:
```bash
python infer_milan.py
```

This will:
- Process all NIfTI files
- Generate predictions for each slice
- Save visualization PNGs
- Save raw masks for NIfTI conversion

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution:**
```python
# In config.py
BATCH_SIZE = 1  # Reduce to minimum
IMG_SIZE = 128  # Use smaller images
```

Or use CPU:
```python
# In config.py
DEVICE = torch.device("cpu")
```

### Issue: "No module named 'torch'"

**Solution:**
```bash
# Activate virtual environment first
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Then install PyTorch
pip install torch torchvision
```

### Issue: Poor segmentation results

**Checklist:**
1. ✓ Are masks correctly labeled? (0, 1, 2, 3...)
2. ✓ Do you have enough training data? (>50 images minimum)
3. ✓ Are images and masks properly paired?
4. ✓ Did you adjust class weights for imbalance?
5. ✓ Did you train long enough? (check if loss still decreasing)

### Issue: Model doesn't improve

**Solutions:**
1. **Check data quality:** Visualize a few training samples
   ```bash
   python view_ground_truth.py 10
   ```

2. **Adjust learning rate:**
   ```python
   LR = 1e-5  # Lower learning rate
   ```

3. **Increase class weights for small classes:**
   ```python
   CLASS_WEIGHTS = [0.5, 5.0, 20.0, 30.0]  # Emphasize small structures
   ```

4. **Train longer:**
   ```python
   EPOCHS = 500
   ```

### Issue: Images and masks don't match

**Solution:**
Check naming convention - must follow `{patient_id}_sliceN.png` pattern:
```bash
python check_pairs.py
```

If files aren't matched, rename them consistently.

---

## Advanced Configuration

### Custom Data Augmentation

Edit `dataset.py` to add/remove augmentations:

```python
def augment(img, mask):
    # Existing augmentations...
    
    # Add custom augmentation
    if random.random() < 0.3:
        # Your custom transformation
        pass
    
    return img, mask
```

### Model Architecture

The U-Net depth is configurable in `model.py`:

```python
# Current: 6 encoder levels (64→128→256→512→512→1024)
# To reduce model size, modify the backbone
```

### Transfer Learning

To use a pretrained model:

```python
# Load pretrained weights
net = models.new_model()
net.load_state_dict(torch.load("path/to/pretrained_model.pth"), strict=False)

# Optionally freeze encoder
for param in net.backbone.parameters():
    param.requires_grad = False
```

---

## File Structure Reference

```
U-Net-Image-Segementation/
├── config.py              # Configuration (EDIT THIS)
├── model.py               # U-Net architecture
├── dataset.py             # Data loading and augmentation
├── training.py            # Training logic
├── evaluation.py          # Evaluation metrics
├── run.py                 # Main training script (RUN THIS)
├── preprocess.py          # NIfTI to PNG converter
├── infer_milan.py         # Inference on NIfTI volumes
├── check_pairs.py         # Verify data pairing
├── view_ground_truth.py   # Visualize samples
│
├── png_data/              # Your PNG dataset goes here
│   ├── images/
│   └── masks/
│
├── models/                # Trained models saved here
│   └── best_model.pth
│
└── plots/                 # Training visualizations
    ├── training_curves.png
    └── segmentation_results.png
```

---

## Quick Start Checklist

- [ ] Install Python 3.8+
- [ ] Create virtual environment
- [ ] Install dependencies (PyTorch + others)
- [ ] Prepare dataset (NIfTI → PNG if needed)
- [ ] Verify data pairing (`check_pairs.py`)
- [ ] Edit `config.py` (classes, paths, hyperparameters)
- [ ] Run training (`python run.py`)
- [ ] Check results in `plots/` and `models/`
- [ ] Evaluate (`python evaluation.py`)
- [ ] Run inference on new data

---

## Need Help?

**Common Questions:**

**Q: How much data do I need?**
A: Minimum 50-100 paired images. More is better (200+ recommended).

**Q: What image size should I use?**
A: 256x256 is a good default. Larger (512x512) gives better detail but needs more GPU memory.

**Q: Should I use multiclass or multibinary?**
A: Use multibinary if you have severe class imbalance or overlapping structures. Use multiclass for mutually exclusive classes.

**Q: How do I know if my model is good?**
A: Dice score > 0.7 is generally good. Compare to baseline methods or human inter-rater agreement if available.

**Q: Can I use this for non-medical images?**
A: Yes! The architecture works for any segmentation task. Just adjust the number of classes and input channels.

---

