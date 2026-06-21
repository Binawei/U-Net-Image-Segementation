import os
import random
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, gaussian_filter

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
import config


# ── Joint augmentation (image + mask simultaneously) ──────────────────────────
def elastic_deform(img_np, mask_np, alpha=80, sigma=10):
    """Apply identical elastic deformation to image and mask numpy arrays."""
    shape = img_np.shape
    dx = gaussian_filter(np.random.randn(*shape), sigma) * alpha
    dy = gaussian_filter(np.random.randn(*shape), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    coords = [np.clip(y + dy, 0, shape[0]-1), np.clip(x + dx, 0, shape[1]-1)]
    img_def  = map_coordinates(img_np,  coords, order=1, mode='reflect').reshape(shape)
    mask_def = map_coordinates(mask_np, coords, order=0, mode='reflect').reshape(shape)
    return img_def, mask_def


def joint_augment(img, mask):
    """img and mask are PIL Images. Returns augmented PIL Images."""
    # Resize first
    img  = TF.resize(img,  (256, 256), interpolation=TF.InterpolationMode.BILINEAR)
    mask = TF.resize(mask, (256, 256), interpolation=TF.InterpolationMode.NEAREST)

    # Random horizontal flip
    if random.random() > 0.5:
        img, mask = TF.hflip(img), TF.hflip(mask)

    # Random vertical flip
    if random.random() > 0.5:
        img, mask = TF.vflip(img), TF.vflip(mask)

    # Random rotation ±15° at 40% probability
    if random.random() < 0.4:
        angle = random.uniform(-15, 15)
        img  = TF.rotate(img,  angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

    # Brightness/contrast jitter on image only (not mask)
    if random.random() < 0.5:
        img = TF.adjust_brightness(img, 1 + random.uniform(-0.3, 0.3))
    if random.random() < 0.5:
        img = TF.adjust_contrast(img, 1 + random.uniform(-0.3, 0.3))

    # Elastic deformation at 30% probability
    if random.random() < 0.3:
        img_np  = np.array(img).astype(np.float32)
        mask_np = np.array(mask).astype(np.float32)
        img_np, mask_np = elastic_deform(img_np, mask_np, alpha=80, sigma=10)
        img  = Image.fromarray(img_np.astype(np.uint8), mode='L')
        mask = Image.fromarray(mask_np.astype(np.uint8))

    return img, mask


# ── Transforms ────────────────────────────────────────────────────────────────

val_transforms = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])


# ── Patient-level split ────────────────────────────────────────────────────────
def get_patient_splits(img_dir, split=config.SPLIT, seed=config.SEED):
    """Split by patient so all slices of a patient stay in the same set."""
    all_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])

    # Extract unique patient names (strip _sliceN.png)
    patients = sorted(set(f.rsplit("_slice", 1)[0] for f in all_files))
    random.seed(seed)
    random.shuffle(patients)

    n = len(patients)
    n_train = int(n * split[0])
    n_val = int(n * split[1])

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train + n_val])
    test_patients = set(patients[n_train + n_val:])

    train_files = [f for f in all_files if f.rsplit("_slice", 1)[0] in train_patients]
    val_files = [f for f in all_files if f.rsplit("_slice", 1)[0] in val_patients]
    test_files = [f for f in all_files if f.rsplit("_slice", 1)[0] in test_patients]

    print(f"Patients  — train: {len(train_patients)}, val: {len(val_patients)}, test: {len(test_patients)}")
    print(f"PNG pairs — train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)}")

    return train_files, val_files, test_files


# ── Python Generator ───────────────────────────────────────────────────────────
def data_generator(file_list, img_dir, mask_dir, mask_transform, batch_size, shuffle=True):
    while True:
        if shuffle:
            random.shuffle(file_list)

        batch_imgs = []
        batch_masks = []

        for fname in file_list:
            img  = Image.open(os.path.join(img_dir,  fname)).convert("L")
            mask = Image.open(os.path.join(mask_dir, fname))

            img, mask = joint_augment(img, mask)

            img_tensor  = transforms.ToTensor()(img)                      # (1, 256, 256) float
            mask_tensor = mask_transform(mask)                            # (1, 256, 256) float
            mask_tensor = (mask_tensor * 255).squeeze(0).long()           # (256, 256) int labels 0-3

            batch_imgs.append(img_tensor)
            batch_masks.append(mask_tensor)

            if len(batch_imgs) == batch_size:
                yield torch.stack(batch_imgs), torch.stack(batch_masks)
                batch_imgs  = []
                batch_masks = []

        if batch_imgs:
            yield torch.stack(batch_imgs), torch.stack(batch_masks)


# ── PyTorch Dataset (for val/test with DataLoader) ─────────────────────────────
class AdenomyosisDataset(Dataset):
    def __init__(self, file_list, img_dir, mask_dir, transform, mask_transform):
        self.file_list = file_list
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("L")
        mask = Image.open(os.path.join(self.mask_dir, fname))

        img_tensor = self.transform(img)
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor * 255).squeeze(0).long()

        return img_tensor, mask_tensor


# ── Get all loaders ────────────────────────────────────────────────────────────
def get_dataloaders():
    train_files, val_files, test_files = get_patient_splits(config.DATA_DIR)

    # Mask transform
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # Training uses generator
    train_gen = data_generator(
        train_files, config.DATA_DIR, config.MASK_DIR,
        mask_transform=mask_transform,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )

    # Val and test use standard DataLoader
    val_ds = AdenomyosisDataset(val_files, config.DATA_DIR, config.MASK_DIR, val_transforms, mask_transform)
    test_ds = AdenomyosisDataset(test_files, config.DATA_DIR, config.MASK_DIR, val_transforms, mask_transform)

    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    steps_per_epoch = len(train_files) // config.BATCH_SIZE

    return train_gen, val_loader, test_loader, test_files, steps_per_epoch
