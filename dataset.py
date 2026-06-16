import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config


# ── Transforms ────────────────────────────────────────────────────────────────

train_transforms = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
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

def data_generator(file_list, img_dir, mask_dir, transform, mask_transform, batch_size, shuffle=True):
    """
    Python generator that yields batches of (image_tensor, mask_tensor).
    Shuffles file list each epoch if shuffle=True.
    """
    while True:
        if shuffle:
            random.shuffle(file_list)

        batch_imgs = []
        batch_masks = []

        for fname in file_list:
            img = Image.open(os.path.join(img_dir, fname)).convert("L")
            mask = Image.open(os.path.join(mask_dir, fname))

            img_tensor = transform(img)                                   # (1, 256, 256) float
            mask_tensor = mask_transform(mask)                            # (1, 256, 256) float
            mask_tensor = (mask_tensor * 255).squeeze(0).long()          # (256, 256) int labels 0-3

            batch_imgs.append(img_tensor)
            batch_masks.append(mask_tensor)

            if len(batch_imgs) == batch_size:
                yield torch.stack(batch_imgs), torch.stack(batch_masks)
                batch_imgs = []
                batch_masks = []

        # Yield remaining samples
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
        transform=train_transforms,
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
