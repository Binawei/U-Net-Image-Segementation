import nibabel as nib
import numpy as np
import os
import random

sources = {
    "old_files (sample_N)": "/Users/postgres/Desktop/data/masks/old_files/nii_files_a",
    "Segmentations Aug 2023": "/Users/postgres/Desktop/data/masks/Segmentations Aug 2023",
    "Segmentations Sep 2023": "/Users/postgres/Desktop/data/masks/Segmentations Sep 2023",
}

for name, folder in sources.items():
    files = [f for f in os.listdir(folder) if f.endswith(".nii")]
    f = random.choice(files)
    data = nib.load(os.path.join(folder, f)).get_fdata().astype(int)

    h, w = data.shape[0], data.shape[1]

    # Find a patch with multiple classes
    patch = None
    for y in range(0, h - 20, 50):
        for x in range(0, w - 20, 50):
            candidate = data[y:y+20, x:x+20, 0]
            if len(np.unique(candidate)) >= 3:
                patch = candidate
                break
        if patch is not None:
            break

    print(f"\n=== {name} ===")
    print(f"File: {f}")
    print(f"Shape: {data.shape}")
    print(f"Unique values in whole mask: {np.unique(data)}")
    print(f"20x20 patch:")
    print(patch)
