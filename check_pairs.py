import os

DATA_DIR = '/Users/postgres/Desktop/phd/Project/adenomyosis/data'
MASK_DIR = '/Users/postgres/Desktop/phd/Project/adenomyosis/masks'

# Get image base names (strip extensions)
images = set(f.replace('.nii.gz', '').replace('.nii', '') for f in os.listdir(DATA_DIR))

# Get mask base names (strip .nii)
masks = set(f.replace('.nii', '') for f in os.listdir(MASK_DIR))

# Masks without corresponding images
orphan_masks = sorted(masks - images)
print(f"Masks without corresponding images: {len(orphan_masks)}")
for m in orphan_masks:
    print(f"  {m}.nii")

# Images without corresponding masks
orphan_images = sorted(images - masks)
print(f"\nImages without corresponding masks: {len(orphan_images)}")
for i in orphan_images:
    print(f"  {i}.nii.gz")

# Matched pairs
matched = images & masks
print(f"\nMatched image-mask pairs: {len(matched)}")
