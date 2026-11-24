import nibabel as nib
import numpy as np
from PIL import Image
import os

IMG_DIR = "/Users/postgres/Desktop/phd/Project/adenomyosis/data"
MASK_DIR = "/Users/postgres/Desktop/phd/Project/adenomyosis/masks"
OUT_IMG_DIR = "/Users/postgres/Desktop/phd/Project/adenomyosis/png_data/images"
OUT_MASK_DIR = "/Users/postgres/Desktop/phd/Project/adenomyosis/png_data/masks"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)


def extract_slices(img_dir, mask_dir, out_img_dir, out_mask_dir):
    paired = 0
    skipped = 0

    img_files = [f for f in os.listdir(img_dir) if f.endswith((".nii", ".nii.gz"))]

    for img_file in sorted(img_files):
        # Find matching mask (same name, .nii extension)
        base = img_file.replace(".nii.gz", "").replace(".nii", "")
        mask_file = base + ".nii"
        mask_path = os.path.join(mask_dir, mask_file)

        if not os.path.exists(mask_path):
            print(f"  SKIP (no mask): {img_file}")
            skipped += 1
            continue

        # Load NIfTI files
        try:
            img_data = nib.load(os.path.join(img_dir, img_file)).get_fdata()
            mask_data = nib.load(mask_path).get_fdata()
        except Exception as e:
            print(f"  SKIP (error loading): {img_file} → {e}")
            skipped += 1
            continue

        num_slices = img_data.shape[2]

        for s in range(num_slices):
            # Extract single slice (1600, 864)
            img_slice = img_data[:, :, s]
            mask_slice = mask_data[:, :, s]

            # Normalize image to 0-1 first, then scale to 0-255 for uint8 PNG
            img_slice = img_slice - img_slice.min()
            if img_slice.max() > 0:
                img_slice = img_slice / img_slice.max()
            img_slice = (img_slice * 255).astype(np.uint8)


            mask_slice = mask_slice.astype(np.uint8)

            # Save as PNG
            out_name = f"{base}_slice{s}.png"
            Image.fromarray(img_slice, mode="L").save(os.path.join(out_img_dir, out_name))
            Image.fromarray(mask_slice, mode="L").save(os.path.join(out_mask_dir, out_name))

            paired += 1

        print(f"  OK: {base} ({num_slices} slices)")

    print(f"\nDone! Total PNG pairs: {paired}, Skipped: {skipped}")


if __name__ == "__main__":
    extract_slices(IMG_DIR, MASK_DIR, OUT_IMG_DIR, OUT_MASK_DIR)
