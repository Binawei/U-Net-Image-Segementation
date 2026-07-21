import shutil
import os

DATA_DIR    = '/Users/postgres/Desktop/phd/Project/adenomyosis/data'
MASK_DIR    = '/Users/postgres/Desktop/phd/Project/adenomyosis/masks'
IMG_SRC     = '/Users/postgres/Downloads/additional_samples/nii_gz_files'
MASK_AUG    = '/Users/postgres/Downloads/additional_samples/Segmentations Aug 2023'
SAMPLE_ROOT = '/Users/postgres/Downloads/additional_samples'


def main():
    img_copied, img_dup = 0, 0
    mask_copied, mask_dup = 0, 0

    # ── Copy images from nii_gz_files ─────────────────────────────────────────
    for f in sorted(os.listdir(IMG_SRC)):
        if not f.endswith('.nii.gz'):
            continue
        dst = os.path.join(DATA_DIR, f)
        if os.path.exists(dst):
            print(f'  DUPLICATE (image): {f}')
            img_dup += 1
            continue
        shutil.copy2(os.path.join(IMG_SRC, f), dst)
        print(f'  IMAGE copied: {f}')
        img_copied += 1

    # ── Copy sample_0.nii.gz ──────────────────────────────────────────────────
    s0_img_src = os.path.join(SAMPLE_ROOT, 'sample_0.nii.gz')
    s0_img_dst = os.path.join(DATA_DIR, 'PT_Unknown_sample0.nii.gz')
    if os.path.exists(s0_img_src):
        if os.path.exists(s0_img_dst):
            print(f'  DUPLICATE (image): PT_Unknown_sample0.nii.gz')
            img_dup += 1
        else:
            shutil.copy2(s0_img_src, s0_img_dst)
            print(f'  IMAGE copied: sample_0.nii.gz -> PT_Unknown_sample0.nii.gz')
            img_copied += 1

    # ── Copy masks from Segmentations Aug 2023 ────────────────────────────────
    for f in sorted(os.listdir(MASK_AUG)):
        if not f.endswith('.nii'):
            continue
        dst = os.path.join(MASK_DIR, f)
        if os.path.exists(dst):
            print(f'  DUPLICATE (mask): {f}')
            mask_dup += 1
            continue
        shutil.copy2(os.path.join(MASK_AUG, f), dst)
        print(f'  MASK copied: {f}')
        mask_copied += 1

    # ── Copy sample_0.nii ─────────────────────────────────────────────────────
    s0_mask_src = os.path.join(SAMPLE_ROOT, 'sample_0.nii')
    s0_mask_dst = os.path.join(MASK_DIR, 'PT_Unknown_sample0.nii')
    if os.path.exists(s0_mask_src):
        if os.path.exists(s0_mask_dst):
            print(f'  DUPLICATE (mask): PT_Unknown_sample0.nii')
            mask_dup += 1
        else:
            shutil.copy2(s0_mask_src, s0_mask_dst)
            print(f'  MASK copied: sample_0.nii -> PT_Unknown_sample0.nii')
            mask_copied += 1

    print(f'\nImages — copied: {img_copied}, duplicates skipped: {img_dup}')
    print(f'Masks  — copied: {mask_copied}, duplicates skipped: {mask_dup}')


if __name__ == '__main__':
    main()
