import os
import shutil

BASE = "/Users/postgres/Desktop/data"
OLD_MASKS = os.path.join(BASE, "masks/old_files")
OUTPUT = "/Users/postgres/Desktop/phd/Project/adenomyosis/masks"
CSV_FILE = os.path.join(BASE, "pairings_others.csv")

# Same mapping as combinedFolder.py
DIR_TO_COLUMN = {
    "nii_files_a": "sample_name_adeno_nii_a",
    "nii_files_a_2": "sample_name_adeno_nii_a_2",
    "nii_files_a_3": "sample_name_adeno_nii_a_3",
    "nii_files_b": "sample_name_normal_nii_b",
    "nii_files_b_2": "sample_name_normal_nii_b_2",
    "nii_files_b_3": "sample_name_normal_nii_b_3",
}


def parse_csv(csv_path):
    """Parse the multi-section CSV into {column_name: {sample_num: real_name}}"""
    mappings = {}
    current_col = None

    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(","):
                current_col = None
                continue
            if line.startswith("sample_num"):
                parts = line.split(",")
                current_col = parts[1].strip()
                mappings[current_col] = {}
                continue
            if current_col:
                parts = line.split(",")
                try:
                    num = int(parts[0].strip())
                except ValueError:
                    continue
                name = parts[1].strip()
                if name:
                    mappings[current_col][num] = name
    return mappings


def copy_named_masks(folder, output, seen_names):
    """Copy masks that already have real names (Aug/Sep 2023 folders)."""
    copied = 0
    skipped_dup = 0
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".nii"):
            continue
        if fname in seen_names:
            print(f"  DUPLICATE skipped: {fname}")
            skipped_dup += 1
            continue
        seen_names.add(fname)
        shutil.copy2(os.path.join(folder, fname), os.path.join(output, fname))
        copied += 1
        print(f"  {os.path.basename(folder)}/{fname} -> {fname}")
    return copied, skipped_dup


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    mappings = parse_csv(CSV_FILE)

    seen_names = set()
    copied = 0
    skipped_dup = 0
    skipped_no_mapping = 0

    for dir_name, col_name in DIR_TO_COLUMN.items():
        dir_path = os.path.join(OLD_MASKS, dir_name)
        if not os.path.isdir(dir_path):
            print(f"WARNING: Directory not found: {dir_path}")
            continue
        if col_name not in mappings:
            print(f"WARNING: Column '{col_name}' not found in CSV")
            continue

        name_map = mappings[col_name]
        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith(".nii"):
                continue

            # Extract sample number (handles "sample_0.nii" and "sample 7.nii")
            num_str = fname.replace("sample_", "").replace("sample ", "").replace(".nii", "").strip()
            try:
                num = int(num_str)
            except ValueError:
                continue

            if num not in name_map:
                print(f"  No mapping for {dir_name}/{fname}")
                skipped_no_mapping += 1
                continue

            # Convert .mp4 name to .nii
            new_name = name_map[num].replace(".mp4", ".nii")

            if new_name in seen_names:
                print(f"  DUPLICATE skipped: {new_name} (from {dir_name}/{fname})")
                skipped_dup += 1
                continue

            seen_names.add(new_name)
            src = os.path.join(dir_path, fname)
            dst = os.path.join(OUTPUT, new_name)
            shutil.copy2(src, dst)
            copied += 1
            print(f"  {dir_name}/{fname} -> {new_name}")

    print(f"\nDone! Copied: {copied}, Duplicates skipped: {skipped_dup}, No mapping: {skipped_no_mapping}")

    # Copy Aug/Sep 2023 masks (already have real names)
    for folder in ["Segmentations Aug 2023", "Segmentations Sep 2023"]:
        print(f"\n--- {folder} ---")
        c, d = copy_named_masks(os.path.join(BASE, "masks", folder), OUTPUT, seen_names)
        copied += c
        skipped_dup += d

    print(f"\nTotal — Copied: {copied}, Duplicates skipped: {skipped_dup}, No mapping: {skipped_no_mapping}")


if __name__ == "__main__":
    main()
