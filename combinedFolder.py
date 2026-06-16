import os
import shutil
import csv

BASE = "/Users/postgres/Desktop/data"
OLD_FILES = os.path.join(BASE, "images/old_files")
OUTPUT = "/Users/postgres/Desktop/phd/project/adenomyosis/data"
CSV_FILE = os.path.join(BASE, "pairings_others.csv")

# Mapping: directory name -> CSV column name
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
            # Detect header lines
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

def main():
    os.makedirs(OUTPUT, exist_ok=True)
    mappings = parse_csv(CSV_FILE)
    
    # Original files are NOT modified - only copying to output directory
    seen_names = set()
    copied = 0
    skipped_dup = 0
    skipped_no_mapping = 0

    for dir_name, col_name in DIR_TO_COLUMN.items():
        dir_path = os.path.join(OLD_FILES, dir_name)
        if not os.path.isdir(dir_path):
            print(f"WARNING: Directory not found: {dir_path}")
            continue
        if col_name not in mappings:
            print(f"WARNING: Column '{col_name}' not found in CSV")
            continue

        name_map = mappings[col_name]
        for fname in sorted(os.listdir(dir_path)):
            if not fname.startswith("sample_"):
                continue
            # Extract sample number
            num_str = fname.replace("sample_", "").replace(".nii.gz", "").replace(".nii", "")
            try:
                num = int(num_str)
            except ValueError:
                continue

            if num not in name_map:
                print(f"  No mapping for {dir_name}/{fname}")
                skipped_no_mapping += 1
                continue

            # Convert .mp4 name to .nii.gz
            new_name = name_map[num].replace(".mp4", ".nii.gz")

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

if __name__ == "__main__":
    main()
