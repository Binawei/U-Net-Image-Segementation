import os
import sys
import config


def search_patient(patient_id):
    img_dir  = config.DATA_DIR
    mask_dir = config.MASK_DIR

    all_images = sorted(os.listdir(img_dir))
    matches = [f for f in all_images if patient_id.lower() in f.lower()]

    if not matches:
        print(f"No images found for patient ID: '{patient_id}'")
        return

    print(f"\nFound {len(matches)} image(s) for '{patient_id}':")
    for f in matches:
        mask_exists = os.path.exists(os.path.join(mask_dir, f))
        print(f"  {f}  |  mask: {'YES' if mask_exists else 'MISSING'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 search_patient.py <patient_id>")
        print("Example: python3 search_patient.py ACG21217436")
        sys.exit(1)

    search_patient(sys.argv[1])
