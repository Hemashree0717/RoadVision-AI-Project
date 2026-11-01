from pathlib import Path
import shutil

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# DeepGlobe training dataset
SOURCE_DIR = PROJECT_ROOT / "train"

# Destination folders
IMAGE_DIR = PROJECT_ROOT / "backend" / "dataset" / "images"
MASK_DIR = PROJECT_ROOT / "backend" / "dataset" / "masks"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

copied = 0
missing_masks = 0

print("=" * 60)
print("RoadVision AI - Dataset Preparation")
print("=" * 60)

# Find all satellite images
satellite_images = list(SOURCE_DIR.glob("*_sat.jpg"))

print(f"Satellite images found: {len(satellite_images)}")
print()

for sat_file in satellite_images:

    # Example:
    # 104_sat.jpg -> 104_mask.png
    image_id = sat_file.stem.replace("_sat", "")
    mask_file = SOURCE_DIR / f"{image_id}_mask.png"

    if not mask_file.exists():
        print(f"WARNING: Mask not found for {sat_file.name}")
        missing_masks += 1
        continue

    # Destination filenames
    destination_image = IMAGE_DIR / sat_file.name
    destination_mask = MASK_DIR / mask_file.name

    # Copy files
    shutil.copy2(sat_file, destination_image)
    shutil.copy2(mask_file, destination_mask)

    copied += 1

    if copied % 100 == 0:
        print(f"Copied {copied} image-mask pairs...")

print()
print("=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)
print(f"Image-mask pairs copied : {copied}")
print(f"Missing masks           : {missing_masks}")
print(f"Images folder           : {IMAGE_DIR}")
print(f"Masks folder            : {MASK_DIR}")
print("=" * 60)