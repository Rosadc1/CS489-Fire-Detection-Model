import os
import shutil
from pathlib import Path

# --- CONFIG ---
# Root folder containing train/, test/, and valid/
SOURCE_DIR = Path("objectDetection")
# Destination folder for the unified dataset
DEST_DIR = Path("Dataset")

# Subfolders for images and labels in the new dataset
IMAGES_OUT = DEST_DIR / "images"
LABELS_OUT = DEST_DIR / "labels"

# Create destination directories if they don't exist
IMAGES_OUT.mkdir(parents=True, exist_ok=True)
LABELS_OUT.mkdir(parents=True, exist_ok=True)

# Accepted image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

counter = 0

# Iterate over train, test, valid
for subset in ["train", "test", "valid"]:
    subset_path = SOURCE_DIR / subset
    image_dir = subset_path / "images"
    label_dir = subset_path / "labels"

    if not image_dir.exists() or not label_dir.exists():
        print(f"Skipping {subset} — missing images or labels folder.")
        continue

    # Iterate over images
    for img_path in image_dir.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        # Corresponding label file
        label_path = label_dir / f"{img_path.stem}.txt"

        # Define new filenames
        new_name = f"fire_obj_detect_{counter}"
        new_img_path = IMAGES_OUT / f"{new_name}{img_path.suffix.lower()}"
        new_label_path = LABELS_OUT / f"{new_name}.txt"

        # Copy image
        shutil.copy2(img_path, new_img_path)

        # Copy label if it exists
        if label_path.exists():
            shutil.copy2(label_path, new_label_path)
        else:
            print(f"⚠️ No label found for {img_path.name}, skipping label.")

        counter += 1

print(f"\n✅ Combined dataset created in: {DEST_DIR}")
print(f"Total files processed: {counter}")
