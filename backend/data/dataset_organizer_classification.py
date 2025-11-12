import os
import shutil
from pathlib import Path

# --- CONFIG ---
# Root folder containing train/, val/, test/
SOURCE_DIR = Path("classification")
# Output folder for unified dataset
DEST_DIR = Path("Dataset_Classification")

# Create destination directories
(DEST_DIR / "Fire").mkdir(parents=True, exist_ok=True)
(DEST_DIR / "Non_Fire").mkdir(parents=True, exist_ok=True)

# Class folders to combine
CLASSES = ["Fire", "Non_Fire"]

counter = {"Fire": 0, "Non_Fire": 0}

for split in ["train", "valid", "test"]:
    for cls in CLASSES:
        src_folder = SOURCE_DIR / split / cls
        if not src_folder.exists():
            print(f"Skipping missing folder: {src_folder}")
            continue

        for img_path in src_folder.iterdir():
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            new_name = f"{cls}_img_{counter[cls]}{img_path.suffix.lower()}"
            dest_path = DEST_DIR / cls / new_name
            shutil.copy2(img_path, dest_path)

            counter[cls] += 1

print("\n✅ Combined classification dataset created:")
print(f"  Fire images: {counter['Fire']}")
print(f"  No fire images: {counter['Non_Fire']}")
print(f"  Location: {DEST_DIR}")
