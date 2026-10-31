
import random
import shutil
from pathlib import Path

random.seed(42)

BASE = Path("image_validation_dataset")

SOURCES = {
    "retinal": BASE / "retinal_clean",
    "non_retinal": BASE / "non_retinal_clean"
}

OUTPUT = BASE / "clean_split"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Create output folders
for split in ["train", "validation", "test"]:
    for label in SOURCES:
        (OUTPUT / split / label).mkdir(
            parents=True, exist_ok=True
        )

for label, source in SOURCES.items():

    images = [
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * 0.70)
    validation_end = train_end + int(total * 0.15)

    splits = {
        "train": images[:train_end],
        "validation": images[train_end:validation_end],
        "test": images[validation_end:]
    }

    print(f"\n{label.upper()}")

    for split, files in splits.items():

        destination = OUTPUT / split / label

        for index, image in enumerate(files):
            new_name = f"{label}_{index:05d}{image.suffix.lower()}"
            shutil.copy2(image, destination / new_name)

        print(f"{split}: {len(files)} images")

print("\nClean dataset split completed.")
print("Output:", OUTPUT)
