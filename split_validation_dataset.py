
from pathlib import Path
import random
import shutil

random.seed(42)

BASE = Path("image_validation_dataset")
OUTPUT = BASE / "split"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

classes = ["retinal", "non_retinal"]

for class_name in classes:

    source = BASE / class_name

    images = [
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)

    split_data = {
        "train": images[:train_end],
        "validation": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split_name, files in split_data.items():

        destination = OUTPUT / split_name / class_name

        destination.mkdir(parents=True, exist_ok=True)

        if any(destination.iterdir()):
            raise RuntimeError(
                f"{destination} is not empty. Stopping to avoid duplicates."
            )

        for index, image in enumerate(files):
            filename = f"{class_name}_{index:05d}{image.suffix.lower()}"

            shutil.copy2(
                image,
                destination / filename
            )

        print(f"{class_name} - {split_name}: {len(files)}")

print("\nDataset splitting completed!")
print("Original images are preserved.")
