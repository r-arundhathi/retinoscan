
import hashlib
import shutil
from pathlib import Path

SOURCE = Path("image_validation_dataset/retinal")
DESTINATION = Path("image_validation_dataset/retinal_clean")

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DESTINATION.mkdir(parents=True, exist_ok=True)

total = 0
copied = 0
duplicates = 0

for class_folder in SOURCE.iterdir():

    if not class_folder.is_dir() or class_folder.name == "split":
        continue

    output_folder = DESTINATION / class_folder.name
    output_folder.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()

    for image_path in sorted(class_folder.rglob("*")):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in EXTENSIONS:
            continue

        total += 1

        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()

        if digest in seen_hashes:
            duplicates += 1
            continue

        seen_hashes.add(digest)

        destination_path = output_folder / image_path.name

        shutil.copy2(image_path, destination_path)
        copied += 1

print("\nCLEANING COMPLETED")
print("Original images:", total)
print("Unique images copied:", copied)
print("Duplicate copies skipped:", duplicates)
print("Clean dataset:", DESTINATION)
