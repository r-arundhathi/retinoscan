
import hashlib
import shutil
from pathlib import Path

SOURCE = Path("image_validation_dataset/non_retinal")
DESTINATION = Path("image_validation_dataset/non_retinal_clean")

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DESTINATION.mkdir(parents=True, exist_ok=True)

seen_hashes = set()
total = 0
copied = 0
duplicates = 0

for image_path in SOURCE.rglob("*"):

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

    relative_path = image_path.relative_to(SOURCE)
    destination_path = DESTINATION / relative_path

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, destination_path)
    copied += 1

print("\nNON-RETINAL CLEANING COMPLETED")
print("Original images:", total)
print("Unique images copied:", copied)
print("Duplicate copies skipped:", duplicates)
print("Clean dataset:", DESTINATION)
