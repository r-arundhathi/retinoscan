
import hashlib
from pathlib import Path
from itertools import combinations
from PIL import Image

BASE_DIR = Path("image_validation_dataset/split")
SPLITS = ["train", "validation", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def sha256_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def difference_hash(path):
    with Image.open(path) as image:
        image = image.convert("L").resize((9, 8))
        pixels = list(image.getdata())

    bits = []
    for row in range(8):
        for col in range(8):
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]
            bits.append("1" if left > right else "0")

    return int("".join(bits), 2)

def hamming_distance(hash1, hash2):
    return (hash1 ^ hash2).bit_count()

# Collect images
records = []

for split in SPLITS:
    for class_name in ["retinal", "non_retinal"]:
        folder = BASE_DIR / split / class_name

        for path in folder.rglob("*"):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                records.append({
                    "path": path,
                    "split": split,
                    "class": class_name,
                    "sha": sha256_hash(path),
                    "dhash": difference_hash(path)
                })
            except Exception as error:
                print(f"Could not process {path}: {error}")

print(f"\nTotal images checked: {len(records)}")

# Check exact duplicates across different splits
exact_duplicates = []

for a, b in combinations(records, 2):
    if a["split"] != b["split"] and a["sha"] == b["sha"]:
        exact_duplicates.append((a, b))

# Check visually similar images across different splits
similar_images = []

for a, b in combinations(records, 2):
    if a["split"] == b["split"]:
        continue

    # Exact duplicates are already reported separately
    if a["sha"] == b["sha"]:
        continue

    distance = hamming_distance(a["dhash"], b["dhash"])

    if distance <= 4:
        similar_images.append((distance, a, b))

similar_images.sort(key=lambda item: item[0])

# Create report
report_path = Path("image_validation_dataset/duplicate_check_report.txt")

lines = [
    "IMAGE VALIDATOR DUPLICATE CHECK",
    f"Total images checked: {len(records)}",
    "",
    f"Exact duplicates across splits: {len(exact_duplicates)}",
    ""
]

for a, b in exact_duplicates:
    lines.append(f"{a['split']} ({a['class']}): {a['path']}")
    lines.append(f"{b['split']} ({b['class']}): {b['path']}")
    lines.append("")

lines.append(
    f"Potential visually similar pairs across splits: {len(similar_images)}"
)
lines.append("")

for distance, a, b in similar_images[:100]:
    lines.append(f"Difference score: {distance}")
    lines.append(f"{a['split']} ({a['class']}): {a['path']}")
    lines.append(f"{b['split']} ({b['class']}): {b['path']}")
    lines.append("")

if len(similar_images) > 100:
    lines.append("Only the first 100 similar pairs are listed.")

report_path.write_text("\n".join(lines), encoding="utf-8")

print(f"Exact duplicate pairs: {len(exact_duplicates)}")
print(f"Potential similar pairs: {len(similar_images)}")
print(f"\nReport saved to: {report_path}")
