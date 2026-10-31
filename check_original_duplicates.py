
import hashlib
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("image_validation_dataset/retinal")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

hash_groups = defaultdict(list)

print("Checking original retinal images...")

for path in BASE_DIR.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    # Ignore the split folder if encountered
    if "split" in path.parts:
        continue

    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hash_groups[digest].append(path)
    except Exception as error:
        print(f"Could not read {path}: {error}")

duplicates = {
    digest: paths
    for digest, paths in hash_groups.items()
    if len(paths) > 1
}

report = [
    "ORIGINAL RETINAL DATASET DUPLICATE CHECK",
    f"Original images checked: {sum(len(v) for v in hash_groups.values())}",
    f"Duplicate groups: {len(duplicates)}",
    ""
]

for paths in duplicates.values():
    report.append("Duplicate group:")
    for path in paths:
        report.append(str(path))
    report.append("")

output = Path(
    "image_validation_dataset/original_retinal_duplicates.txt"
)

output.write_text("\n".join(report), encoding="utf-8")

print(f"Duplicate groups found: {len(duplicates)}")
print(f"Report saved to: {output}")
