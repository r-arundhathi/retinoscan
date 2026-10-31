
import hashlib
import random
import shutil
from pathlib import Path

SOURCE_DIR = Path(r"C:\retina_project\processed")
OUTPUT_DIR = Path(r"C:\retina_project\split")

CLASSES = [
    "anemia",
    "diabeties",
    "heart_disease",
    "hypertension",
    "stroke",
    "normal",
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Source folder not found: {SOURCE_DIR}")

    rng = random.Random(SEED)
    assignments = {"train": [], "val": [], "test": []}
    seen_hashes = set()
    total_duplicates = 0

    # Group identical images so they cannot cross dataset splits.
    for cls in CLASSES:
        class_dir = SOURCE_DIR / cls

        if not class_dir.is_dir():
            print(f"Skipping missing class: {cls}")
            continue

        groups = {}

        for path in class_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            digest = get_hash(path)

            if digest in seen_hashes:
                total_duplicates += 1
                print(f"Skipping duplicate image: {path}")
                continue

            seen_hashes.add(digest)
            groups.setdefault(digest, []).append(path)

        # Each unique image is assigned to exactly one split.
        images = [paths[0] for paths in groups.values()]
        rng.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        assignments["train"].extend((cls, p) for p in images[:n_train])
        assignments["val"].extend(
            (cls, p) for p in images[n_train:n_train + n_val]
        )
        assignments["test"].extend(
            (cls, p) for p in images[n_train + n_val:]
        )

    # Refuse to overwrite existing results automatically.
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"{OUTPUT_DIR} already contains files. "
            "Rename or move the existing split folder before rebuilding."
        )

    for split, items in assignments.items():
        for cls, source_path in items:
            destination_dir = OUTPUT_DIR / split / cls
            destination_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_dir / source_path.name)

    print("\n=== NEW SPLIT COMPLETE ===")
    for split, items in assignments.items():
        print(f"{split}: {len(items)} images")

    print(f"Exact duplicate files skipped: {total_duplicates}")
    print(f"Total unique images assigned: {sum(map(len, assignments.values()))}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
