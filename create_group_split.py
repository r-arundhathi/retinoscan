
import hashlib
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

SOURCE_DIR = Path("processed")
OUTPUT_DIR = Path("split_grouped")

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

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


def get_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_source_id(filename, class_name):
    name = Path(filename).stem

    if class_name == "anemia":
        name = re.sub(
            r"_(forniceal_palpebral|forniceal|palpebral|papebral)$",
            "",
            name,
            flags=re.IGNORECASE,
        )
        name = re.sub(r" - Copy$", "", name, flags=re.IGNORECASE)
        name = re.sub(r"\(\d+\)$", "", name)

    elif class_name == "heart_disease":
        name = re.sub(r"-GF-600", "-600", name, flags=re.IGNORECASE)
        name = re.sub(
            r"-(BF|FA|HB|HBF|HF|HFF|SF|FF|FS|ALL)$",
            "",
            name,
            flags=re.IGNORECASE,
        )
        name = re.sub(r"-600$", "", name, flags=re.IGNORECASE)

    return name


def main():
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(
            f"Source folder not found: {SOURCE_DIR.resolve()}"
        )

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"{OUTPUT_DIR} already contains files. "
            "Nothing was overwritten."
        )

    rng = random.Random(SEED)
    groups_by_class = defaultdict(lambda: defaultdict(list))
    seen_hashes = set()
    duplicate_count = 0

    for cls in CLASSES:
        class_dir = SOURCE_DIR / cls

        if not class_dir.is_dir():
            print(f"WARNING: Missing class folder: {class_dir}")
            continue

        for path in class_dir.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            digest = get_hash(path)

            if digest in seen_hashes:
                duplicate_count += 1
                continue

            seen_hashes.add(digest)

            source_id = get_source_id(path.name, cls)
            groups_by_class[cls][source_id].append(path)

    assignments = {"train": [], "val": [], "test": []}

    for cls in CLASSES:
        groups = list(groups_by_class[cls].items())
        rng.shuffle(groups)

        n_groups = len(groups)
        train_end = int(n_groups * TRAIN_RATIO)
        val_end = train_end + int(n_groups * VAL_RATIO)

        split_groups = {
            "train": groups[:train_end],
            "val": groups[train_end:val_end],
            "test": groups[val_end:],
        }

        for split_name, selected_groups in split_groups.items():
            for source_id, paths in selected_groups:
                for path in paths:
                    assignments[split_name].append((cls, path))

    # Copy only after the group assignments have been prepared.
    for split_name, items in assignments.items():
        for cls, source_path in items:
            destination = OUTPUT_DIR / split_name / cls
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination / source_path.name)

    print("\n=== GROUPED SPLIT COMPLETE ===")

    for split_name, items in assignments.items():
        print(f"{split_name}: {len(items)} images")

    print(f"Exact duplicate files skipped: {duplicate_count}")
    print(f"Total images copied: {sum(len(v) for v in assignments.values())}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")
    print("\nNote: Filename grouping is only an approximation of source identity.")


if __name__ == "__main__":
    main()
