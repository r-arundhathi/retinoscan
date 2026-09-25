import os
import shutil
import random

SOURCE_DIR = r"C:\retina_project\processed"
OUTPUT_DIR = r"C:\retina_project\split"
CLASSES = ["anemia", "diabeties", "heart_disease", "hypertension", "stroke", "normal"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

def main():
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    summary = []

    for cls in CLASSES:
        src_folder = os.path.join(SOURCE_DIR, cls)
        if not os.path.isdir(src_folder):
            print(f"Skipping missing folder: {src_folder}")
            continue

        files = [f for f in os.listdir(src_folder) if f.lower().endswith(".jpg")]
        random.shuffle(files)

        n_total = len(files)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        n_test = n_total - n_train - n_val

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        for fname in train_files:
            shutil.copy2(os.path.join(src_folder, fname), os.path.join(OUTPUT_DIR, "train", cls, fname))
        for fname in val_files:
            shutil.copy2(os.path.join(src_folder, fname), os.path.join(OUTPUT_DIR, "val", cls, fname))
        for fname in test_files:
            shutil.copy2(os.path.join(src_folder, fname), os.path.join(OUTPUT_DIR, "test", cls, fname))

        summary.append((cls, n_train, n_val, n_test, n_total))
        print(f"{cls}: train={n_train}, val={n_val}, test={n_test}, total={n_total}")

    print("\n=== SPLIT COMPLETE ===")
    total_train = sum(s[1] for s in summary)
    total_val = sum(s[2] for s in summary)
    total_test = sum(s[3] for s in summary)
    print(f"Total train: {total_train}")
    print(f"Total val: {total_val}")
    print(f"Total test: {total_test}")
    print(f"Output saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
