import os
import cv2
import numpy as np

DATASET_DIR = r"C:\retina_project"
OUTPUT_DIR = r"C:\retina_project\processed"
CLASSES = ["anemia", "diabeties", "heart_disease", "hypertension", "stroke", "normal"]
IMG_SIZE = 224

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

def preprocess_image(fpath, size=IMG_SIZE):
    img = cv2.imread(fpath)
    if img is None:
        return None
    img = apply_clahe(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_processed = 0
    total_skipped = 0

    for cls in CLASSES:
        src_folder = os.path.join(DATASET_DIR, cls)
        dst_folder = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(dst_folder, exist_ok=True)

        if not os.path.isdir(src_folder):
            print(f"Skipping missing folder: {src_folder}")
            continue

        count = 0
        for fname in os.listdir(src_folder):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                continue

            src_path = os.path.join(src_folder, fname)
            processed = preprocess_image(src_path)

            if processed is None:
                print(f"Skipped unreadable file: {src_path}")
                total_skipped += 1
                continue

            base_name = os.path.splitext(fname)[0]
            dst_path = os.path.join(dst_folder, base_name + ".jpg")
            cv2.imwrite(dst_path, processed, [cv2.IMWRITE_JPEG_QUALITY, 95])
            count += 1
            total_processed += 1

        print(f"{cls}: processed {count} images")

    print(f"\nTotal processed: {total_processed}")
    print(f"Total skipped: {total_skipped}")
    print(f"Output saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()