import os
import cv2
import pandas as pd

DATASET_DIR = r"C:\retina_project"
CLASSES = ["anemia", "diabeties", "heart_disease", "hypertension", "stroke", "normal"]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

report = []

for cls in CLASSES:
    folder = os.path.join(DATASET_DIR, cls)
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        continue

    total = 0
    corrupt = 0
    formats = {}
    widths, heights = [], []

    for fname in os.listdir(folder):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in VALID_EXTS:
            continue
        fpath = os.path.join(folder, fname)
        total += 1
        formats[ext] = formats.get(ext, 0) + 1

        img = cv2.imread(fpath)
        if img is None:
            corrupt += 1
            print(f"Genuinely unreadable file: {fpath}")
        else:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)

    report.append({
        "class": cls,
        "total_images": total,
        "corrupt_images": corrupt,
        "formats": formats,
        "min_width": min(widths) if widths else None,
        "max_width": max(widths) if widths else None,
        "min_height": min(heights) if heights else None,
        "max_height": max(heights) if heights else None,
    })

df = pd.DataFrame(report)
print("\n=== DATASET SUMMARY ===")
print(df.to_string(index=False))

os.makedirs(os.path.join(DATASET_DIR, "outputs"), exist_ok=True)
df.to_csv(os.path.join(DATASET_DIR, "outputs", "dataset_summary.csv"), index=False)
print("\nSaved summary to outputs/dataset_summary.csv")

if len(df) > 0 and df["total_images"].sum() > 0:
    max_count = df["total_images"].max()
    min_count = df["total_images"].min()
    ratio = max_count / min_count if min_count > 0 else float("inf")
    print(f"\nClass imbalance ratio (max/min): {ratio:.2f}")
