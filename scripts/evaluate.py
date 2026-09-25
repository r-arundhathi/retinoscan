import tensorflow as tf
import numpy as np
import json
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

SPLIT_DIR = r"C:\retina_project\split"
MODEL_PATH = r"C:\retina_project\models\best_model_finetuned.keras"
OUTPUT_DIR = r"C:\retina_project\outputs"
IMG_SIZE = 224
BATCH_SIZE = 32

def main():
    with open(os.path.join(OUTPUT_DIR, "class_names.json")) as f:
        class_names = json.load(f)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(SPLIT_DIR, "test"),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    print(f"Loading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    y_true = []
    y_pred_probs = []
    for x, y in test_ds:
        preds = model.predict(x, verbose=0)
        y_pred_probs.extend(preds)
        y_true.extend(np.argmax(y.numpy(), axis=1))

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n=== CLASSIFICATION REPORT ===")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()
    print(f"\nConfusion matrix saved to outputs/confusion_matrix.png")

    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_names))
    auc_scores = {}
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        auc = roc_auc_score(y_true_onehot[:, i], y_pred_probs[:, i])
        auc_scores[name] = auc
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"))
    plt.close()

    print("\n=== ROC-AUC SCORES (per class) ===")
    for name, auc in auc_scores.items():
        print(f"  {name}: {auc:.4f}")

    with open(os.path.join(OUTPUT_DIR, "auc_scores.json"), "w") as f:
        json.dump(auc_scores, f, indent=2)

    print(f"\nAll evaluation outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()