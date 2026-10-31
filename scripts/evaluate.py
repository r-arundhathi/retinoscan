
import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(r"C:\retina_project")
SPLIT_DIR = ROOT / "split_grouped"
MODEL_DIR = ROOT / "models_grouped"
OUTPUT_DIR = ROOT / "outputs_grouped"

IMG_SIZE = 224
BATCH_SIZE = 32

MODELS = {
    "phase1": MODEL_DIR / "best_model_phase1.keras",
    "phase2": MODEL_DIR / "best_model_finetuned.keras",
}


def main():
    with open(OUTPUT_DIR / "class_names.json") as f:
        class_names = json.load(f)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        SPLIT_DIR / "test",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    y_true = np.concatenate([
        np.argmax(y.numpy(), axis=1)
        for _, y in test_ds
    ])

    results = {}

    for model_name, model_path in MODELS.items():
        if not model_path.is_file():
            print(f"Skipping missing model: {model_path}")
            continue

        print(f"\n{'=' * 55}")
        print(f"Evaluating {model_name.upper()}: {model_path}")

        model = tf.keras.models.load_model(model_path)

        probabilities = np.concatenate([
            model.predict(x, verbose=0)
            for x, _ in test_ds
        ])

        predictions = np.argmax(probabilities, axis=1)

        report = classification_report(
            y_true,
            predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
        print("\nCLASSIFICATION REPORT")
        print(report)

        model_output = OUTPUT_DIR / f"evaluation_{model_name}"
        model_output.mkdir(parents=True, exist_ok=True)

        (model_output / "classification_report.txt").write_text(report)

        cm = confusion_matrix(
            y_true, predictions,
            labels=list(range(len(class_names)))
        )

        plt.figure(figsize=(9, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {model_name.upper()}")
        plt.tight_layout()
        plt.savefig(model_output / "confusion_matrix.png")
        plt.close()

        y_onehot = tf.keras.utils.to_categorical(
            y_true, num_classes=len(class_names)
        )

        auc_scores = {}
        plt.figure(figsize=(9, 7))

        for i, class_name in enumerate(class_names):
            auc = roc_auc_score(y_onehot[:, i], probabilities[:, i])
            auc_scores[class_name] = float(auc)

            fpr, tpr, _ = roc_curve(
                y_onehot[:, i], probabilities[:, i]
            )
            plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc:.3f})")

        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves - {model_name.upper()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(model_output / "roc_curves.png")
        plt.close()

        with open(model_output / "auc_scores.json", "w") as f:
            json.dump(auc_scores, f, indent=2)

        accuracy = float(np.mean(predictions == y_true))
        macro_f1 = float(
            __import__("sklearn.metrics", fromlist=["f1_score"])
            .f1_score(y_true, predictions, average="macro", zero_division=0)
        )

        results[model_name] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "macro_auc": float(np.mean(list(auc_scores.values()))),
        }

        print("ROC-AUC SCORES")
        for name, score in auc_scores.items():
            print(f"{name}: {score:.4f}")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Macro AUC: {results[model_name]['macro_auc']:.4f}")
        print(f"Saved results to: {model_output}")

        del model

    with open(OUTPUT_DIR / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== MODEL COMPARISON ===")
    for name, metrics in results.items():
        print(
            f"{name}: "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Macro F1={metrics['macro_f1']:.4f}, "
            f"Macro AUC={metrics['macro_auc']:.4f}"
        )

    print(f"\nComparison saved to: {OUTPUT_DIR / 'model_comparison.json'}")


if __name__ == "__main__":
    main()
