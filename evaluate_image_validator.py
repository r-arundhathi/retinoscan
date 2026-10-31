
import tensorflow as tf
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Paths
BASE_DIR = Path("image_validation_dataset/clean_split/test")
MODEL_PATH = "image_validation_dataset/models/best_validator.keras"
REPORT_PATH = Path("image_validation_dataset/validator_evaluation_report.txt")

# Settings
IMG_SIZE = (160, 160)
BATCH_SIZE = 8
THRESHOLD = 0.5

print("Loading test dataset...")

test_data = tf.keras.utils.image_dataset_from_directory(
    BASE_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=["non_retinal", "retinal"],
    label_mode="binary",
    shuffle=False
)

print("Loading trained validator...")
model = tf.keras.models.load_model(MODEL_PATH)

# Actual labels
y_true = np.concatenate([
    labels.numpy().flatten()
    for images, labels in test_data
]).astype(int)

# Predicted probabilities
predictions = model.predict(test_data, verbose=1).flatten()

# Convert probabilities into classes
y_pred = (predictions >= THRESHOLD).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
auc = roc_auc_score(y_true, predictions)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["Non-retinal", "Retinal"],
    zero_division=0
)

# Display results
results = f"""
========================================
       IMAGE VALIDATOR EVALUATION
========================================

Test images: {len(y_true)}
Classification threshold: {THRESHOLD}

Accuracy  : {accuracy * 100:.2f}%
Precision : {precision * 100:.2f}%
Recall    : {recall * 100:.2f}%
F1-score  : {f1 * 100:.2f}%
ROC-AUC   : {auc:.4f}

CONFUSION MATRIX
Rows = Actual, Columns = Predicted

                  Predicted
                  Non-retinal  Retinal

Actual Non-retinal    {cm[0][0]}          {cm[0][1]}
Actual Retinal        {cm[1][0]}          {cm[1][1]}

CLASSIFICATION REPORT

{report}

========================================
"""

print(results)

# Save report
REPORT_PATH.write_text(results, encoding="utf-8")

print("Report saved successfully:")
print(REPORT_PATH)
