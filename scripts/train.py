import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import json
import os

from model import build_model

SPLIT_DIR = r"C:\retina_project\split_grouped"
MODEL_DIR = r"C:\retina_project\models_grouped"
OUTPUT_DIR = r"C:\retina_project\outputs_grouped"
IMG_SIZE = 224
BATCH_SIZE = 32
INITIAL_EPOCHS = 20

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.15),
    layers.RandomContrast(0.15),
])

def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(SPLIT_DIR, "train"),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=42
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(SPLIT_DIR, "val"),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(SPLIT_DIR, "test"),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )
    class_names = train_ds.class_names
    return train_ds, val_ds, test_ds, class_names

def compute_weights(train_ds, class_names):
    labels = []
    for _, y in train_ds.unbatch():
        labels.append(np.argmax(y.numpy()))
    labels = np.array(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=labels
    )
    class_weight_dict = {i: w for i, w in enumerate(weights)}
    print("\nClass weights (to counter imbalance):")
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_weight_dict[i]:.3f}")
    return class_weight_dict

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading datasets...")
    train_ds, val_ds, test_ds, class_names = load_datasets()
    print(f"Classes found: {class_names}")

    with open(os.path.join(OUTPUT_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    class_weight_dict = compute_weights(train_ds, class_names)

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    print("\nBuilding model...")
    model, base_model = build_model(num_classes=len(class_names), freeze_base=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_path = os.path.join(MODEL_DIR, "best_model_phase1.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    print("\n=== PHASE 1: Training classification head (base frozen) ===")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    with open(os.path.join(OUTPUT_DIR, "history_phase1.json"), "w") as f:
        json.dump(history.history, f)

    print("\nPhase 1 training complete.")
    print(f"Best model saved to: {checkpoint_path}")

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nPhase 1 test accuracy: {test_acc:.4f}")
    print(f"Phase 1 test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
