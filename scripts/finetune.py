import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import json
import os

SPLIT_DIR = r"C:\retina_project\split_grouped"
MODEL_DIR = r"C:\retina_project\models_grouped"
OUTPUT_DIR = r"C:\retina_project\outputs_grouped"
IMG_SIZE = 224
BATCH_SIZE = 32
FINE_TUNE_EPOCHS = 15
UNFREEZE_LAYERS = 30

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
    return {i: w for i, w in enumerate(weights)}

def main():
    print("Loading datasets...")
    train_ds, val_ds, test_ds, class_names = load_datasets()
    class_weight_dict = compute_weights(train_ds, class_names)

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    print("Loading Phase 1 best model...")
    phase1_path = os.path.join(MODEL_DIR, "best_model_phase1.keras")
    model = tf.keras.models.load_model(phase1_path)

    base_model = None
    for layer in model.layers:
        if layer.name == "resnet50":
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not find resnet50 layer in loaded model")

    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"\nUnfroze top {UNFREEZE_LAYERS} layers of ResNet50")
    print(f"Trainable layers in base: {trainable_count} / {len(base_model.layers)}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_path = os.path.join(MODEL_DIR, "best_model_finetuned.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    ]

    print("\n=== PHASE 2: Fine-tuning top ResNet50 layers ===")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    with open(os.path.join(OUTPUT_DIR, "history_phase2.json"), "w") as f:
        json.dump(history.history, f)

    print("\nFine-tuning complete.")
    print(f"Best fine-tuned model saved to: {checkpoint_path}")

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nFine-tuned test accuracy: {test_acc:.4f}")
    print(f"Fine-tuned test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
