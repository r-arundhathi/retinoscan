
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

# Configuration
BASE_DIR = Path("image_validation_dataset/clean_split")
IMG_SIZE = (160, 160)
BATCH_SIZE = 8
EPOCHS = 10
SEED = 42

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    BASE_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["non_retinal", "retinal"],
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    BASE_DIR / "validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["non_retinal", "retinal"],
    seed=SEED
)

# Optimize input pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Lightweight pretrained model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = layers.Input(shape=(160, 160, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(),
             tf.keras.metrics.Recall()]
)

model.summary()

# Save best model
Path("image_validation_dataset/models").mkdir(
    parents=True, exist_ok=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "image_validation_dataset/models/best_validator.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

model.save("image_validation_dataset/models/final_validator.keras")

print("\nTraining completed successfully!")
