import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

IMG_SIZE = 224
NUM_CLASSES = 6

def build_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE, freeze_base=True):
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = not freeze_base

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model

if __name__ == "__main__":
    model, base = build_model()
    model.summary()
    print(f"\nBase model trainable: {base.trainable}")
    print(f"Total layers in base model: {len(base.layers)}")
