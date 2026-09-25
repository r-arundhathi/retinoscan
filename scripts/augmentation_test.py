import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

SAMPLE_IMG_DIR = r"C:\retina_project\split\train\anemia"
OUTPUT_PREVIEW = r"C:\retina_project\outputs\augmentation_preview.png"

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.15),
    layers.RandomContrast(0.15),
])

def main():
    files = [f for f in os.listdir(SAMPLE_IMG_DIR) if f.lower().endswith(".jpg")]
    sample_path = os.path.join(SAMPLE_IMG_DIR, files[0])

    img = tf.keras.utils.load_img(sample_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    plt.figure(figsize=(12, 6))
    for i in range(8):
        augmented = data_augmentation(img_array, training=True)
        ax = plt.subplot(2, 4, i + 1)
        plt.imshow(augmented[0].numpy().astype("uint8"))
        plt.axis("off")

    plt.suptitle(f"Augmentation preview: {files[0]}")
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PREVIEW), exist_ok=True)
    plt.savefig(OUTPUT_PREVIEW)
    print(f"Saved preview to: {OUTPUT_PREVIEW}")

if __name__ == "__main__":
    main()
