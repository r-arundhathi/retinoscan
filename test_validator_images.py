
import tensorflow as tf
import numpy as np
from pathlib import Path

MODEL_PATH = "image_validation_dataset/models/best_validator.keras"

model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (160, 160)

print("Upload an image path to test.")
print("Type exit to stop.")

while True:
    path = input("\nImage path: ").strip().strip('"')

    if path.lower() == "exit":
        break

    image_path = Path(path)

    if not image_path.is_file():
        print("File not found.")
        continue

    try:
        image = tf.keras.utils.load_img(
            image_path,
            target_size=IMG_SIZE
        )

        image = tf.keras.utils.img_to_array(image)
        image = np.expand_dims(image, axis=0)

        probability = float(model.predict(image, verbose=0)[0][0])

        if probability >= 0.5:
            print("RESULT: RETINAL IMAGE")
            print(f"Confidence: {probability * 100:.2f}%")
        else:
            print("RESULT: NON-RETINAL IMAGE")
            print(f"Confidence: {(1 - probability) * 100:.2f}%")

    except Exception as error:
        print("Could not process image:", error)
