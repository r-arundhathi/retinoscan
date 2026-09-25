import os
import io
import base64
import json
from datetime import datetime

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_file
from fpdf import FPDF

MODEL_PATH = r"C:\retina_project\models\best_model_finetuned.keras"
CLASS_NAMES_PATH = r"C:\retina_project\outputs\class_names.json"
IMG_SIZE = 224
UNCERTAIN_THRESHOLD = 55.0

DISEASE_INFO = {
    "anemia": {"label": "Anemia", "icon": "🩸", "color": "#e74c3c"},
    "diabeties": {"label": "Diabetes", "icon": "🩹", "color": "#e67e22"},
    "heart_disease": {"label": "Heart Disease", "icon": "🫀", "color": "#c0392b"},
    "hypertension": {"label": "Hypertension", "icon": "💢", "color": "#9b59b6"},
    "stroke": {"label": "Stroke", "icon": "🧠", "color": "#2980b9"},
    "normal": {"label": "Normal", "icon": "✅", "color": "#2ecc71"},
}

app = Flask(__name__)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH) as f:
    CLASS_NAMES = json.load(f)
print("Model loaded. Classes:", CLASS_NAMES)

RESNET_LAYER_NAME = "resnet50"


def preprocess_image(pil_image):
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr


def make_gradcam_heatmap(img_array, pred_index):
    """Manually steps through the model's layers instead of building a
    sub-model, which avoids the 'not connected to inputs' issue that can
    happen with Keras 3 after saving/loading a .keras file."""
    resnet_layer = model.get_layer(RESNET_LAYER_NAME)
    resnet_idx = model.layers.index(resnet_layer)

    x = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:
        x_pre = tf.keras.applications.resnet50.preprocess_input(x)
        conv_outputs = resnet_layer(x_pre, training=False)
        tape.watch(conv_outputs)

        h = conv_outputs
        for layer in model.layers[resnet_idx + 1:]:
            h = layer(h, training=False)

        class_channel = h[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(pil_image, heatmap, alpha=0.45):
    img = np.array(pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlayed = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return Image.fromarray(overlayed)


def pil_to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


LAST_RESULT = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    pil_image = Image.open(file.stream)

    img_array = preprocess_image(pil_image)
    predictions = model.predict(img_array, verbose=0)[0]

    predicted_idx = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(predictions[predicted_idx] * 100)

    heatmap = make_gradcam_heatmap(img_array, predicted_idx)
    overlay_img = overlay_heatmap(pil_image, heatmap)

    is_uncertain = confidence < UNCERTAIN_THRESHOLD
    is_normal = (not is_uncertain) and predicted_class == "normal"

    info = DISEASE_INFO.get(predicted_class, {"label": predicted_class, "icon": "⚠️", "color": "#e74c3c"})

    result = {
        "is_uncertain": is_uncertain,
        "is_normal": is_normal,
        "predicted_class": predicted_class,
        "label": info["label"],
        "icon": info["icon"],
        "color": info["color"],
        "confidence": round(confidence, 1),
        "original_image": pil_to_base64(pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))),
        "heatmap_image": pil_to_base64(overlay_img),
    }

    global LAST_RESULT
    LAST_RESULT = result

    return jsonify(result)


@app.route("/download_report")
def download_report():
    if not LAST_RESULT:
        return "No prediction yet", 400

    r = LAST_RESULT
    orig_bytes = base64.b64decode(r["original_image"])
    heat_bytes = base64.b64decode(r["heatmap_image"])

    orig_path = "temp_orig.png"
    heat_path = "temp_heat.png"
    with open(orig_path, "wb") as f:
        f.write(orig_bytes)
    with open(heat_path, "wb") as f:
        f.write(heat_bytes)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "EyeScan - Prediction Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    y_start = pdf.get_y()
    pdf.image(orig_path, x=25, y=y_start, w=75)
    pdf.image(heat_path, x=110, y=y_start, w=75)
    pdf.set_y(y_start + 78)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(95, 6, "Original Image", align="C")
    pdf.cell(95, 6, "Model Attention (Grad-CAM)", align="C")
    pdf.ln(12)

    pdf.set_font("Helvetica", "B", 15)
    if r["is_uncertain"]:
        pdf.set_text_color(230, 180, 20)
        pdf.cell(0, 10, "Result: Unable to Determine", ln=True)
    elif r["is_normal"]:
        pdf.set_text_color(46, 204, 113)
        pdf.cell(0, 10, "Result: Looks Normal", ln=True)
    else:
        pdf.set_text_color(200, 60, 50)
        pdf.cell(0, 10, f"Result: {r['label']}", ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 12)
    if not r["is_uncertain"]:
        pdf.cell(0, 8, f"Confidence: {r['confidence']}%", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(
        0, 6,
        "The highlighted regions in the Grad-CAM image show which areas of the photo "
        "most influenced the model's prediction. This report is generated by an AI "
        "screening tool and is NOT a medical diagnosis. Please consult a qualified "
        "healthcare provider for confirmation."
    )

    pdf_bytes = bytes(pdf.output())
    os.remove(orig_path)
    os.remove(heat_path)

    buf = io.BytesIO(pdf_bytes)
    buf.seek(0)
    filename = f"eyescan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name=filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)