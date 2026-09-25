import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import json
import os
import webbrowser
import threading
import tempfile
from datetime import datetime
from fpdf import FPDF

MODEL_PATH = r"C:\retina_project\models\best_model_finetuned.keras"
CLASS_NAMES_PATH = r"C:\retina_project\outputs\class_names.json"
IMG_SIZE = 224
UNCERTAIN_THRESHOLD = 55.0
APP_URL = "http://localhost:8501"

DISEASE_INFO = {
    "anemia": {"label": "Anemia", "icon": "🩸", "color": "#e74c3c"},
    "diabeties": {"label": "Diabetes", "icon": "🩹", "color": "#e67e22"},
    "heart_disease": {"label": "Heart Disease", "icon": "🫀", "color": "#c0392b"},
    "hypertension": {"label": "Hypertension", "icon": "💢", "color": "#9b59b6"},
    "stroke": {"label": "Stroke", "icon": "🧠", "color": "#2980b9"},
    "normal": {"label": "Normal", "icon": "✅", "color": "#2ecc71"},
}

CUSTOM_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.hero { text-align: center; padding: 1.5rem 0 0.5rem 0; }
.hero-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero-subtitle { font-size: 1.05rem; color: #a0a0a0; margin-bottom: 1rem; }
.result-card {
    padding: 2rem; border-radius: 18px; margin-top: 1.5rem;
    text-align: center; box-shadow: 0 8px 24px rgba(0,0,0,0.15);
}
.result-normal {
    background: linear-gradient(135deg, rgba(46, 204, 113, 0.15), rgba(46, 204, 113, 0.05));
    border: 1px solid rgba(46, 204, 113, 0.4);
}
.result-disease {
    background: linear-gradient(135deg, rgba(231, 76, 60, 0.15), rgba(231, 76, 60, 0.05));
    border: 1px solid rgba(231, 76, 60, 0.4);
}
.result-uncertain {
    background: linear-gradient(135deg, rgba(241, 196, 15, 0.15), rgba(241, 196, 15, 0.05));
    border: 1px solid rgba(241, 196, 15, 0.4);
}
.result-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.result-label { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.4rem; }
.result-confidence { font-size: 1.1rem; color: #cfcfcf; font-weight: 500; }
.confidence-bar-bg {
    background-color: rgba(255,255,255,0.1); border-radius: 20px;
    height: 10px; width: 100%; margin-top: 0.8rem; overflow: hidden;
}
.confidence-bar-fill { height: 100%; border-radius: 20px; }
.heatmap-caption { text-align: center; color: #999; font-size: 0.85rem; margin-top: 0.4rem; }
</style>
"""

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    return model, class_names

def preprocess_image(pil_image):
    img = pil_image.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def make_gradcam_heatmap(img_array, model, pred_index, last_conv_layer_name="resnet50"):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, pred_index]

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

def generate_pdf(original_img, heatmap_img, predicted_label, confidence, is_normal, is_uncertain):
    tmp_dir = tempfile.mkdtemp()
    orig_path = os.path.join(tmp_dir, "original.png")
    heat_path = os.path.join(tmp_dir, "heatmap.png")
    original_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)).save(orig_path)
    heatmap_img.save(heat_path)

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
    if is_uncertain:
        pdf.set_text_color(230, 180, 20)
        pdf.cell(0, 10, "Result: Unable to Determine", ln=True)
    elif is_normal:
        pdf.set_text_color(46, 204, 113)
        pdf.cell(0, 10, "Result: Looks Normal", ln=True)
    else:
        pdf.set_text_color(200, 60, 50)
        pdf.cell(0, 10, f"Result: {predicted_label}", ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 12)
    if not is_uncertain:
        pdf.cell(0, 8, f"Confidence: {confidence:.1f}%", ln=True)
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

    return bytes(pdf.output())

def open_browser_once():
    if "browser_opened" not in st.session_state:
        st.session_state.browser_opened = True
        threading.Timer(1.0, lambda: webbrowser.open_new(APP_URL)).start()

def main():
    st.set_page_config(page_title="EyeScan", page_icon="👁️", layout="centered")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    open_browser_once()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">👁️ EyeScan</div>
            <div class="hero-subtitle">Upload an eye photo to check for early signs of common conditions</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    model, class_names = load_model()

    uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        st.image(pil_image, width="stretch")

        with st.spinner("Analyzing..."):
            img_array = preprocess_image(pil_image)
            predictions = model.predict(img_array, verbose=0)[0]

            predicted_idx = int(np.argmax(predictions))
            predicted_class = class_names[predicted_idx]
            confidence = predictions[predicted_idx] * 100

            heatmap = make_gradcam_heatmap(img_array, model, predicted_idx)
            overlay_img = overlay_heatmap(pil_image, heatmap)

        is_uncertain = confidence < UNCERTAIN_THRESHOLD
        is_normal = (not is_uncertain) and predicted_class == "normal"

        if is_uncertain:
            st.markdown(
                """
                <div class="result-card result-uncertain">
                    <div class="result-icon">🤔</div>
                    <div class="result-label" style="color:#f1c40f;">Unable to Determine</div>
                    <div class="result-confidence">This image doesn't clearly match a recognized pattern</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.warning(
                "Please upload a clear, close-up eye/retinal image showing a single condition, "
                "similar to a standard clinical photo."
            )
        elif is_normal:
            st.markdown(
                """
                <div class="result-card result-normal">
                    <div class="result-icon">✅</div>
                    <div class="result-label" style="color:#2ecc71;">Looks Normal</div>
                    <div class="result-confidence">No significant signs detected</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            info = DISEASE_INFO.get(predicted_class, {"label": predicted_class, "icon": "⚠️", "color": "#e74c3c"})
            st.markdown(
                f"""
                <div class="result-card result-disease">
                    <div class="result-icon">{info['icon']}</div>
                    <div class="result-label" style="color:{info['color']};">{info['label']}</div>
                    <div class="result-confidence">{confidence:.0f}% confidence</div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill" style="width:{confidence:.0f}%; background-color:{info['color']};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_image.resize((IMG_SIZE, IMG_SIZE)), caption="Original")
        with col2:
            st.image(overlay_img, caption="Where the model looked")
        st.markdown(
            '<div class="heatmap-caption">Red/yellow areas influenced the prediction most</div>',
            unsafe_allow_html=True
        )

        pdf_bytes = generate_pdf(
            pil_image, overlay_img,
            DISEASE_INFO.get(predicted_class, {}).get("label", predicted_class),
            confidence, is_normal, is_uncertain
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"eyescan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            width="stretch"
        )

    st.markdown(
        "<div style='text-align:center; margin-top:2rem; color:#666; font-size:0.8rem;'>"
        "Not a medical diagnosis — please consult a doctor for confirmation."
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()