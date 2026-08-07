import os
import urllib.request
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------
# Model Download Configuration (Original GitHub Release)
# ---------------------------------------------------------
MODEL_URL = "https://github.com/Pearandmeemo1234/Pneumonia-Detection/releases/download/v1.0/pneumonia_model.keras"
MODEL_PATH = "pneumonia_model.keras"

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detection AI",
    page_icon="🫁",
    layout="centered",
)


# ---------------------------------------------------------
# Load & Cache Model (Downloads from GitHub Release if missing)
# ---------------------------------------------------------
@st.cache_resource
def load_pneumonia_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model... Please wait a moment."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_pneumonia_model()
except Exception as e:
    st.error(f"❌ Failed to load model from Release: {e}")
    model = None

# ---------------------------------------------------------
# App Interface Header
# ---------------------------------------------------------
st.title("🫁 Chest X-Ray Pneumonia Detection AI")
st.write(
    "Upload a chest X-Ray image or use your camera to run an automated"
    " diagnostic analysis using MobileNetV2 deep learning model."
)

st.markdown("---")

# ---------------------------------------------------------
# Input Method Selection (Upload File vs. Take Photo)
# ---------------------------------------------------------
input_option = st.radio(
    "Select Input Method:",
    ("📁 Upload Chest X-Ray Image", "📷 Take Photo with Camera"),
)

uploaded_file = None

if input_option == "📁 Upload Chest X-Ray Image":
    uploaded_file = st.file_uploader(
        "Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"]
    )
else:
    uploaded_file = st.camera_input("Take a photo of the Chest X-Ray film")

# ---------------------------------------------------------
# Image Preprocessing & Prediction
# ---------------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Selected Chest X-Ray Image", use_column_width=True)

    if model is not None:
        with st.spinner("Analyzing Chest X-Ray... Please wait."):
            # Preprocessing
            img = image.convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            prediction = model.predict(img_array)[0][0]
            pneumonia_prob = float(prediction)
            normal_prob = 1.0 - pneumonia_prob

            st.markdown("---")
            st.subheader("Diagnostic Result")

            if pneumonia_prob > 0.50:
                st.error("🚩 **PNEUMONIA DETECTED**")
            else:
                st.success("✅ **NORMAL (NO PNEUMONIA)**")

            st.subheader("Confidence Score")

            st.write(f"**Normal:** {normal_prob * 100:.2f}%")
            st.progress(normal_prob)

            st.write(f"**Pneumonia:** {pneumonia_prob * 100:.2f}%")
            st.progress(pneumonia_prob)