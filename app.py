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
# Custom CSS Theme (Fixed for Dark Mode & High Contrast)
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* 1. บังคับพื้นหลังหลักของเว็บ */
    .stApp {
        background: linear-gradient(180deg, #eefbfb 0%, #f7fdfd 100%) !important;
    }

    /* 2. บังคับสีตัวหนังสือทั่วไป ไม่ว่าจะเปิด Dark Mode หรือ Light Mode */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
        color: #004d40 !important;
    }

    /* 3. หัวข้อหลักและหัวข้อย่อย */
    h1 {
        color: #005f60 !important;
        font-weight: 800 !important;
    }
    h2, h3, h4 {
        color: #00796b !important;
        font-weight: 700 !important;
    }

    /* 4. กล่องตัวเลือก Radio และ File Uploader */
    div[data-testid="stFileUploader"], div[role="radiogroup"] {
        background-color: #ffffff !important;
        padding: 20px;
        border-radius: 16px;
        border: 1.5px solid #7FFFD4 !important;
        box-shadow: 0 4px 15px rgba(0, 206, 209, 0.08);
    }

    /* บังคับสีตัวหนังสือในกล่องให้เข้มชัดเจนเสมอ */
    div[role="radiogroup"] label *, div[data-testid="stFileUploader"] label * {
        color: #004d40 !important;
        font-weight: 600 !important;
    }

    /* 5. ปรับแต่งพื้นที่อัปโหลดไฟล์ (Dropzone) ไม่ให้ดำใน Dark Mode */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #f0fdfa !important;
        border: 2px dashed #00CED1 !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #005f60 !important;
    }

    /* 6. ปุ่มกด */
    .stButton>button {
        background-color: #00CED1 !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 10px 24px !important;
        font-weight: bold !important;
        box-shadow: 0 4px 10px rgba(0, 206, 209, 0.25);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00b4b7 !important;
        box-shadow: 0 6px 14px rgba(0, 206, 209, 0.4);
        transform: translateY(-2px);
    }

    /* 7. Progress bar */
    .stProgress > div > div > div > div {
        background-color: #00CED1 !important;
    }

    /* 8. เส้นแบ่งโซน */
    hr {
        border-top: 2px solid #7FFFD4 !important;
        opacity: 0.6;
    }
    </style>
""", unsafe_allow_html=True)


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
    st.image(
        image, caption="Selected Chest X-Ray Image", use_container_width=True
    )

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