import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pneumonia Detection AI", page_icon="🫁", layout="centered"
)

st.title("🫁 AI Chest X-Ray Pneumonia Detection System")
st.write(
    "A deep learning web application to classify chest X-ray images into Normal"
    " or Pneumonia."
)


# ---------------------------------------------------------
# 2. Load AI Model
# ---------------------------------------------------------
@st.cache_resource
def load_pneumonia_model():
    model = tf.keras.models.load_model("pneumonia_model.h5")
    return model


try:
    model = load_pneumonia_model()
    st.success("✅ AI Model Loaded Successfully!")
except Exception as e:
    st.error(
        "❌ Model file 'pneumonia_model.h5' not found. Please check your"
        " directory."
    )

st.divider()

# ---------------------------------------------------------
# 3. Input Selection (Upload File vs. Take Photo)
# ---------------------------------------------------------
option = st.radio(
    "Select input method:",
    ("📁 Upload Image File", "📷 Take a Photo via Camera"),
)

uploaded_image = None

if option == "📁 Upload Image File":
    uploaded_image = st.file_uploader(
        "Choose a Chest X-Ray scan (JPG, PNG)", type=["jpg", "jpeg", "png"]
    )
else:
    uploaded_image = st.camera_input("Take a photo of the X-ray film")

# ---------------------------------------------------------
# 4. Image Preprocessing & Prediction
# ---------------------------------------------------------
if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Selected X-Ray Image", use_container_width=True)

    if st.button("🔍 Analyze with AI", type="primary"):
        with st.spinner("Analyzing image..."):

            # --- Step A: Data Preprocessing ---
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # --- Step B: Prediction ---
            prediction = model.predict(img_array)
            probability = float(prediction[0][0])

            st.divider()
            st.subheader("📊 Diagnostic Results")

            # --- Step C: Display Results ---
            if probability > 0.5:
                p_percent = probability * 100
                st.error("⚠️ **Risk Detected: Pneumonia**")
                st.metric("Confidence Score", f"{p_percent:.2f}%")
            else:
                n_percent = (1 - probability) * 100
                st.success("🎉 **Diagnosis: Normal (Healthy)**")
                st.metric("Confidence Score", f"{n_percent:.2f}%")

            st.caption(
                "Disclaimer: This AI system provides preliminary screening"
                " results only and should not replace professional medical"
                " diagnosis by a radiologist."
            )