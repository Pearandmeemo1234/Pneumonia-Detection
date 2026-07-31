import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. ตั้งค่าหน้าตาเว็บ (Page Title & Icon)
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detection AI",
    page_icon="🫁",
    layout="centered"
)

# 2. โหลดโมเดลแบบใช้ Cache
@st.cache_resource
def load_pneumonia_model():
    return tf.keras.models.load_model('pneumonia_model.keras')

model = load_pneumonia_model()

# 3. หัวข้อและรายละเอียดภาษาอังกฤษ
st.title("🫁 Chest X-Ray Pneumonia Detection AI")
st.write("Upload a chest X-Ray image to run an automated diagnostic analysis using MobileNetV2 deep learning model.")

st.markdown("---")

# 4. กล่องอัปโหลดรูปภาพ
uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chest X-Ray Image", use_column_width=True)
    
    with st.spinner("Analyzing Chest X-Ray... Please wait."):
        img = image.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
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