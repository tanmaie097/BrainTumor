import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
import openai  # for real AI assistant

# MUST be the first Streamlit command
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")

# =========================
# Load model (download if needed)
# =========================
@st.cache_resource
def load_model():
    model_path = 'brain_tumor_model.h5'
    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading brain tumor detection model..."):
            url = 'https://drive.google.com/uc?id=13wz4umsZx-UgPBdYGxSxNmXA1hLLVmEG'
            gdown.download(url, model_path, fuzzy=True, quiet=False)
    return tf.keras.models.load_model(model_path)

model = load_model()
classes = ['No Tumor', 'Pituitary Tumor']

# =========================
# OpenAI AI Chat Assistant
# =========================
st.sidebar.markdown("### 🤖 AI Assistant")
user_input = st.sidebar.text_input("Ask me anything")

def chat_with_gpt(prompt):
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

if user_input:
    with st.sidebar:
        st.markdown("*AI Response:*")
        try:
            st.write(chat_with_gpt(user_input))
        except Exception as e:
            st.error("⚠️ Error connecting to OpenAI. Check your API key.")

# =========================
# UI Theme & Title
# =========================
if 'history' not in st.session_state:
    st.session_state.history = []

theme_mode = st.sidebar.radio("🌗 Theme", ("Light", "Dark"))

if theme_mode == "Dark":
    st.markdown(
        "<style>.stApp {background-color: #0E1117; color: #FAFAFA;}</style>",
        unsafe_allow_html=True
    )

st.markdown("""
<h1 style='text-align: center; color: #6a0dad;'>🧠 Brain Tumor Classification</h1>
<p style='text-align: center;'>Upload an MRI image to detect brain tumors using AI</p>
""", unsafe_allow_html=True)

# =========================
# File Upload & Prediction
# =========================
uploaded_file = st.file_uploader("📤 Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.markdown("### 🖼 Uploaded Image")
    st.image(image, use_column_width=True)

    image = image.resize((224, 224))
    img_array = np.array(image)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    with col1:
        if predicted_class == 0:
            st.success("✅ *No Tumor Detected*")
        else:
            st.error("⚠ *Pituitary Tumor Detected*")
    with col2:
        st.markdown("*Confidence Level:*")
        st.progress(int(confidence * 100))
        st.write(f"{confidence * 100:.2f}%")

    st.session_state.history.append({
        "image": uploaded_file.name,
        "prediction": classes[predicted_class],
        "confidence": f"{confidence * 100:.2f}%"
    })

# =========================
# Prediction History
# =========================
with st.expander("🕓 View Prediction History"):
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"{i}. **{entry['image']}** — Prediction: *{entry['prediction']}*, Confidence: {entry['confidence']}")
    else:
        st.write("No history yet.")

st.markdown("---")
st.caption("🔍 Note: This tool is for educational/demo purposes only. Always consult a specialist for medical advice.")
