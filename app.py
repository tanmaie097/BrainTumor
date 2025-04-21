import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
import requests

# ================================
# Mistral 7B Chatbot Config (Free)
# ================================
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HF_HEADERS = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

def chat_with_mistral(prompt):
    try:
        payload = {"inputs": f"[INST] {prompt} [/INST]"}
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)
        response.raise_for_status()
        return response.json()[0]["generated_text"]
    except Exception as e:
        return f"⚠️ Chatbot error: {e}"

# ================================
# Streamlit Config
# ================================
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")

@st.cache_resource
def load_model():
    model_path = 'brain_tumor_model.h5'
    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading model..."):
            url = 'https://drive.google.com/uc?id=13wz4umsZx-UgPBdYGxSxNmXA1hLLVmEG'
            gdown.download(url, model_path, fuzzy=True, quiet=False)
    return tf.keras.models.load_model(model_path)

model = load_model()
classes = ['No Tumor', 'Pituitary Tumor']

# Sidebar - Mistral Chatbot
st.sidebar.markdown("### 🤖 Mistral Chat Assistant")
user_input = st.sidebar.text_input("Ask me anything")

if user_input:
    st.sidebar.markdown("*Mistral says:*")
    st.sidebar.write(chat_with_mistral(user_input))

# Theme Toggle
if 'history' not in st.session_state:
    st.session_state.history = []

theme_mode = st.sidebar.radio("🌗 Theme", ("Light", "Dark"))
if theme_mode == "Dark":
    st.markdown(
        "<style>.stApp { background-color: #0E1117; color: #FAFAFA; }</style>",
        unsafe_allow_html=True
    )

# Title and Header
st.markdown("""
<h1 style='text-align: center; color: #6a0dad;'>🧠 Brain Tumor Classification</h1>
<p style='text-align: center;'>Upload an MRI image to detect brain tumors using AI</p>
""", unsafe_allow_html=True)

# File Upload
uploaded_file = st.file_uploader("📤 Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True, caption="Uploaded Image")

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

# Prediction History
with st.expander("🕓 View Prediction History"):
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"{i}. **{entry['image']}** — *{entry['prediction']}*, Confidence: {entry['confidence']}")
    else:
        st.write("No history yet.")

st.markdown("---")
st.caption("🔍 Note: This tool is for educational/demo purposes only. Always consult a specialist for medical advice.")
