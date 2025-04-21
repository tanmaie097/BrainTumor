import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
import google.generativeai as genai  # Gemini!

# Streamlit page setup
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")

# =============== Gemini Setup ================
if "GEMINI_API_KEY" not in st.secrets:
    st.sidebar.error("❌ Gemini API key missing. Add it to Streamlit secrets.")
else:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel("gemini-pro")

# =============== Model Loader ================
@st.cache_resource
def load_model():
    model_path = "brain_tumor_model.h5"
    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading model..."):
            url = "https://drive.google.com/uc?id=13wz4umsZx-UgPBdYGxSxNmXA1hLLVmEG"
            gdown.download(url, model_path, fuzzy=True, quiet=False)
    return tf.keras.models.load_model(model_path)

model = load_model()
classes = ['No Tumor', 'Pituitary Tumor']

# =============== Sidebar AI Chatbot ================
st.sidebar.markdown("### 🤖 Gemini Assistant")
user_input = st.sidebar.text_input("Ask me anything")

if user_input:
    try:
        gemini_response = gemini_model.generate_content(user_input)
        st.sidebar.markdown("*Gemini says:*")
        st.sidebar.write(gemini_response.text)
    except Exception as e:
        st.sidebar.error("⚠️ Error using Gemini API. Check key or try again later.")

# =============== Theme and UI ================
if 'history' not in st.session_state:
    st.session_state.history = []

theme = st.sidebar.radio("🌗 Theme", ("Light", "Dark"))
if theme == "Dark":
    st.markdown("""
    <style>.stApp { background-color: #0E1117; color: #FAFAFA; }</style>
    """, unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align: center; color: #6a0dad;'>🧠 Brain Tumor Classification</h1>
<p style='text-align: center;'>Upload an MRI image to detect brain tumors using AI</p>
""", unsafe_allow_html=True)

# =============== Image Upload & Prediction ================
uploaded_file = st.file_uploader("📤 Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
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

# =============== Prediction History ================
with st.expander("🕓 View Prediction History"):
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"{i}. **{entry['image']}** — Prediction: *{entry['prediction']}*, Confidence: {entry['confidence']}")
    else:
        st.write("No history yet.")

st.caption("🔍 Note: This tool is for educational/demo purposes only. Always consult a specialist for medical advice.")
