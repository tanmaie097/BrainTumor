import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
import requests
from duckduckgo_search import DDGS

# ===============================
# Config
# ===============================
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")

# ===============================
# Hugging Face + DuckDuckGo Search Chatbot
# ===============================
HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"  # ✅ Zephyr model (open access)
HF_HEADERS = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

def search_web(query, num_results=5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, region='wt-wt', safesearch='Moderate', max_results=num_results):
            snippet = r.get("body", "") or r.get("description", "")
            if snippet:
                results.append(f"- {snippet}")
    return results

def chat_with_zephyr(question):
    snippets = search_web(question)
    if not snippets:
        return "No web results found. Try rephrasing your question."

    context = "\n".join(snippets)
    prompt = f"""[INST] You are a helpful assistant like Perplexity AI. Use the web information below to answer the user's question clearly.

Web context:
{context}

Question: {question}
[/INST]"""
    try:
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": prompt})
        response.raise_for_status()
        return response.json()[0]["generated_text"].split("[/INST]")[-1].strip()
    except Exception as e:
        return f"⚠️ Chatbot error: {e}"

# ===============================
# Load Tumor Detection Model
# ===============================
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

# ===============================
# Sidebar: Zephyr Chatbot
# ===============================
st.sidebar.markdown("### 🤖 Web-Powered Assistant")
user_input = st.sidebar.text_input("Ask anything (real-time info)")

if user_input:
    st.sidebar.markdown("*Searching + Generating Answer...*")
    reply = chat_with_zephyr(user_input)
    st.sidebar.write(reply)

# ===============================
# UI + Theme
# ===============================
if 'history' not in st.session_state:
    st.session_state.history = []

theme_mode = st.sidebar.radio("🌗 Theme", ("Light", "Dark"))
if theme_mode == "Dark":
    st.markdown("<style>.stApp { background-color: #0E1117; color: #FAFAFA; }</style>", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align: center; color: #6a0dad;'>🧠 Brain Tumor Classification</h1>
<p style='text-align: center;'>Upload an MRI image to detect brain tumors using AI</p>
""", unsafe_allow_html=True)

# ===============================
# Image Upload + Prediction
# ===============================
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

# ===============================
# Prediction History
# ===============================
with st.expander("🕓 View Prediction History"):
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"{i}. **{entry['image']}** — *{entry['prediction']}*, Confidence: {entry['confidence']}")
    else:
        st.write("No history yet.")

st.markdown("---")
st.caption("🔍 Note: This tool is for educational/demo purposes only. Always consult a specialist for medical advice.")
