import streamlit as st
import requests

# ===============================
# Page Config
# ===============================
st.set_page_config(page_title="FLAN-T5 Chatbot", page_icon="🤖", layout="centered")

# ===============================
# Hugging Face FLAN-T5 Setup
# ===============================
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HF_HEADERS = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

def ask_flan(prompt):
    try:
        payload = {"inputs": prompt}
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)
        response.raise_for_status()
        return response.json()[0]["generated_text"]
    except Exception as e:
        return f"⚠️ Error: {e}"

# ===============================
# Streamlit Chatbot UI
# ===============================
st.title("🤖 FLAN-T5 AI Chatbot")

st.markdown("Ask me anything! I’ll respond using the FLAN-T5 model from Hugging Face 🤖")

user_input = st.text_input("💬 Enter your question")

if user_input:
    with st.spinner("Thinking..."):
        answer = ask_flan(user_input)
        st.markdown("### 💡 Answer")
        st.write(answer)

st.markdown("---")
st.caption("🚀 Powered by FLAN-T5 on Hugging Face Inference API")
