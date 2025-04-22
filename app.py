import streamlit as st
import requests

# ----------------------------
# Hugging Face Setup
# ----------------------------
HF_API_URL = "https://api-inference.huggingface.co/models/bigscience/bloomz-560m"
HF_HEADERS = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

# ----------------------------
# Function to generate reply
# ----------------------------
def ask_bloom(prompt):
    try:
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": prompt})
        response.raise_for_status()
        return response.json()[0]["generated_text"]
    except Exception as e:
        return f"⚠️ Chatbot error: {e}"

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Free Chatbot", page_icon="💬")
st.title("🤖 Free & Friendly Chatbot")
st.markdown("Ask me anything! I’ll try to answer using a free Hugging Face model.")

user_input = st.text_input("💬 Your question")

if user_input:
    with st.spinner("Thinking..."):
        answer = ask_bloom(user_input)
        st.markdown("### 💡 Answer")
        st.write(answer)

st.markdown("---")
st.caption("🔓 Powered by `bigscience/bloomz-560m` via Hugging Face")
