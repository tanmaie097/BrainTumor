import streamlit as st
import requests

# -------------------------------
# Config
# -------------------------------
st.set_page_config(page_title="Groq AI Chatbot", page_icon="🤖", layout="centered")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# -------------------------------
# Chat Function
# -------------------------------
def chat_with_groq(prompt):
    payload = {
        "model": "llama3-8b-8192",  # other option: "mixtral-8x7b-32768"
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(GROQ_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Chatbot error: {e}"

# -------------------------------
# UI
# -------------------------------
st.title("🤖 Groq LLaMA 3 Chatbot")
st.markdown("Ask me anything. I'm running on lightning-fast Groq hardware ⚡")

user_input = st.text_input("💬 Your message")

if user_input:
    with st.spinner("Thinking..."):
        answer = chat_with_groq(user_input)
        st.markdown("### 💡 Answer")
        st.write(answer)

st.markdown("---")
st.caption("🚀 Powered by Groq + Meta's LLaMA 3")
