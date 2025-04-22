import streamlit as st
from openai import OpenAI

# =====================
# Config
# =====================
st.set_page_config(page_title="ChatGPT Assistant", page_icon="🤖", layout="centered")

# =====================
# OpenAI Setup
# =====================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def ask_chatgpt(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error: {e}"

# =====================
# UI
# =====================
st.title("🤖 ChatGPT Assistant")
st.markdown("Ask me anything!")

user_input = st.text_input("💬 Your question")

if user_input:
    with st.spinner("Thinking..."):
        answer = ask_chatgpt(user_input)
        st.markdown("### 💡 Answer")
        st.write(answer)

st.markdown("---")
st.caption("🔐 Powered by OpenAI | Model: GPT-3.5 Turbo")
