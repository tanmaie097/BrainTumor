import streamlit as st
import openai

# =====================
# Streamlit Config
# =====================
st.set_page_config(page_title="ChatGPT AI Assistant", page_icon="🤖", layout="centered")

# =====================
# OpenAI API Setup
# =====================
openai.api_key = st.secrets["OPENAI_API_KEY"]

def ask_chatgpt(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Or gpt-4 if you have access
            messages=[
                {"role": "system", "content": "You are a helpful and intelligent assistant."},
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
        response = ask_chatgpt(user_input)
        st.markdown("### 💡 Answer")
        st.write(response)

st.markdown("---")
st.caption("🔐 Powered by OpenAI | Model: GPT-3.5-Turbo")
