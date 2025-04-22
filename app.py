import streamlit as st
import google.generativeai as genai

# Page setup
st.set_page_config(page_title="Gemini Pro Chatbot", page_icon="🤖")

# Gemini API key from secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load Gemini Pro model
model = genai.GenerativeModel(model_name="models/gemini-pro")

# UI
st.title("🤖 Gemini Pro Chatbot")
prompt = st.text_input("💬 Ask me anything")

if prompt:
    with st.spinner("Generating response..."):
        try:
            response = model.generate_content(prompt)
            st.markdown("### 💡 Response")
            st.write(response.text)
        except Exception as e:
            st.error(f"⚠️ Chatbot error: {e}")
