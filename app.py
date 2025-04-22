import streamlit as st
import google.generativeai as genai

# ====================
# Gemini Setup
# ====================
st.set_page_config(page_title="Gemini Pro Chatbot", page_icon="🧠")

# Load API key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Initialize Gemini model
model = genai.GenerativeModel(model_name="models/gemini-pro")

# ====================
# Streamlit UI
# ====================
st.title("🤖 Gemini Pro Chatbot")
st.markdown("Ask me anything — I'm powered by Google's Gemini Pro.")

prompt = st.text_input("💬 Your question")

if prompt:
    with st.spinner("Generating response..."):
        try:
            response = model.generate_content(prompt)
            st.markdown("### 💡 Response")
            st.write(response.text)
        except Exception as e:
            st.error(f"⚠️ Chatbot error: {e}")
