import streamlit as st
import google.generativeai as genai

# ===============================
# Page config
# ===============================
st.set_page_config(page_title="Gemini Chatbot", page_icon="💬", layout="centered")

# ===============================
# Configure Gemini
# ===============================
if "GEMINI_API_KEY" not in st.secrets:
    st.error("🔑 Please add your Gemini API key to Streamlit secrets.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ===============================
# Initialize Gemini Model
# ===============================
try:
    model = genai.GenerativeModel("gemini-pro")
except Exception as e:
    st.error(f"❌ Failed to load Gemini model: {e}")
    st.stop()

# ===============================
# Streamlit UI
# ===============================
st.title("🤖 Gemini Pro Chatbot")
st.markdown("Ask me anything! I'm powered by Google's Gemini model.")

user_input = st.text_input("💬 Type your message:")

if user_input:
    with st.spinner("Thinking..."):
        try:
            response = model.generate_content(user_input)
            st.markdown("### 💡 Response")
            st.write(response.text)
        except Exception as e:
            st.error(f"⚠️ Chatbot error: {e}")

st.markdown("---")
st.caption("✨ Powered by Google Gemini via Generative AI API")
