import streamlit as st
import requests
from duckduckgo_search import DDGS

# ===============================
# Page Config
# ===============================
st.set_page_config(page_title="Web-Powered Chatbot", page_icon="🤖", layout="centered")

# ===============================
# Hugging Face Zephyr LLM Setup
# ===============================
HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"
HF_HEADERS = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

# ===============================
# Search + Generate Answer
# ===============================
def search_web(query, num_results=5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, region='wt-wt', safesearch='Moderate', max_results=num_results):
            snippet = r.get("body", "") or r.get("description", "")
            if snippet:
                results.append(f"- {snippet}")
    return results

def generate_answer(question):
    snippets = search_web(question)
    if not snippets:
        return "No relevant web results found."

    context = "\n".join(snippets)
    prompt = f"""[INST] You are a helpful assistant. Use the following web search context to answer the user's question clearly.

Context:
{context}

Question: {question}
[/INST]"""

    try:
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": prompt})
        response.raise_for_status()
        return response.json()[0]["generated_text"].split("[/INST]")[-1].strip()
    except Exception as e:
        return f"⚠️ Error: {e}"

# ===============================
# Streamlit Chatbot UI
# ===============================
st.title("🤖 Web-Powered AI Chatbot")

st.markdown("Ask me anything! I’ll search the web and give a helpful answer.")

user_input = st.text_input("💬 Enter your question")

if user_input:
    with st.spinner("Searching and generating answer..."):
        response = generate_answer(user_input)
        st.markdown("### 💡 Answer")
        st.write(response)

    with st.expander("🔍 Web Sources Used"):
        for snippet in search_web(user_input):
            st.markdown(snippet)

st.markdown("---")
st.caption("🚀 Powered by DuckDuckGo + Zephyr LLM on Hugging Face")
