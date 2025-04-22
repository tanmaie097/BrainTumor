from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests

# 🔍 Get web results
def search_web(query, num_results=5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, region='wt-wt', safesearch='Moderate', max_results=num_results):
            snippet = r.get("body", "") or r.get("description", "")
            if snippet:
                results.append(f"- {snippet}")
    return results

# 🧠 Generate response using Hugging Face (Mistral)
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HF_HEADERS = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

def generate_answer(context, question):
    try:
        prompt = f"""[INST] You are a helpful assistant like Perplexity AI. Use the info below to answer the user's question clearly.

Context:
{context}

Question: {question}
[/INST]"""
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": prompt})
        response.raise_for_status()
        return response.json()[0]["generated_text"].split("[/INST]")[-1].strip()
    except Exception as e:
        return f"⚠️ Error: {e}"

# 🧠 Streamlit Sidebar Assistant
st.sidebar.markdown("### 🤖 Web-Powered Assistant")
user_question = st.sidebar.text_input("Ask anything (real-time info)")

if user_question:
    with st.sidebar:
        st.markdown("*Searching the web...*")
        snippets = search_web(user_question)
        if not snippets:
            st.error("No results found.")
        else:
            context = "\n".join(snippets)
            st.markdown("*Generating answer...*")
            response = generate_answer(context, user_question)
            st.write(response)
