import tempfile
import streamlit as st
import google.generativeai as genai
import docx2txt
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

st.set_page_config(page_title="Document Q&A", layout="centered")
st.title("📄 Gemini Document Q&A – Minimal")

uploaded = st.file_uploader("Upload PDF or DOCX", type=["pdf", "doc", "docx"])
question = st.text_input("Ask your question:")

if uploaded and question:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded.name) as tmp:
        tmp.write(uploaded.getvalue())
        path = tmp.name

    try:
        if uploaded.name.endswith(".pdf"):
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text = docx2txt.process(path)

        st.success("📄 Document loaded!")

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
        prompt = f"""
Document:
{text}

Question:
{question}

Answer:"""

        with st.spinner("Generating answer..."):
            answer = llm(prompt)
            st.success("✅ Here’s the answer:")
            st.write(answer["content"])

    except Exception as e:
        st.error(f"Error: {e}")
