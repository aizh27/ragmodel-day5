import streamlit as st
import tempfile
import google.generativeai as genai
import docx2txt
from pypdf import PdfReader

# Set Gemini API Key from secrets
genai.configure(api_key=st.secrets["AIzaSyARc-6LVuLXB1VEcwUed6cEdCK_8tf7s_0"])

st.set_page_config(page_title="📄 Gemini Doc Q&A", layout="centered")
st.title("📄 Ask Questions from Your Document")

uploaded_file = st.file_uploader("📎 Upload PDF or DOCX file", type=["pdf", "docx"])
question = st.text_input("🔍 Enter your question:")

if uploaded_file and question:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    try:
        if uploaded_file.name.endswith(".pdf"):
            reader = PdfReader(path)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        else:
            text = docx2txt.process(path)

        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        prompt = f"""
You are a helpful assistant. Use the document below to answer the question.

Document:
{text}

Question:
{question}

Answer:
"""
        with st.spinner("Thinking..."):
            response = model.generate_content(prompt)
            st.success("✅ Answer:")
            st.write(response.text)

    except Exception as e:
        st.error(f"❌ Error: {e}")
