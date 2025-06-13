import streamlit as st
import tempfile
import google.generativeai as genai
import docx2txt
from pypdf import PdfReader

# Set Gemini API Key from Streamlit secrets
genai.configure(api_key=st.secrets["AIzaSyARc-6LVuLXB1VEcwUed6cEdCK_8tf7s_0"])

# Streamlit UI
st.set_page_config(page_title="Gemini Q&A", layout="centered")
st.title("📄 Ask Questions from Your Document")
st.caption("Powered by Google Gemini 2.0 Flash")

# Upload + Input
uploaded_file = st.file_uploader("📎 Upload PDF or DOCX", type=["pdf", "docx"])
question = st.text_input("💬 Enter your question:")

# Process file & answer
if uploaded_file and question:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    try:
        # Read file content
        if uploaded_file.name.endswith(".pdf"):
            reader = PdfReader(temp_path)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        else:
            text = docx2txt.process(temp_path)

        st.success("✅ Document processed!")

        # Gemini model call
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
Document:
{text}

Question:
{question}

Answer:
"""
        with st.spinner("🤖 Generating answer..."):
            response = model.generate_content(prompt)
            st.success("✅ Answer:")
            st.write(response.text)

    except Exception as e:
        st.error(f"🚨 Error: {e}")
