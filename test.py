import os
import tempfile
import streamlit as st
import google.generativeai as genai
import docx2txt
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader

# Configure Gemini API
genai.configure(api_key=st.secrets["AIzaSyARc-6LVuLXB1VEcwUed6cEdCK_8tf7s_0"])

st.set_page_config(page_title="📄 Ask Your Document", layout="centered")
st.title("📄 Gemini Document Q&A (No Embedding)")
st.caption("Lightweight version without FAISS")

uploaded_file = st.file_uploader("📎 Upload a PDF or Word Document", type=["pdf", "doc", "docx"])
question = st.text_input("🔎 Ask a question from the document")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    try:
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
        elif uploaded_file.name.endswith((".doc", ".docx")):
            text = docx2txt.process(temp_path)
            documents = [Document(page_content=text)]
        else:
            st.error("Unsupported file format.")
            st.stop()

        st.success("✅ Document loaded!")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.2,
            convert_system_message_to_human=True,
        )

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an AI assistant. Use the following document context to answer the user's question.
Context:
{context}

Question:
{question}

Answer:
"""
        )

        chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

        if question:
            with st.spinner("💬 Generating answer..."):
                result = chain.run(input_documents=documents, question=question)
                st.success("✅ Answer ready!")
                st.markdown("### 📘 Answer:")
                st.write(result)

    except Exception as e:
        st.error(f"🚨 Error: {str(e)}")
