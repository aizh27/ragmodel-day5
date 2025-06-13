import os
import tempfile
import streamlit as st
import google.generativeai as genai
import docx2txt

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document

# ✅ Configure Gemini API Key securely from Streamlit Secrets
genai.configure(api_key=st.secrets["AIzaSyARc-6LVuLXB1VEcwUed6cEdCK_8tf7s_0"])

# UI Setup
st.set_page_config(page_title="📄 Ask Your Document", layout="centered")
st.title("📄 Document Q&A with Gemini 2.0 Flash")
st.caption("Upload a document and ask questions using Google Gemini and LangChain")

# Embedding Model (cached)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Gemini LLM (cached)
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        convert_system_message_to_human=True,
    )

# Upload and process
uploaded_file = st.file_uploader("📎 Upload a PDF, DOC or DOCX file", type=["pdf", "doc", "docx"])
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

        with st.spinner("📚 Loading document..."):
            st.success("✅ Document processed!")

        # Vector DB
        embeddings = load_embeddings()

        @st.cache_resource
        def create_vectorstore(docs):
            return FAISS.from_documents(docs, embeddings)

        vectorstore = create_vectorstore(documents)
        retriever = vectorstore.as_retriever()

        llm = load_llm()

        # Custom prompt
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Use the following context to answer the question.
Context:
{context}

Question:
{question}

Answer:
"""
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )

        if question:
            with st.spinner("🤖 Generating answer..."):
                answer = chain.run(question)
                st.success("✅ Answer generated!")
                st.markdown("### 📘 Answer:")
                st.write(answer)

    except Exception as e:
        st.error(f"🚨 Error: {str(e)}")
