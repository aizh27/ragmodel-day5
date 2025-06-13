import os
import tempfile
import streamlit as st
import google.generativeai as genai

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader

# ✅ Setup Gemini API key from secrets (Streamlit Cloud compatible)
genai.configure(api_key=st.secrets["AIzaSyARc-6LVuLXB1VEcwUed6cEdCK_8tf7s_0"])

# Page setup
st.set_page_config(page_title="📄 Ask Your Document", layout="centered")
st.title("📄 Document Q&A with Gemini 2.0 Flash")
st.caption("Upload a document and ask questions using Google Gemini and LangChain")

# Load embedding model once
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load Gemini model once
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        convert_system_message_to_human=True,
    )

# File uploader
uploaded_file = st.file_uploader("📎 Upload a PDF, DOC or DOCX file", type=["pdf", "doc", "docx"])
question = st.text_input("🔎 Ask a question from the document")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    # Load file using proper loader
    try:
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        elif uploaded_file.name.endswith((".doc", ".docx")):
            loader = UnstructuredWordDocumentLoader(temp_path)
        else:
            st.error("Unsupported file format.")
            st.stop()

        with st.spinner("📚 Reading document..."):
            documents = loader.load()
            st.success("✅ Document loaded")

        # Vector store setup
        embeddings = load_embeddings()

        @st.cache_resource
        def create_vectorstore(docs):
            return FAISS.from_documents(docs, embeddings)

        vectorstore = create_vectorstore(documents)
        retriever = vectorstore.as_retriever()

        llm = load_llm()

        # Prompt setup
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Use the provided context to answer the question below.
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
            with st.spinner("🔍 Generating answer..."):
                answer = chain.run(question)
                st.success("✅ Answer generated!")
                st.markdown("### 📘 Answer:")
                st.write(answer)

    except Exception as e:
        st.error(f"🚨 Error: {str(e)}")
