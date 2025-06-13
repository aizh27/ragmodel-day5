import os
import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ====== Set your Gemini API Key here ======
GOOGLE_API_KEY = "AIzaSyBZVFzrY9P7nz4XZP7vHQTCmWpbPNbqkec"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ====== Streamlit UI ======
st.set_page_config(page_title="Ask Your Document", layout="centered")
st.title("📄 Ask Questions from Your Document using Gemini")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "doc", "docx"])

question = st.text_input("❓ Enter your question")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    # Load the document
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif uploaded_file.name.endswith((".doc", ".docx")):
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        st.error("Unsupported file type.")
        st.stop()

    docs = loader.load()

    # Create embeddings & vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever()

    # Set up the Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        convert_system_message_to_human=True,
    )

    # Define custom prompt
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant. Use the following context from the uploaded document to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # Create RAG pipeline
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": custom_prompt}
    )

    if question:
        with st.spinner("🔍 Finding the answer..."):
            result = qa_chain.run(question)
            st.success("✅ Answer retrieved successfully!")
            st.markdown("### 📘 Answer:")
            st.write(result)
