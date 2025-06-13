import os
import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Set Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBZVFzrY9P7nz4XZP7vHQTCmWpbPNbqkec"

st.set_page_config(page_title="📄 Ask Your Document | Gemini 2.0 Flash", layout="centered")
st.title("📄 Ask Questions from Your Document")
st.caption("Powered by Google Gemini 2.0 Flash + LangChain")

# Cache embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load Gemini LLM
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        convert_system_message_to_human=True,
    )

uploaded_file = st.file_uploader("📎 Upload a document (.pdf, .doc, .docx)", type=["pdf", "doc", "docx"])
question = st.text_input("❓ Ask a question based on your document")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    # Load document
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif uploaded_file.name.endswith((".doc", ".docx")):
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        st.error("Unsupported file type.")
        st.stop()

    docs = loader.load()
    st.success("✅ Document loaded successfully!")

    embeddings = load_embeddings()

    @st.cache_resource
    def create_vectorstore(documents):
        return FAISS.from_documents(documents, embeddings)

    vectorstore = create_vectorstore(docs)
    retriever = vectorstore.as_retriever()

    llm = load_llm()

    # Prompt Template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use the following context extracted from a document to answer the user's question.
Context:
{context}

Question:
{question}

Answer:"""
    )

    # Create Retrieval QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    if question:
        with st.spinner("🤖 Generating answer..."):
            response = qa_chain.run(question)
            st.success("✅ Answer generated!")
            st.markdown("### 📘 Answer:")
            st.write(response)
