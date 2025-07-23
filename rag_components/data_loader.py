import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import config

def load_and_vectorize_pdf():
    """
    Loads the PDF, splits it into chunks, creates embeddings,
    and saves the vector store to disk.
    """
    if os.path.exists(config.VECTOR_STORE_PATH):
        print("Vector store already exists. Skipping creation.")
        return

    print(f"Loading document from: {config.PDF_PATH}")
    if not os.path.exists(config.PDF_PATH):
        raise FileNotFoundError(f"The PDF file was not found at {config.PDF_PATH}")

    loader = PyPDFLoader(config.PDF_PATH)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    split_docs = text_splitter.split_documents(docs)

    print("Creating embeddings and vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)
    vector_store = FAISS.from_documents(split_docs, embeddings)

    vector_store.save_local(config.VECTOR_STORE_PATH)
    print(f"Vector store created and saved at {config.VECTOR_STORE_PATH}")

def get_vector_store():
    """
    Loads the FAISS vector store from the local path.
    """
    if not os.path.exists(config.VECTOR_STORE_PATH):
        raise FileNotFoundError("Vector store not found. Please run the data loading process first.")
        
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)
    return FAISS.load_local(
        config.VECTOR_STORE_PATH, 
        embeddings,
        allow_dangerous_deserialization=True 
    )