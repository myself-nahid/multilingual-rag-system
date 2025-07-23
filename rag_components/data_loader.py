import os
import pdfplumber
from langchain_community.vectorstores import FAISS
from .embedding_model import get_embedding_model
from .smart_chunker import get_semantic_chunks_from_page
import config


def load_and_vectorize_pdf():
    """
    Loads the PDF using pdfplumber and then uses an LLM to perform
    semantic chunking into paragraphs for maximum precision.
    """
    if os.path.exists(config.VECTOR_STORE_PATH):
        print("Vector store already exists. To re-build, please delete the 'vector_store' directory.")
        return

    if not os.path.exists(config.PDF_PATH):
        raise FileNotFoundError(
            f"The PDF file was not found at {config.PDF_PATH}")

    print(f"Loading document with pdfplumber from: {config.PDF_PATH}")
    all_semantic_chunks = []
    with pdfplumber.open(config.PDF_PATH) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                print(f"Processing page {i+1} with semantic chunker...")
                page_chunks = get_semantic_chunks_from_page(
                    page_text, page_number=i+1)
                all_semantic_chunks.extend(page_chunks)

    print(f"Total semantic chunks created: {len(all_semantic_chunks)}")

    print("Creating embeddings and vector store with multilingual-e5-large...")
    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(all_semantic_chunks, embeddings)

    vector_store.save_local(config.VECTOR_STORE_PATH)
    print(f"Vector store created and saved at {config.VECTOR_STORE_PATH}")


def get_vector_store():
    if not os.path.exists(config.VECTOR_STORE_PATH):
        raise FileNotFoundError(
            "Vector store not found. Please run main.py to create it.")

    embeddings = get_embedding_model()
    return FAISS.load_local(
        config.VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
