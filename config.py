import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

PDF_PATH = os.path.join("data", "HSC26-Bangla1st-Paper.pdf")
VECTOR_STORE_PATH = "vector_store" # Path to save/load FAISS index

EMBEDDING_MODEL_NAME = "models/text-embedding-004"
LLM_MODEL_NAME = "gemini-2.5-flash"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 250
RETRIEVER_K = 5 # Number of chunks to retrieve