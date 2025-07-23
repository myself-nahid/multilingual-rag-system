import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

PDF_PATH = os.path.join("data", "HSC26-Bangla1st-Paper.pdf")
VECTOR_STORE_PATH = "vector_store"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
LLM_MODEL_NAME = "gemini-2.5-flash"

CHUNK_SIZE = 750
CHUNK_OVERLAP = 150
RETRIEVER_K = 7

API_HOST = "127.0.0.1"
API_PORT = 8000
