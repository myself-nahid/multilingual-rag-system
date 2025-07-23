from fastapi import FastAPI, Depends
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from rag_components.pipeline import create_rag_chain

app = FastAPI(
    title="Multilingual RAG API",
    description="An API for querying the HSC Bangla literature RAG system."
)

rag_chain = None

@app.on_event("startup")
def startup_event():
    global rag_chain
    rag_chain = create_rag_chain()

def get_rag_chain():
    return rag_chain

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome to the RAG API. Go to /docs for the API interface."}

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def handle_query(request: QueryRequest, chain=Depends(get_rag_chain)):
    """
    Accepts a query and returns a grounded answer from the document.
    """
    response = await chain.ainvoke({"input": request.query})
    return QueryResponse(answer=response['answer'])