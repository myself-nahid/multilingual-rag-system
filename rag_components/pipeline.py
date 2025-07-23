from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import config
from .data_loader import get_vector_store

def create_rag_chain():
    """
    Creates the complete, non-conversational RAG chain.
    """
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": config.RETRIEVER_K})

    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL_NAME, temperature=0)

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question based only on the following context.
    If the answer is not available in the context, say "The answer is not available in the provided document."
    Provide a concise answer in the same language as the user's question.

    Context:
    {context}

    Question:
    {input}

    Answer:
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    print("RAG chain created successfully.")
    return retrieval_chain