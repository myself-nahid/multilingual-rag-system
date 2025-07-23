from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from .data_loader import get_vector_store
import config

def create_conversational_rag_chain():
    """
    Creates a RAG chain that supports conversational memory.
    """
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL_NAME, temperature=0.1)

    memory = ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history",
        return_messages=True
    )

    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False 
    )
    print("Conversational RAG chain created successfully.")
    return conversational_chain