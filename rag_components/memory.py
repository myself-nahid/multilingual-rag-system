from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from .data_loader import get_vector_store
import config
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


def create_conversational_rag_chain():
    """
    Creates a powerful RAG chain that uses a MultiQueryRetriever for high-precision
    retrieval and supports conversational memory.
    """
    vector_store = get_vector_store()
    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL_NAME, temperature=0)

    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": config.RETRIEVER_K})

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

    print("Upgraded to MultiQueryRetriever.")

    memory = ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=multi_query_retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )

    print("Conversational RAG chain created successfully.")
    return conversational_chain
