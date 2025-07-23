import sys
from getpass import getpass
from rag_components.data_loader import load_and_vectorize_pdf
from rag_components.pipeline import create_rag_chain
from rag_components.memory import create_conversational_rag_chain
import config

def check_api_key():
    if not config.GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY environment variable not found.")
        print("Please create a .env file and add your key, or set it in your environment.")

        config.GOOGLE_API_KEY = getpass("Enter your Google API Key: ")
        if not config.GOOGLE_API_KEY:
            sys.exit("API Key is required to run the application.")

def main():
    check_api_key()
    load_and_vectorize_pdf()

    rag_chain = create_conversational_rag_chain()

    print("\n--- Multilingual RAG Chatbot ---")
    print("Ask a question in English or Bangla. Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = rag_chain.invoke({"question": user_input})
        
        print(f"AI: {response['answer']}")


if __name__ == "__main__":
    main()