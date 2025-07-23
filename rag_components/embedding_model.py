from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
import config


class MultilingualE5Embeddings(HuggingFaceEmbeddings):
    """
    A custom embedding class for the multilingual-e5-large model.
    It automatically adds the required 'query: ' and 'passage: ' prefixes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(model_name=config.EMBEDDING_MODEL_NAME, *args, **kwargs)

    def _add_prefix(self, texts: List[str], prefix: str) -> List[str]:
        """Adds a prefix to each text in a list."""
        return [f"{prefix}{text}" for text in texts]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents for storage.
        Adds the 'passage: ' prefix before embedding.
        """
        prefixed_texts = self._add_prefix(texts, "passage: ")
        return super().embed_documents(prefixed_texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query for retrieval.
        Adds the 'query: ' prefix before embedding.
        """
        prefixed_text = f"query: {text}"
        return super().embed_query(prefixed_text)


def get_embedding_model():
    """Factory function to get our custom E5 embedding model."""
    return MultilingualE5Embeddings()
