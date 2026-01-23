from langchain_huggingface import HuggingFaceEmbeddings
from rag_engine.embeddings.base import BaseEmbeddingModel

class LangChainEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str):
        """
        Initialize the HuggingFace embedding model via LangChain.
        """
        # We store the actual model instance in self.model
        self.model = HuggingFaceEmbeddings(model_name=model_name)
    
    def load(self):
        """
        Returns the internal LangChain object.
        """
        return self.model