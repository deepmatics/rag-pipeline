from langchain_chroma import Chroma
from rag_engine.vector_db.base import BaseVectorDB

class LangChainChroma(BaseVectorDB):

    def __init__(self, embedding_func, persist_dir: str):
        self.client = Chroma(
            embedding_function=embedding_func,
            persist_directory=persist_dir
        )

    def ingest_batch_documents(self, batch):
        return self.client.add_documents(batch)
    
    def retrieve_data(self, k):
        return self.client.as_retriever(search_kwargs = {"k" : k})
    