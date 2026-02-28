from faiss import IndexFlatL2
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from rag_engine.vector_db.base import BaseVectorDB

class Langchainfaiss(BaseVectorDB):

    def __init__(self, embedding_func, persist_dir: str):
        self.client = FAISS(
            embedding_function=embedding_func,
            index=IndexFlatL2(1536),
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
        )

    def ingest_batch_documents(self, batch):
        return self.client.add_documents(batch)
    
    def retrieve_data(self, k):
        return self.client.as_retriever(search_kwargs = {"k" : k})
    