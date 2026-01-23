from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_engine.chunking.base import BaseChunker

class LangChainRecursive(BaseChunker):
    def __init__(self, chunk_config: dict):
        self.chunk_size = chunk_config.get("size", 1000)
        self.chunk_overlap = chunk_config.get("overlap", 200)
        
        # Initialize the underlying LangChain tool
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def split(self, documents: List) -> List:
        if not documents:
            return []
            
        splits = self.splitter.split_documents(documents)
        return splits