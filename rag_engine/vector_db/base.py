from abc import ABC, abstractmethod
from typing import List
class BaseVectorDB(ABC):

    @abstractmethod
    def ingest_batch_documents(self, input_document: List):
        pass

    @abstractmethod
    def retrieve_data(self, k:int):
        pass