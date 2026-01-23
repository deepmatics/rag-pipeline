from abc import ABC, abstractmethod
from typing import List

class BaseChunker(ABC):

    @abstractmethod
    def split(self, documents: List) -> List:
        pass