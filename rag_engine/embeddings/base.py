from abc import ABC, abstractmethod
from typing import Any

class BaseEmbeddingModel(ABC):
    
    @abstractmethod
    def load(self) -> Any:
        """Abstract method to load and return the embedding model."""
        pass