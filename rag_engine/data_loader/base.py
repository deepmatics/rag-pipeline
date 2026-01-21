from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseDataLoader(ABC):
    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        pass
    

