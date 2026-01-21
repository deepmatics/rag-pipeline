import json
import os
import glob
from rag_engine.data_loader.base import BaseDataLoader
from langchain_core.documents import Document

class OpenRagBenchJSON(BaseDataLoader):
    def __init__(self, path: str):
        self.path = path

    @staticmethod
    def load(path: str) -> list[Document]:
        docs = []
        for file_path in glob.glob(os.path.join(path, "*.json")):
            with open(file_path, 'r') as f:
                file_data = json.load(f)
                for section in file_data.get('sections', []):
                    if 'text' in section and 'section_id' in section:
                        docs.append(Document(
                            page_content=section['text'],
                            metadata={"source": os.path.basename(file_path), "section_id": section['section_id']}
                        ))

        return docs