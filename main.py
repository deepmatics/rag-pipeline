import os
import torch
from tqdm import tqdm
from rag_engine.utils.config_loader import YamlFile
from rag_engine.data_loader import OpenRagBenchJSON
from rag_engine.vector_db.chroma import LangchainChroma
from rag_engine.chunking.recursive import LangchainRecursive
from rag_engine.embeddings.langchain import LangchainEmbeddingModel
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from typing import List, Dict
from rank_bm25 import BM25Okapi

rag_config = YamlFile.load("config/rag.yaml")
prompts_template = YamlFile.load("rag_engine/prompts/rag.yaml")

INPUT_DATA_PATH = rag_config["input_data"]["path"]
EMBEDDING_MODEL = os.path.expanduser(rag_config["hf"]["embeddings"])
LLM_MODEL = os.path.expanduser(rag_config["hf"]["llm_model"])
DB_BATCH_SIZE = rag_config["vector_db"]["batch_size"]
DB_PERSIST_DIR = rag_config["vector_db"]["persist_dir"]
SYSTEM_PROMPT = prompts_template["system"]

docs = OpenRagBenchJSON.load(INPUT_DATA_PATH)

embedding = LangchainEmbeddingModel(model_name=EMBEDDING_MODEL)
llm = init_chat_model(LLM_MODEL, model_provider="huggingface")

chunker = LangchainRecursive(rag_config["chunking"])
final_splits = chunker.split(docs)

vectorstore = LangchainChroma(
    embedding_func=embedding,
    persist_dir= DB_PERSIST_DIR
)

with tqdm(total = len(final_splits), desc= "Indexing chunks", unit = "chunks") as pbar:
    for i in range(0, len(final_splits), DB_BATCH_SIZE):
        batch = final_splits[i : i + DB_BATCH_SIZE]
        vectorstore.ingest_batch_documents(batch)
        torch.mps.empty_cache()
        pbar.update(len(batch))

bm25_corpus = [doc.page_content for doc in final_splits]

retriever = vectorstore.retrieve_data(k=3)

prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":

    query = "How is second-order smoothness achieved in Tikhonov regularization?"

    print("🤔 RAG Chain Thinking...")
    response = rag_chain.invoke(query)
    print(response)

