import torch
from rag_engine.utils.config_loader import ConfigLoader
from rag_engine.data_loader import OpenRagBenchJSON
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

rag_config = ConfigLoader.load_yaml("config/rag.yaml")

docs = OpenRagBenchJSON.load(rag_config["input_data"]["path"])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    add_start_index=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

final_splits = text_splitter.split_documents(docs)
print(f"Split documents into {len(final_splits)} chunks for indexing.")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Create the DB
embedding_model = HuggingFaceEmbeddings(model_name=rag_config["hf"]["embeddings"] )

vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory="./chroma_db/OpenRagBench"
)

batch_size = 100
total_chunks = len(final_splits)

with tqdm(total = total_chunks, desc= "Indexing chunks", unit = "chunks") as pbar:
    for i in range(0, total_chunks, batch_size):
        batch = final_splits[i : i + batch_size]
        vectorstore.add_documents(batch)
        torch.mps.empty_cache()
        pbar.update(len(batch))

llm_model = "/Users/syenwc6b/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/fb45cab2e2d05204ac7e12f3051d144981d59f41"

llm = init_chat_model(llm_model, model_provider="huggingface")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
template = """Answer the question based only on the following context shared. 
1. be crisp with your answers:
2. if the context doesn't have the answer, then call out that you don't know
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


#################################testing for a single query#####################################
query = "How is second-order smoothness achieved in Tikhonov regularization?"

print("🤔 RAG Chain Thinking...")
response = rag_chain.invoke(query)
print(response)

