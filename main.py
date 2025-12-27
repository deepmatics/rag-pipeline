import os
import json
import glob
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import torch

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool

# Path where your JSON files are
folder_path = "./data/pdf/arxiv/corpus/" 
all_docs = []

for file_path in glob.glob(os.path.join(folder_path, "*.json")):
    with open(file_path, 'r') as f:
        file_data = json.load(f)
        text_list = [item['text'] for item in file_data.get('sections', []) if 'text' in item]
        combined_text = "\n\n".join(text_list)
        all_docs.append(Document(combined_text, metadata={"source": file_path}))

# Create a splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # 1000 characters per chunk
    chunk_overlap=100  # 100 characters overlap so context isn't lost
)

# Split the documents
final_splits = text_splitter.split_documents(all_docs)
print(f"Split {len(all_docs)} documents into {len(final_splits)} chunks.")

# Create the DB
hf = HuggingFaceEmbeddings(model_name="/Users/syenwc6b/.cache/huggingface/hub/models--google--embeddinggemma-300m/snapshots/ea082e2d5ef48d95602e8589f0ae7c2799987143" )

vectorstore = Chroma(
    embedding_function=hf,
    persist_directory="./chroma_db"
)

batch_size = 300
total_chunks = len(final_splits)

with tqdm(total = total_chunks, desc= "Indexing chunks", unit = "chunks") as pbar:
    for i in range(0, total_chunks, batch_size):
        batch = final_splits[i : i + batch_size]
        vectorstore.add_documents(batch)
        torch.mps.empty_cache()
        pbar.update(len(batch))

llm_model = "/Users/syenwc6b/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/fb45cab2e2d05204ac7e12f3051d144981d59f41"

llm = init_chat_model(llm_model, model_provider="huggingface")

@tool
def search_knowledge_base(query: str) -> str:
    """Useful for answering questions about RAG, RMSE, and MLMM."""
    results = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in results])

agent = create_agent(
    model=llm,
    tools=[search_knowledge_base],
    system_prompt="You are a helpful AI assistant. Use the search_knowledge_base tool to answer questions. Be concise and answer only what has been asked."
)

query = "How is second-order smoothness achieved in Tikhonov regularization?"

print("🤔 Agent Thinking...")
response = agent.invoke({"messages": [{"role": "user", "content": query}]})

print(response["messages"][-1].content)