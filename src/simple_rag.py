import os
import json
import glob
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import torch

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Path where your JSON files are
folder_path = "./data/input/OpenRagBench/pdf/arxiv/corpus/"
all_docs = []

for file_path in glob.glob(os.path.join(folder_path, "*.json")):
    with open(file_path, 'r') as f:
        file_data = json.load(f)
        for section in file_data.get('sections', []):
            if 'text' in section and 'section_id' in section:
                all_docs.append(Document(
                    page_content=section['text'],
                    metadata={"source": file_path, "section_id": section['section_id']}
                ))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    add_start_index=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

final_splits = text_splitter.split_documents(all_docs)
print(f"Split documents into {len(final_splits)} chunks for indexing.")

# Create the DB
embedding_model = HuggingFaceEmbeddings(model_name="/Users/syenwc6b/.cache/huggingface/hub/models--google--embeddinggemma-300m/snapshots/ea082e2d5ef48d95602e8589f0ae7c2799987143" )

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

