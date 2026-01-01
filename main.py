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
    chunk_overlap=150
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


######################generation of answers for comparision#####################################

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
from ragas.run_config import RunConfig

df = pd.read_json("data/input/OpenRagbench/pdf/arxiv/queries.json", orient='index')
df = df.reset_index().rename(columns={'index': 'query_id'})

df_answers = pd.read_json("data/input/OpenRagBench/pdf/arxiv/answers.json", orient='index')
df_answers = df_answers.reset_index().rename(columns={'index': 'query_id', 0: 'answers'})

merged_df = pd.merge(df, df_answers, on='query_id')
merged_df = merged_df[merged_df['source'] == "text"].reset_index()
merged_df = merged_df.drop(['index'], axis = 1)

# 1. Chain Modification to get context
# The original rag_chain is good for simple invocation, but for evaluation we need the context.
rag_chain_input = {
    "context": itemgetter("query") | retriever,
    "question": itemgetter("query"),
    "query_id": itemgetter("query_id")
}
llm_chain = prompt | llm | StrOutputParser()

# Create a cleaning function
def extract_model_response(text):
    marker = "<start_of_turn>model\n"
    if marker in text:
        return text.split(marker)[1].strip()
    return text

def llm_chain_with_context(input_dict):
    raw_answer = llm_chain.invoke(input_dict)
    cleaned_answer = extract_model_response(raw_answer)
    return {
        "question": input_dict["question"],
        "query_id": input_dict["query_id"],
        "contexts": [doc.page_content for doc in input_dict["context"]], # Ragas expects "contexts"
        "section_id": [doc.metadata.get('section_id') for doc in input_dict["context"]],
        "answer": cleaned_answer,
    }

eval_chain = rag_chain_input | RunnableLambda(llm_chain_with_context)

# 2. Generate results with context
print("Generating answers and contexts for evaluation...")
queries = merged_df[['query_id', 'query']].to_dict('records')
results = eval_chain.batch(queries[0:2], {"max_concurrency": 5})

# Convert results to DataFrame
eval_results_df = pd.DataFrame(results)

# 3. Prepare dataset for Ragas
# Merge with ground truth answers
eval_df = pd.merge(eval_results_df, merged_df[['query_id', 'answers']], on='query_id')

# Rename columns for Ragas
eval_df = eval_df.rename(columns={"answers": "ground_truth"})
eval_df = eval_df[["question", "answer", "contexts", "ground_truth", "section_id"]]

# Ensure ground_truth is a list of strings
eval_df['ground_truth'] = eval_df['ground_truth'].apply(lambda x: x if isinstance(x, str) else x)

# Convert to Ragas dataset format
ragas_dataset = Dataset.from_pandas(eval_df)

# 4. Configure and run Ragas evaluation
print("Running Ragas evaluation...")

# Define metrics
metrics = [
    faithfulness,          # Is the answer grounded in the context?
    answer_relevancy,      # Is the answer relevant to the question?
    context_recall,        # Does the context contain the ground truth?
    context_precision,     # Is the context relevant to the question?
]

# Wrap your LangChain LLM and Embeddings for Ragas
ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

# Configure metrics with your models
for m in metrics:
    m.llm = ragas_llm
    if hasattr(m, "embeddings"):
        m.embeddings = ragas_embeddings

# 3. Configure sequential execution
run_config = RunConfig(max_workers=1, timeout=300)

# 4. Run evaluate with NO individual metric overrides
result = evaluate(
    dataset=ragas_dataset,
    metrics=metrics,
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=run_config,
    raise_exceptions=False,
    allow_nest_asyncio=False
)

print(result.to_pandas())
# Save results to CSV
result.to_csv("data/output/OpenRagBench/ragas_results.csv", index=False)
print("Ragas evaluation results saved to output_data/ragas_evaluation_results.csv")