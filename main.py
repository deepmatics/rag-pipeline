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
embedding_model = HuggingFaceEmbeddings(model_name="/Users/syenwc6b/.cache/huggingface/hub/models--google--embeddinggemma-300m/snapshots/ea082e2d5ef48d95602e8589f0ae7c2799987143" )

vectorstore = Chroma(
    embedding_function=embedding_model,
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

retriever = vectorstore.as_retriever()
template = """Answer the question based only on the following context:
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

query = "How is second-order smoothness achieved in Tikhonov regularization?"

print("🤔 RAG Chain Thinking...")
response = rag_chain.invoke(query)
print(response)


######################generation of answers for comparision#####################################

df = pd.read_json("data/pdf/arxiv/queries.json", orient='index')
df = df.reset_index().rename(columns={'index': 'query_id'})

df_answers = pd.read_json("data/pdf/arxiv/answers.json", orient='index')
df_answers = df_answers.reset_index().rename(columns={'index': 'query_id', 0: 'answers'})

merged_df = pd.merge(df, df_answers, on='query_id')
merged_df = merged_df[merged_df['source'] == "text"].reset_index()

results = rag_chain.batch(merged_df['query'].tolist(), {"max_concurrency": 5})

# 3. Assign the full list to your new column
merged_df['llm_response'] = results

merged_df.columns

# Create a cleaning function
def extract_model_response(text):
    marker = "<start_of_turn>model\n"
    if marker in text:
        # Split at the marker and take the part at index 1 (the text after)
        # .strip() removes any leading/trailing whitespace
        return text.split(marker)[1].strip()
    return text # Return original if marker not found

# Apply it to your DataFrame
df['llm_response'] = df['llm_response'].apply(extract_model_response)

# Preview the cleaned data
print(merged_df['llm_response'].iloc[0])

merged_df.to_csv('output_data/responses.csv')


##################### Experimenting with Ragas##############
import re
import json
import torch
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

df = pd.read_csv("data/output_data/OpenRagBench/responses.csv")

class Qwen3DeepEval(DeepEvalBaseLLM):
    def __init__(self, model_name="Qwen/Qwen3-4B-Instruct-2507"):
        self.model_name = model_name
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype="auto",
            local_files_only=True
        )
        self.pipeline = pipeline(
            "text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        print("\n--- DEBUG: PROMPT SENT TO MODEL ---")
        print(prompt)
        
        results = self.pipeline(prompt, max_new_tokens=512, temperature=0.1) # Lower temp for better JSON
        output = results[0]['generated_text']
        
        print("\n--- DEBUG: RAW OUTPUT FROM MODEL ---")
        print(output)
        
        return output

    async def a_generate(self, prompt: str) -> str:
        # DeepEval is async-first, so we implement this
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

qwen_model = Qwen3DeepEval(model_name="Qwen/Qwen3-4B-Instruct-2507")

# ---------------------------------------------------------
# 2. Define the Metrics
# ---------------------------------------------------------
# Faithfulness: Checks if the answer is derived from retrieval context
# Answer Relevancy: Checks if the answer actually addresses the query
faithfulness_metric = FaithfulnessMetric(
    threshold=0.7, 
    model=qwen_model
)

relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7, 
    model=qwen_model
)

# ---------------------------------------------------------
# 3. Create a Test Case and Evaluate
# ---------------------------------------------------------
# Pick a row from your merged_df
sample_row = df.iloc[0]

test_case = LLMTestCase(
    input=sample_row['query'],
    actual_output=sample_row['llm_response'],
    retrieval_context=[sample_row['source']], # Pass the text from the vector store
    expected_output=sample_row['answers']      # The ground truth
)

# Run evaluation
faithfulness_metric.measure(test_case)
print(f"Faithfulness Score: {faithfulness_metric.score}")
print(f"Reason: {faithfulness_metric.reason}")

relevancy_metric.measure(test_case)
print(f"Relevancy Score: {relevancy_metric.score}")