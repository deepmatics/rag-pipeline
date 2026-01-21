import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from tqdm import tqdm
from flow_judge import Hf, FlowJudge, EvalInput, list_all_metrics
from flow_judge.metrics import RESPONSE_FAITHFULNESS_5POINT, RESPONSE_CORRECTNESS_5POINT, RESPONSE_RELEVANCE_5POINT
import chromadb
from chromadb.config import Settings

# Preparing the benchmark dataset
df = pd.read_json("data/input/OpenRagbench/pdf/arxiv/queries.json", orient="index")
df = df.reset_index().rename(columns={"index": "query_id"})

df_answers = pd.read_json("data/input/OpenRagBench/pdf/arxiv/answers.json", orient="index")
df_answers = df_answers.reset_index().rename(columns={"index": "query_id", 0: "answers"})
df_sec_id = pd.read_json("data/input/OpenRagBench/pdf/arxiv/qrels.json", orient="index")
df_sec_id = df_sec_id.reset_index().rename(columns={"index": "query_id", "section_id": "answer_section"})

merged_df = pd.merge(df, df_answers, on="query_id")
merged_df = merged_df[merged_df["source"] == "text"].reset_index()
merged_df = merged_df.drop(["index"], axis = 1)

embedding_model = HuggingFaceEmbeddings(model_name="/Users/syenwc6b/.cache/huggingface/hub/models--google--embeddinggemma-300m/snapshots/ea082e2d5ef48d95602e8589f0ae7c2799987143" )
llm_model = "/Users/syenwc6b/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/fb45cab2e2d05204ac7e12f3051d144981d59f41"

llm_pipeline = HuggingFacePipeline.from_model_id(
    model_id=llm_model,
    task="text-generation" # ,
    #pipeline_kwargs={"max_new_tokens": 512, "temperature": 0.1}
)
llm = ChatHuggingFace(llm=llm_pipeline)

def format_docs(docs):
    return " \n\n ".join(doc.page_content for doc in docs)

client = chromadb.Client(Settings(anonymized_telemetry=False))
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory="./chroma_db/OpenRagBench"
)

template = """
Using the information contained in the context, give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
If the answer cannot be deduced from the context, do not give an answer.

Context:
{context}

Now here is the question you need to answer.

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_response(row):

    query = row["query"]
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(query)
    section_ids = [doc.metadata.get("section_id") for doc in retrieved_docs]
    
    rag_chain = (
        {"context": lambda x: format_docs(retrieved_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    llm_response = rag_chain.invoke(query)
    return pd.Series([format_docs(retrieved_docs), llm_response, section_ids])

tqdm.pandas(desc="RAG Evaluation Progress")
merged_df[["retrieved_docs", "llm_response", "section_id"]] = merged_df.progress_apply(get_rag_response, axis=1)

def extract_gemma_response(raw_text):
    """
    Extracts only the assistant's response from a Gemma 3 formatted string.
    """
    marker = "<start_of_turn>model"
    
    if marker in raw_text:
        response = raw_text.split(marker)[-1]
        response = response.replace("<end_of_turn>", "")
        return response.strip()
    
    return raw_text # Return original if marker isn't found

# Clean the responses in your dataframe
merged_df['llm_response'] = merged_df['llm_response'].apply(extract_gemma_response)

merged_df.to_csv("data/output/OpenRagbench/llm_responses.csv", index=False)

############################## Evaluations Start Here ##############################
df = pd.read_csv('data/output/OpenRagBench/llm_responses.csv')
df = df.merge(df_sec_id, on="query_id")

def calculate_rag_metrics(row, k=3):
    ground_truth = str(row['answer_section'])
    # Ensure retrieved_ids is a list of strings for comparison
    retrieved_ids = [str(i) for i in row['section_id']] 
    
    # 1. Check for a "Hit"
    hit = 1 if ground_truth in retrieved_ids else 0
    
    # 2. Precision: How much of the context was useful?
    # (Since only 1 is correct, max precision is 1/k)
    precision = hit / k
    
    # 3. Recall: Did we get the right document?
    # (Since only 1 exists, recall is binary: 1.0 or 0.0)
    recall = float(hit)
    
    # 4. MRR: How high up in the top 3 was it?
    mrr = 0.0
    if hit:
        rank = retrieved_ids.index(ground_truth) + 1
        mrr = 1.0 / rank

    return precision, recall, mrr

# Apply to your merged_df
df[['precision', 'recall', 'mrr']] = df.apply(
    lambda row: calculate_rag_metrics(row), 
    axis=1, 
    result_type='expand'
)

# 1. Overall Averages
avg_precision = df['precision'].mean()
avg_recall = df['recall'].mean()

# 2. Mean Reciprocal Rank (MRR) - Very important for RAG
mrr_score = df['mrr'].mean()

# 3. F1-Score (The Harmonic Mean of the averages)
# This balances the trade-off between precision and recall
f1_overall = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)

print(f"--- RAG System Performance ---")
print(f"Mean Precision: {avg_precision:.4f}")
print(f"Mean Recall:    {avg_recall:.4f}")
print(f"Overall F1:     {f1_overall:.4f}")
print(f"Overall MRR:    {mrr_score:.4f}")


######################## LLM as a Judge Start Here ##########################
model = Hf(flash_attn=False, local_files_only=True, device_map="mps")

faithfulness_judge = FlowJudge(
    metric=RESPONSE_FAITHFULNESS_5POINT,
    model=model
)

correctness_judge = FlowJudge(
    metric=RESPONSE_CORRECTNESS_5POINT,
    model=model
)

relevance_judge = FlowJudge(
    metric=RESPONSE_RELEVANCE_5POINT,
    model=model
)

def get_evals(row):

    query = str(row["query"])
    context = str(row["retrieved_docs"])
    response = str(row["llm_response"])
    
    eval_input = EvalInput(
        inputs=[
            {"query": query},
            {"context": context},
        ],
        output={"response": response},
    )

    # Run the evaluation
    faithfullness = faithfulness_judge.evaluate(eval_input, save_results=False)
    relevance = faithfulness_judge.evaluate(eval_input, save_results=False)
    correctness = faithfulness_judge.evaluate(eval_input, save_results=False)
    
    fj= faithfullness.feedback
    rj= relevance.feedback
    cj= correctness.feedback
    fs= faithfullness.score
    rs= relevance.score
    cs= correctness.score
 
    return pd.Series([fj, fs, cj, cs, rj, rs])

tqdm.pandas(desc="RAG Evaluation Progress")
df[["faithfullness", "f-score", "correctness", "c-score", "relevance", "r-score"]] = df.progress_apply(get_evals, axis=1)