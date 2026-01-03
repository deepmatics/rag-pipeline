import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Preparing the benchmark dataset
df = pd.read_json("data/input/OpenRagbench/pdf/arxiv/queries.json", orient="index")
df = df.reset_index().rename(columns={"index": "query_id"})

df_answers = pd.read_json("data/input/OpenRagBench/pdf/arxiv/answers.json", orient="index")
df_answers = df_answers.reset_index().rename(columns={"index": "query_id", 0: "answers"})

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

vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory="./chroma_db/OpenRagBench"
)

template = """
Using the information contained in the context, give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.

Context:
{context}

Now here is the question you need to answer.

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

def get_rag_response(row):
    # Use the 'question' column from your merged_df
    query = row["query"]
    
    # 1. Setup retriever for this specific query
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 2. Get documents to extract metadata (section_ids)
    retrieved_docs = retriever.invoke(query)
    
    # Extract section_id from metadata; handle missing keys with .get()
    # This creates a list like [5, 21, 14]
    section_ids = [doc.metadata.get("section_id") for doc in retrieved_docs]

    # 3. Create the chain (as you defined it)
    # We use a custom format function to join contents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": lambda x: format_docs(retrieved_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 4. Generate response
    # Since we already retrieved docs manually for section_ids, 
    # we just pass the query to the chain
    llm_response = rag_chain.invoke(query)

    # Return as a Series so pandas can split it into two columns
    return pd.Series([llm_response, section_ids])

tqdm.pandas(desc="RAG Evaluation Progress")
merged_df[["llm_response", "section_id"]] = merged_df.progress_apply(get_rag_response, axis=1)

merged_df.to_csv("data/output/OpenRagbench/llm_responses.csv", index=False)

tokenizer = AutoTokenizer.from_pretrained("flowaicom/Flow-Judge-v0.1", trust_remote_code=False, local_files_only = True)
model = AutoModelForCausalLM.from_pretrained("flowaicom/Flow-Judge-v0.1", trust_remote_code=False, local_files_only = True)
messages = [
    {"role": "user", "content": "What is your purpose??"},
]

inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

from flow_judge import Hf, FlowJudge, EvalInput, list_all_metrics

df = pd.read_csv("data/output/OpenRagBench/ragas_results.csv")

# Now try initializing again
model = Hf(flash_attn=False, local_files_only=True, device_map=None)
from flow_judge.metrics import RESPONSE_FAITHFULNESS_5POINT

faithfulness_judge = FlowJudge(
    metric=RESPONSE_FAITHFULNESS_5POINT,
    model=model
)

# Sample to evaluate
query = df["user_input"][1]
context = df["retrieved_contexts"][1]
response = df["response"][1]

# Create an EvalInput
# We want to evaluate the response to the customer issue based on the context and the user instructions
eval_input = EvalInput(
    inputs=[
        {"query": query},
        {"context": context},
    ],
    output={"response": response},
)

# Run the evaluation
score_faithfullness = faithfulness_judge.evaluate(eval_input, save_results=False)














########################## Hugging Face Code #################################################

import datasets
from langchain_core.vectorstores import VectorStore
from typing import Optional, List, Tuple
import json
import tqdm
from langchain_core.documents import Document as LangchainDocument
import os
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
import glob

repo_id = "HuggingFaceH4/zephyr-7b-beta"
READER_MODEL_NAME = "zephyr-7b-beta"

# Since HF_API_TOKEN is empty, I'm assuming the user wants to use a local model.
# I'll use a placeholder for the model path, which the user should replace with the correct path.
# Following the pattern in simple_rag.py and existing eval.py
# Using the same model as in simple_rag.py.
from langchain.chat_models import init_chat_model
READER_LLM = init_chat_model("/Users/syenwc6b/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/fb45cab2e2d05204ac7e12f3051d144981d59f41", model_provider="huggingface")


RAG_PROMPT_TEMPLATE = """
<|system|>
Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.</s>
<|user|>
Context:
{context}
---
Now here is the question you need to answer.

Question: {question}
</s>
<|assistant|>
"""

RAW_KNOWLEDGE_BASE = "data/input/OpenRagBench/pdf/arxiv/corpus/"

def load_knowledge_base(
    folder_path: str,
) -> List[LangchainDocument]:
    all_docs = []
    for file_path in glob.glob(os.path.join(folder_path, "*.json")):
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            for section in file_data.get('sections', []):
                if 'text' in section and 'section_id' in section:
                    all_docs.append(LangchainDocument(
                        page_content=section['text'],
                        metadata={"source": file_path, "section_id": section['section_id']}
                    ))
    return all_docs

def load_embeddings(
    knowledge_base_docs: List[LangchainDocument],
    chunk_size: int,
    embedding_model_name: str,
) -> VectorStore:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    final_splits = text_splitter.split_documents(knowledge_base_docs)
    
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    vectorstore = Chroma.from_documents(
        documents=final_splits,
        embedding=embedding_model,
        persist_directory=f"./chroma_db/OpenRagBench_{chunk_size}_{embedding_model_name.replace('/', '~')}"
    )
    return vectorstore

def answer_with_rag(
    question: str,
    llm,
    knowledge_index: VectorStore,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 7,
) -> Tuple[str, List[LangchainDocument]]:
    """Answer a question using RAG with the given knowledge index."""
    # Gather documents with retriever
    relevant_docs = knowledge_index.similarity_search(
        query=question, k=num_retrieved_docs
    )
    
    relevant_docs_content = [doc.page_content for doc in relevant_docs]
    relevant_docs_content = relevant_docs_content[:num_docs_final]


    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs_content)]
    )

    prompt = PromptTemplate(template=RAG_PROMPT_TEMPLATE, input_variables=["context", "question"])
    chain = prompt | llm
    answer = chain.invoke({"context": context, "question": question})

    return answer, relevant_docs_content

def run_rag_tests(
    eval_dataset: datasets.Dataset,
    llm,
    knowledge_index: VectorStore,
    output_file: str,
    verbose: Optional[bool] = True,
    test_settings: Optional[str] = None,  # To document the test settings used
):
    """Runs RAG tests on the given dataset and saves the results to the given output file."""
    try:  # load previous generations if they exist
        with open(output_file, "r") as f:
            outputs = json.load(f)
    except:
        outputs = []

    for example in tqdm.tqdm(eval_dataset):
        question = example["question"]
        if question in [output["question"] for output in outputs]:
            continue

        answer, relevant_docs = answer_with_rag(
            question, llm, knowledge_index
        )
        if verbose:
            print("=======================================================")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f'True answer: {example["answer"]}')
        result = {
            "question": question,
            "true_answer": example["answer"],
            "generated_answer": answer,
            "retrieved_docs": [doc for doc in relevant_docs],
        }
        if test_settings:
            result["test_settings"] = test_settings
        outputs.append(result)

        with open(output_file, "w") as f:
            json.dump(outputs, f)


EVALUATION_PROMPT = """###Task Description:
An instruction, a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage


evaluation_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a fair evaluator language model."),
        HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT),
    ]
)

eval_chat_model = init_chat_model("/Users/syenwc6b/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/fb45cab2e2d05204ac7e12f3051d144981d59f41", model_provider="huggingface")
evaluator_name = "gemma3"

def evaluate_answers(
    answer_path: str,
    eval_chat_model,
    evaluator_name: str,
    evaluation_prompt_template: ChatPromptTemplate,
) -> None:
    """Evaluates generated answers. Modifies the given answer file in place for better checkpointing."""
    answers = []
    if os.path.isfile(answer_path):  # load previous generations if they exist
        answers = json.load(open(answer_path, "r"))

    for experiment in tqdm(answers):
        if f"eval_score_{evaluator_name}" in experiment:
            continue

        eval_prompt = evaluation_prompt_template.format_messages(
            instruction=experiment["question"],
            response=experiment["generated_answer"],
            reference_answer=experiment["true_answer"],
        )
        eval_result = eval_chat_model.invoke(eval_prompt)
        feedback, score = [
            item.strip() for item in eval_result.content.split("[RESULT]")
        ]
        experiment[f"eval_score_{evaluator_name}"] = score
        experiment[f"eval_feedback_{evaluator_name}"] = feedback

        with open(answer_path, "w") as f:
            json.dump(answers, f)

if not os.path.exists("./output"):
    os.mkdir("./output")

# Load the dataset
df = pd.read_json("data/input/OpenRagbench/pdf/arxiv/queries.json", orient='index')
df = df.reset_index().rename(columns={'index': 'query_id'})

df_answers = pd.read_json("data/input/OpenRagBench/pdf/arxiv/answers.json", orient='index')
df_answers = df_answers.reset_index().rename(columns={'index': 'query_id', 0: 'answers'})

merged_df = pd.merge(df, df_answers, on='query_id')
merged_df = merged_df[merged_df['source'] == "text"].reset_index()
merged_df = merged_df.drop(['index'], axis = 1)
merged_df = merged_df.rename(columns={'answers': 'answer', 'query': 'question'})
print(merged_df.columns)
eval_dataset = datasets.Dataset.from_pandas(merged_df)


knowledge_base_docs = load_knowledge_base(RAW_KNOWLEDGE_BASE)

rerank = False

for chunk_size in [200]:  # Add other chunk sizes (in tokens) as needed
    for embeddings in ["thenlper/gte-small"]:  # Add other embeddings as needed
        settings_name = f"chunk:{chunk_size}_embeddings:{embeddings.replace('/', '~')}_reader-model:{READER_MODEL_NAME}"
        output_file_name = f"./output/rag_{settings_name}.json"

        print(f"Running evaluation for {settings_name}:")

        print("Loading knowledge base embeddings...")
        knowledge_index = load_embeddings(
            knowledge_base_docs,
            chunk_size=chunk_size,
            embedding_model_name=embeddings,
        )

        print("Running RAG...")
        run_rag_tests(
            eval_dataset=eval_dataset,
            llm=READER_LLM,
            knowledge_index=knowledge_index,
            output_file=output_file_name,
            verbose=False,
            test_settings=settings_name,
        )

        print("Running evaluation...")
        evaluate_answers(
            output_file_name,
            eval_chat_model,
            evaluator_name,
            evaluation_prompt_template,
        )