#https://github.com/FareedKhan-dev/complex-RAG-guide/blob/main/RAG_pipeline.ipynb

import yaml
from langchain_core.prompts import PromptTemplate
from time import monotonic
from dotenv import load_dotenv
from pprint import pprint
import os
from datasets import Dataset
from typing_extensions import TypedDict
from typing import List, TypedDict
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    faithfulness,
    answer_relevancy,
    context_recall,
    answer_similarity
)

from utils.helper import generate_summary_with_hf_model

from utils.pdf import (
    read_from_url,
    split_into_chapters,
    extract_book_quotes_as_documents
)
from utils.text import (
    replace_t_with_space
)
# Define the path to the Harry Potter PDF file.
pdf_path = "https://kvongcmehsanalibrary.wordpress.com/wp-content/uploads/2021/07/harry-potter-sorcerers-stone.pdf"

# --- Split the PDF into chapters a nd preprocess the text ---
text = read_from_url(pdf_path)
chapters = split_into_chapters(text)
chapters = replace_t_with_space(chapters)

# --- Summarization Prompt Template for LLM-based Summarization ---

# Load YAML data from a file

with open('prompts/prompt.yaml', 'r') as file:
    prompt_config = yaml.safe_load(file)

summarization_prompt_template = prompt_config['prompt']['summarize']

# Create a PromptTemplate object using the template string.
# The input variable "text" will be replaced with the content to summarize.
summarization_prompt = PromptTemplate(
    template=summarization_prompt_template,
    input_variables=["text"]
)

def create_chapter_summary(chapter):
    """
    Creates a summary of a chapter using a large language model (LLM).
    Args:
        chapter: A Document object representing the chapter to summarize.
    Returns:
        A Document object containing the summary of the chapter.
    """
    # Extract the text content from the chapter
    chapter_txt = chapter.page_content
    # Specify the LLM model and configuration
    model_name = "microsoft/phi-4-mini-instruct"
    # Generate the summary using the Hugging Face model
    summary_text = generate_summary_with_hf_model(chapter_txt, model_name)



    # Create a Document object for the summary, preserving chapter metadata

    doc_summary = Document(page_content=summary_text, metadata=chapter.metadata)



    return doc_summary
