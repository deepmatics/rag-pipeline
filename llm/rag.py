import yaml
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from utils.helper import generate_summary_with_hf_model

# --- Document Loading and Vector Store ---
from langchain_core.document_loaders import PyPDFLoader
from langchain_core.vectorstores import FAISS
from langchain_core.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings 

from utils.pdf import (
    read_from_url,
    split_into_chapters
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
    chapter_txt = chapters[0].page_content
    # Specify the LLM model and configuration
    model_name = "microsoft/phi-4-mini-instruct"
    # Generate the summary using the Hugging Face model
    summary_text = generate_summary_with_hf_model(chapter_txt, model_name)
    # Create a Document object for the summary, preserving chapter metadata
    doc_summary = Document(page_content=summary_text, metadata=chapter.metadata)
    return doc_summary


# --- Generate Summaries for Each Chapter ---
# Initialize an empty list to store the summaries of each chapter
chapter_summaries = []

# Iterate over each chapter in the chapters list
for chapter in chapters:
    # Generate a summary for the current chapter using the create_chapter_summary function
    summary = create_chapter_summary(chapter)
    # Append the summary to the chapter_summaries list
    chapter_summaries.append(summary)

    	
def encode_book(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a FAISS vector store using OpenAI embeddings.

    Args:
        path (str): The path to the PDF file.
        chunk_size (int): The desired size of each text chunk.
        chunk_overlap (int): The amount of overlap between consecutive chunks.

    Returns:
        FAISS: A FAISS vector store containing the encoded book content.
    """

    # 1. Load the PDF document using PyPDFLoader
    loader = PyPDFLoader(path)
    documents = loader.load()

    # 2. Split the document into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)

    # 3. Clean up the text chunks (replace unwanted characters)
    cleaned_texts = replace_t_with_space(texts)

    # 4. Create HuggingFace embeddings using Gemma model and encode the cleaned text chunks into a FAISS vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="google/embeddinggemma-300m",
        query_encode_kwargs={"prompt_name": "query"},
        encode_kwargs={"prompt_name": "document"}
    )
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    # 5. Return the vector store
    return vectorstore