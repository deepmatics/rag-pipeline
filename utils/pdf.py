import PyPDF2
from langchain_core.documents import Document
import re
import requests
import io

# ============================================================================
# PDF PROCESSING FUNCTIONS
# ============================================================================

def read_from_url(url):
    """
    Reads a PDF from a URL.

    Args:
        book_url (str): The URL to the PDF book file.

    Returns:
        text: text content and chapter number metadata.
    
    """
    response = requests.get(url)
    pdf_file = io.BytesIO(response.content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    documents = pdf_reader.pages

    # Concatenate text from all pages
    text = " ".join([doc.extract_text() for doc in documents])

    return text

def split_into_chapters(text):
    """
    Splits a PDF book from a URL into chapters based on chapter title patterns.

    Args:
        book_url (str): The URL to the PDF book file.

    Returns:
        list: A list of Document objects, each representing a chapter with its 
              text content and chapter number metadata.
    """

    # Split text into chapters based on chapter title pattern
    chapters = re.split(r'(CHAPTER\s[A-Z]+(?:\s[A-Z]+)*)', text)

    # Create Document objects with chapter metadata
    chapter_docs = []
    chapter_num = 1
    for i in range(1, len(chapters), 2):
        chapter_text = chapters[i] + chapters[i + 1]  # Combine title and content
        doc = Document(page_content=chapter_text, metadata={"chapter": chapter_num})
        chapter_docs.append(doc)
        chapter_num += 1

    return chapter_docs


def extract_book_quotes_as_documents(documents, min_length=50):
    """
    Extracts quotes from documents and returns them as separate Document objects.

    Args:
        documents (list): List of Document objects to extract quotes from.
        min_length (int, optional): Minimum length of quotes to extract. Defaults to 50.

    Returns:
        list: List of Document objects containing extracted quotes.
    """
    quotes_as_documents = []
    # Pattern for quotes longer than min_length characters, including line breaks
    quote_pattern_longer_than_min_length = re.compile(rf'"(.{{{min_length},}}?)"', re.DOTALL)

    for doc in documents:
        content = doc.page_content
        content = content.replace('\n', ' ')
        found_quotes = quote_pattern_longer_than_min_length.findall(content)
        
        for quote in found_quotes:
            quote_doc = Document(page_content=quote)
            quotes_as_documents.append(quote_doc)
    
    return quotes_as_documents
