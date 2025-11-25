import tiktoken
import textwrap
import re

# =============================================================================
# TEXT PROCESSING FUNCTIONS
# =============================================================================

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Calculates the number of tokens in a given string using a specified encoding.

    Args:
        string (str): The input string to tokenize.
        encoding_name (str): The name of the encoding to use (e.g., 'cl100k_base').

    Returns:
        int: The number of tokens in the string according to the specified encoding.
    """
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def replace_t_with_space(list_of_documents):
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document.

    Args:
        list_of_documents (list): A list of document objects, each with a 'page_content' attribute.

    Returns:
        list: The modified list of documents with tab characters replaced by spaces.
    """
    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')
    return list_of_documents


def replace_double_lines_with_one_line(text):
    """
    Replaces consecutive double newline characters ('\n\n') with a single newline character ('\n').

    Args:
        text (str): The input text string.

    Returns:
        str: The text string with double newlines replaced by single newlines.
    """
    cleaned_text = re.sub(r'\n\n', '\n', text)
    return cleaned_text


def escape_quotes(text):
    """
    Escapes both single and double quotes in a string.

    Args:
        text (str): The string to escape.

    Returns:
        str: The string with single and double quotes escaped.
    """
    return text.replace('"', '\\"').replace("'", "\\'")


def text_wrap(text, width=120):
    """
    Wraps the input text to the specified width.

    Args:
        text (str): The input text to wrap.
        width (int, optional): The width at which to wrap the text. Defaults to 120.

    Returns:
        str: The wrapped text.
    """
    return textwrap.fill(text, width=width)