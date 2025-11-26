"""
Helper Functions for Controllable RAG System

This module contains utility functions for text processing, document manipulation,
PDF handling, similarity analysis, and metric evaluation for RAG applications.
"""

# Standard library imports
import re

# Third-party imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import PyPDF2
import pylcs
import pandas as pd
import dill
import os

def generate_summary_with_hf_model(text_to_summarize, model_name="google/gemma-3-4b-it"):
    """
    Generates a summary of a given text using a Hugging Face transformer model.

    Args:
        text_to_summarize (str): The text to be summarized.
        model_name (str): The name of the Hugging Face model to use.

    Returns:
        str: The generated summary.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    # It's important to craft a prompt that instructs the model to summarize
    prompt = f"Summarize the following text:\n\n{text_to_summarize}\n\nSummary:"
    
    result = summarizer(prompt, max_length=len(text_to_summarize) // 2, num_return_sequences=1)
    
    return result[0]['generated_text']


# =============================================================================
# SIMILARITY AND ANALYSIS FUNCTIONS
# =============================================================================

def is_similarity_ratio_lower_than_th(large_string, short_string, th):
    """
    Checks if the similarity ratio between two strings is lower than a given threshold.

    Uses the Longest Common Subsequence (LCS) algorithm to calculate similarity.

    Args:
        large_string (str): The larger string to compare.
        short_string (str): The shorter string to compare.
        th (float): The similarity threshold (0.0 to 1.0).

    Returns:
        bool: True if the similarity ratio is lower than the threshold, False otherwise.
    """
    # Calculate the length of the longest common subsequence (LCS)
    lcs = pylcs.lcs_sequence_length(large_string, short_string)

    # Calculate the similarity ratio
    similarity_ratio = lcs / len(short_string)

    # Check if the similarity ratio is lower than the threshold
    return similarity_ratio < th


def analyse_metric_results(results_df):
    """
    Analyzes and prints the results of various RAG evaluation metrics.

    Args:
        results_df (pandas.DataFrame): A pandas DataFrame containing the metric results.
    """
    metric_descriptions = {
        "faithfulness": "Measures how well the generated answer is supported by the retrieved documents.",
        "answer_relevancy": "Measures how relevant the generated answer is to the question.",
        "context_precision": "Measures the proportion of retrieved documents that are actually relevant.",
        "context_relevancy": "Measures how relevant the retrieved documents are to the question.",
        "context_recall": "Measures the proportion of relevant documents that are successfully retrieved.",
        "context_entity_recall": "Measures the proportion of relevant entities mentioned in the question that are also found in the retrieved documents.",
        "answer_similarity": "Measures the semantic similarity between the generated answer and the ground truth answer.",
        "answer_correctness": "Measures whether the generated answer is factually correct."
    }

    for metric_name, metric_value in results_df.items():
        print(f"\n**{metric_name.upper()}**")

        # Extract the numerical value from the Series object
        if isinstance(metric_value, pd.Series):
            metric_value = metric_value.values[0]

        # Print explanation and score for each metric
        if metric_name in metric_descriptions:
            print(metric_descriptions[metric_name])
            print(f"Score: {metric_value:.4f}")
        else:
            print(f"Score: {metric_value:.4f}")


# =============================================================================
# OBJECT SERIALIZATION FUNCTIONS
# =============================================================================

def save_object(obj, filename):
    """
    Save a Python object to a file using dill serialization.
    
    Args:
        obj: The Python object to save.
        filename (str): The name of the file where the object will be saved.
    """
    with open(filename, 'wb') as file:
        dill.dump(obj, file)
    print(f"Object has been saved to '{filename}'.")


def load_object(filename):
    """
    Load a Python object from a file using dill deserialization.
    
    Args:
        filename (str): The name of the file from which the object will be loaded.
    
    Returns:
        object: The loaded Python object.
    """
    with open(filename, 'rb') as file:
        obj = dill.load(file)
    print(f"Object has been loaded from '{filename}'.")
    return obj


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
# save_object(plan_and_execute_app, 'plan_and_execute_app.pkl')
# plan_and_execute_app = load_object('plan_and_execute_app.pkl')
