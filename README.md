# RAG System Evaluation & Benchmark Report

**Environment:** MacBook Pro (48GB RAM) <br>
**Model:** Gemma 3 (Local via Hugging Face) <br>
**Dataset:** ArXiv Corpus (Vectara Open RAG Bench)

---

## 📊 Simple RAG
*Recorded on: 2026-01-06*

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Mean Precision** | `16%` | Only 1 in 6 retrieved chunks is relevant on average. |
| **Mean Recall** | `50%` | The correct document is missing 50% of the time. |
| **Overall F1** | `25%` | Low balance between accuracy and retrieval breadth. |
| **Mean Reciprocal Rank (MRR)** | `25%` | On average, the correct info is at the bottom of the Top 3. |


# Project Directory Structure

```text
├── LICENSE
├── README.md
├── chroma_db
├── config
    ├── rag.yaml
├── data
    ├── input
    ├── output
├── docs
├── experiments
├── main.py
├── rag_engine
    ├── app.py
    ├── chunking
        ├── recursive.py
    ├── data_loader
        ├── json_data.py
    ├── embeddings
        ├── langchain.py
    ├── eval.py
    ├── prompts
        ├── rag.yaml
    ├── utils
        ├── config_loader.py
    ├── vector_db
        ├── chroma.py