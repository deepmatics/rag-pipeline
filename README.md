# RAG System Evaluation & Benchmark Report

**Environment:** MacBook Pro (48GB RAM) <br>
**Model:** Gemma 3 (Local via Hugging Face) <br>
**Dataset:** ArXiv Corpus (Vectara Open RAG Bench)

---

## 📊 RAG Baseline (v1.0)
*Recorded on: 2026-01-06*

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Mean Precision** | `16%` | Only 1 in 6 retrieved chunks is relevant on average. |
| **Mean Recall** | `50%` | The correct document is missing 50% of the time. |
| **Overall F1** | `25%` | Low balance between accuracy and retrieval breadth. |
| **Mean Reciprocal Rank (MRR)** | `25%` | On average, the correct info is at the bottom of the Top 3. |

---

## 🔍 Failure Diagnosis
1. **Low Recall (0.50):** The current Embedding model + Vector Search is failing to "find" the document in half of the test cases. 
2. **Ranking issues (0.25 MRR):** Even when the document is found, it is rarely the #1 result. The LLM is forced to process "noise" before getting to the facts.

---

## 🛠️ Step-by-Step Improvement Plan

### 1. Hybrid Search (Recall Focus)
* **Action:** Add BM25 Keyword matching alongside Vector Search.
* **Goal:** Capture technical terms and specific IDs that the vector model misses.

### 2. Cross-Encoder Re-ranking (MRR Focus)
* **Action:** Retrieve Top 20 chunks, then use a re-ranker.

### 3. Semantic Chunking
* **Action:** Replace fixed-length character splitting with NLTK/Spacy sentence splitting.
* **Goal:** Ensure chunks contain complete thoughts, making them easier to embed and retrieve.
---

## 📉 Benchmark Evolution Tracker

| Version | Description | Recall | MRR | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| v.1 | Naive RAG | 50% | 25% | 25% |
| v.2| Hybrid Search | *Pending* | *Pending* | *Pending* |
| v1.0 | Re-ranking (Cross-Encoder) | *Pending* | *Pending* | *Pending* |


# Project Directory Structure

```text
rag-project/
├── .venv/                 # Virtual environment
├── config/                # Configuration files (YAML/TOML)
│   ├── base_config.yaml
│   └── experimental.yaml
├── data/                  # Local datasets (Git-ignored)
│   ├── raw/
│   └── processed/
├── docs/                  # Project documentation
│   ├── benchmarks.md      # Performance history
│   └── architecture.md    # Design patterns and logic
├── notebooks/             # R&D and prototyping
├── src/                   # Source Logic
│   ├── __init__.py
│   ├── components/        # Strategy Pattern implementations
│   │   ├── retriever.py   # Vector, Hybrid, BM25 logic
│   │   ├── generator.py   # LLM interaction logic
│   │   └── evaluator.py   # DeepEval metrics
│   ├── utils/             # JSON cleaning, text processing
│   ├── factory.py         # Build pipeline based on config
│   └── pipeline.py        # Orchestration (retrieve -> gen -> eval)
├── tests/                 # Unit tests
├── main.py                # Entry point
├── README.md              # Project overview
└── pyproject.toml         # Dependencies (uv/pip)