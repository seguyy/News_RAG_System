# News RAG System (Retrieval-Augmented Generation)

This project implements a Retrieval-Augmented Generation (RAG) pipeline over a news dataset (information from BBC News). Given a user question, the system:
1) retrieves the most relevant news items via similarity search, and  
2) injects those items into a prompt to help an LLM produce a grounded answer.

The notebook walks through dataset loading, retrieval, prompt construction, and an end-to-end LLM call.

---

## What’s inside

### Core components implemented in the notebook
- `query_news(indices)`: fetch documents from the dataset by index.
- `retrieve(query, top_k)`: returns indices of the top-k most similar documents (provided via `utils`).
- `get_relevant_data(query, top_k)`: runs retrieval + fetches the corresponding rows.
- `format_relevant_data(relevant_data)`: formats retrieved items as readable context (title, description, date, URL).
- `generate_final_prompt(query, top_k, use_rag, prompt)`: builds the final prompt with optional RAG context.
- `llm_call(query, ...)`: calls the LLM with the final prompt.


### Utilities (`utils.py`)
- Loads a SentenceTransformer embedding model: **BAAI/bge-base-en-v1.5**
- Loads precomputed embeddings from `embeddings.joblib`
- Computes cosine similarity to retrieve top-k most relevant articles
- Provides a small **ipywidgets** UI to compare answers with/without RAG
- Supports LLM calls via:
  - a proxy endpoint (if no API key is provided), or
  - Together API (if `TOGETHER_API_KEY` is set)
---

## Dataset

The notebook expects a deduplicated CSV:
- `news_data_dedup.csv`

Key fields used:
- `title`
- `description`
- `url`
- `published_at`

> Note: The retrieval context is treated as “additional information” to enrich the answer, not the only source of truth.

---

## Requirements

### 1) Required files (same folder as the notebook)
- `news_data_dedup.csv` (loaded from `./news_data_dedup.csv`)
- `embeddings.joblib` (loaded from `./embeddings.joblib`)
- `utils.py`
- `News_RAG.ipynb`
- `unittests.py` (optional, only if you want to run tests)

### 2) Python
- Python **3.9+** (recommended)

### 3) Environment variables

#### Required (for embeddings model)
`utils.py` loads the SentenceTransformer model from a local directory:
- `MODEL_PATH` must point to a folder that contains: `BAAI/bge-base-en-v1.5`

Example:
```bash
export MODEL_PATH="/path/to/models"
# Model must exist at:
# $MODEL_PATH/BAAI/bge-base-en-v1.5
