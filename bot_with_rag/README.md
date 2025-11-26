# Bot with RAG

This is a complete RAG system implementing data loading, preprocessing, FAISS/Qdrant indexing, hybrid retrieval, and a refusal mechanism.

## Features

- **Data Loading**: Supports PDF, TXT, MD, etc.
- **Preprocessing**: Cleaning, chunking, deduplication.
- **Vector Store**: FAISS (default) or Qdrant.
- **Retrieval**: Hybrid (BM25 + Vector) with Cross-Encoder Reranking.
- **Refusal Logic**: Rejects queries with low confidence or insufficient context.
- **Evaluation**: RAGAS metrics pipeline.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Add your documents to the `data/` directory.

3. Run the bot:
   ```bash
   python rag_final.py
   ```

## Configuration

- **Vector Store**: Toggle `USE_QDRANT` in `rag_final.py` to switch between FAISS and Qdrant.
- **LLM**: Configured to use Ollama (`deepseek-v3.1:671b-cloud`).
- **Embeddings**: Configured to use Ollama (`qwen3-embedding:8b`).

## Structure

- `rag_final.py`: Main entry point.
- `src/preprocess.py`: Document processing logic.
- `src/retriever.py`: Hybrid retriever implementation.
- `src/rag_engine.py`: RAG logic with refusal policy.
- `src/metrics.py`: Evaluation pipeline.
