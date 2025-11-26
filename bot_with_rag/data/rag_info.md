# RAG System Architecture

Retrieval-Augmented Generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources.

## Components

1.  **Retrieval**: Finds relevant documents based on the user query.
2.  **Generation**: The LLM generates an answer using the retrieved documents as context.

## Hybrid Retrieval

Hybrid retrieval combines keyword-based search (BM25) with semantic search (Vector Embeddings) to improve recall.

## Refusal Mechanism

A robust RAG system should refuse to answer if:
- No relevant documents are found.
- The confidence score of the retrieved documents is below a threshold.
