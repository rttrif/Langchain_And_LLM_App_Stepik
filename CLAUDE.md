# CLAUDE.md

## System persona: 
 Ты – senior python developer, эксперт по работе с последней версией Langchain, который знает и умеет применять
лучшие практики в работе с LLM. https://docs.langchain.com/oss/python/langchain/overview

## System tone:
Отвечай коротко и по сути, о чем тебя попросили. Если не знаешь или не уверен, то так и скажи. 


## Constraint: 
    - отвечай только на тот вопрос, что тебя попросили 
    - не добавляй в ответ, ничего лишнего 
    - не создавай примеров использования, readme и прочего, если тебя явно об этом не попросили
    - НИКОГДА не пиши демонстрационных примеров, примеров использования, инструкций и readme, пока я сам тебя об этом не попрошу 
    - Используй последнюю версию документации Langchain 

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangChain learning and experimentation repository focused on building LLM applications, RAG (Retrieval-Augmented Generation) systems, and chatbots. 
The codebase demonstrates various LangChain patterns from basic prompt templates to advanced hybrid retrieval with reranking.

## Key Architecture Components

### RAG System Architecture

The repository implements a multi-layered RAG approach with increasing sophistication:

1. **Basic RAG** (`rag_system/basic_rag.py`): Vector similarity search with conversational memory
2. **Hybrid Retrieval** (`rag_system/HybridRetriever.py`): Combines BM25 (lexical) + Vector (semantic) search with cross-encoder reranking
3. **Advanced RAG** (`rag_system/advanced_rag.py`): Multi-collection support with query rewriting and contextual compression

**Hybrid Retriever Design Pattern:**
- Uses both `BM25Okapi` (lexical matching) and `QdrantVectorStore` (semantic embeddings)
- Fusion strategies: RRF (Reciprocal Rank Fusion) or weighted combination
- Cross-encoder reranking with `sentence-transformers/CrossEncoder`
- Parent-Child chunking: searches small chunks, returns larger parent documents
- LLM-based contextual compression for relevance filtering

### Document Processing Pipeline

`rag_system/DocumentPreprocessor.py` provides a comprehensive preprocessing system:
- Supports: PDF, DOCX, TXT, MD, CSV, HTML, XLSX
- Chunking strategies: recursive, token-based, character-based, markdown-aware
- Text cleaning: removes URLs, emails, HTML tags
- Deduplication: exact hash matching or MinHash LSH
- Table detection and fragment merging
- Metadata enrichment (source, topic, date, language)

### LLM Configuration

**Default Models:**
- LLM: `deepseek-v3.1:671b-cloud` via ChatOllama (base_url: "https://ollama.com")
- Embeddings: `qwen3-embedding:8b` via OllamaEmbeddings
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Vector Database:**
- Primary: Qdrant (running at http://localhost:6333)
- FAISS also used in some examples

## Project Structure

```
rag_system/          # Production RAG implementations
├── basic_rag.py              # Basic RAG with conversational memory
├── advanced_rag.py           # Multi-collection RAG
├── HybridRetriever.py        # Hybrid BM25+Vector+Rerank
├── DocumentPreprocessor.py   # Document ingestion pipeline
└── quality_evaluation_pipeline.py  # RAG evaluation metrics

ecom-bot/            # E-commerce chatbot with FAQ and order lookup
├── app.py                    # Main chatbot application
├── data/faq.json             # FAQ knowledge base
└── data/orders.json          # Order status database

study_process/       # Learning examples and notebooks
study_rag/           # RAG experimentation
prompts/             # PromptOps and prompt versioning examples
sql_refactor_test/   # SQL prompt templating demos
```



## RAG Implementation Patterns

### Query Rewriting

The RAG systems implement conversational query rewriting to resolve pronouns and context references:

```python
# Detects pronouns like 'it', 'this', 'the first one'
# Uses LLM to rewrite query with full context from chat history
# Located in basic_rag.py:122-153 and advanced_rag.py:116-147
```

### Retrieval Configuration

Standard retrieval parameters across the codebase:
- `k`: 5 (number of documents to retrieve)
- `score_threshold`: 0.4 (minimum similarity score)
- `search_type`: "similarity_score_threshold"

For hybrid retrieval:
- `k_bm25`: 12 (BM25 candidates)
- `k_vector`: 12 (vector search candidates)
- `rrf_k`: 60 (RRF parameter)
- `fusion`: "rrf" or "weighted"

### Conversational Memory

Uses LangChain's `RunnableWithMessageHistory` pattern:
- In-memory store: `ChatMessageHistory`
- Session-based isolation
- Last 6 messages included in context for query rewriting

## Testing Approach

### RAG Evaluation

The `quality_evaluation_pipeline.py` provides metrics:
- Precision@K: relevant documents in top K results
- Recall: coverage of relevant documents
- MRR (Mean Reciprocal Rank): position of first relevant result
- Faithfulness: answer grounded in retrieved context
- Answer relevance: answer addresses the question

### Comparing Retrievers

Use `compare_retrievers()` function in `basic_rag.py:239-262` to benchmark:
1. Basic vector similarity retriever
2. Hybrid retriever (BM25 + Vector + Cross-encoder)
3. Advanced hybrid with parent-child chunking

## Important Constraints

- All prompts use Russian language by default (system constraint from learning context)
- Document processing assumes bilingual content (English/Russian)
- Chat logs are written in JSONL format with timestamps
- The codebase is educational - focuses on demonstrating patterns rather than production optimization

## Prompt Engineering Notes

The `prompts/` directory contains PromptOps examples from `prompt.yml`:
- Version-controlled prompts with metadata tracking
- A/B testing configuration (80/20 splits)
- Metrics per prompt version (satisfaction_score, resolution_rate)
- Feature flags for gradual rollout
- Structured changelog and ownership

Key pattern: prompts are stored with multiple versions, metrics, and rollout strategies for production use.
