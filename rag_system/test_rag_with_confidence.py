import logging
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from DocumentPreprocessor import DocumentPreprocessor
from HybridRetriever import HybridRetriever, AdvancedHybridRetriever
from BasicRAG import QAWithConfidence, RefusalPolicy, FallbackHandlers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
logging.info("Environment variables loaded")

llm = ChatOllama(
    model="deepseek-v3.1:671b-cloud",
    base_url="https://ollama.com",
    temperature=0.0
)
logging.info("LLM initialized")

embed_model = OllamaEmbeddings(model="qwen3-embedding:8b", num_gpu=1)
logging.info("Embedding model initialized")

raw_document_path = '../ORDER_FLOW_Trading_Setups.pdf'

preprocessor = DocumentPreprocessor()
docs = preprocessor.process_file(file_path=raw_document_path)
logging.info(f"Document processed: {len(docs)} chunks created")

qdrant_client = QdrantClient("http://localhost:6333")
collection_name = "order_flow_trading_setups_collection"

collection_exists = qdrant_client.collection_exists(collection_name)
points_count = qdrant_client.get_collection(collection_name).points_count if collection_exists else 0

if collection_exists and points_count > 0:
    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embed_model
    )
    logging.info(f"Using existing collection with {points_count} points")
else:
    vectorstore = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embed_model,
        url="http://localhost:6333",
        collection_name=collection_name,
        batch_size=512
    )
    logging.info("Collection created and populated")

hybrid_retriever = HybridRetriever(
    vectorstore=vectorstore,
    documents=docs,
    fusion="rrf",
    k_bm25=12,
    k_vector=12,
    cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
logging.info("HybridRetriever initialized")

parent_documents = {f"parent_{i}": doc for i, doc in enumerate(docs)}
for i, doc in enumerate(docs):
    doc.metadata["parent_id"] = f"parent_{i}"

advanced_hybrid_retriever = AdvancedHybridRetriever(
    child_vectorstore=vectorstore,
    child_documents=docs,
    parent_documents=parent_documents,
    llm=llm,
    fusion="rrf",
    k_bm25=12,
    k_vector=12,
    cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    use_contextual_compression=True
)
logging.info("AdvancedHybridRetriever initialized")

policy = RefusalPolicy(threshold=0.55, min_docs=1)

qa_with_hybrid = QAWithConfidence(
    retriever=hybrid_retriever,
    policy=policy,
    llm=llm,
    fallback=FallbackHandlers.suggest_rephrase,
    k_context=3
)
logging.info("QAWithConfidence with HybridRetriever initialized")

qa_with_advanced = QAWithConfidence(
    retriever=advanced_hybrid_retriever,
    policy=policy,
    llm=llm,
    fallback=FallbackHandlers.suggest_rephrase,
    k_context=3
)
logging.info("QAWithConfidence with AdvancedHybridRetriever initialized")


def test_query(qa_system, query: str, system_name: str):
    print(f"\n{'=' * 100}")
    print(f"SYSTEM: {system_name}")
    print(f"QUERY: {query}")
    print(f"{'=' * 100}")

    result = qa_system.answer_with_confidence(query)

    print(f"\nSTATUS: {result['status']}")
    print(f"\nSTATS:")
    for key, value in result['stats'].items():
        print(f"  {key}: {value}")

    print(f"\nANSWER:\n{result['answer']}")

    if result.get('fallback'):
        print(f"\nFALLBACK SUGGESTION:\n{result['fallback']}")

    print(f"\nRETRIEVED DOCS ({len(result['docs'])}):")
    for i, doc in enumerate(result['docs'], 1):
        print(f"  [{i}] Score: {doc.score:.4f} | {doc.text[:150]}...")

    return result


if __name__ == "__main__":
    test_queries = [
        ("what kind of trade setups exist?", "В базе знаний"),
        ("what is order flow trading?", "В базе знаний"),
        ("who is the author of this document?", "Возможно есть"),
        ("what is the GDP of USA in 2023?", "Точно НЕТ в базе"),
        ("how to cook pasta carbonara?", "Точно НЕТ в базе"),
    ]

    print("\n" + "=" * 100)
    print("ТЕСТИРОВАНИЕ С HYBRID RETRIEVER")
    print("=" * 100)

    for query, category in test_queries:
        print(f"\n[Категория: {category}]")
        test_query(qa_with_hybrid, query, "HybridRetriever")

    print("\n\n" + "=" * 100)
    print("ТЕСТИРОВАНИЕ С ADVANCED HYBRID RETRIEVER")
    print("=" * 100)

    for query, category in test_queries:
        print(f"\n[Категория: {category}]")
        test_query(qa_with_advanced, query, "AdvancedHybridRetriever")

    print("\n" + "=" * 100)
    print("СРАВНЕНИЕ ПОЛИТИК ОТКАЗА")
    print("=" * 100)

    policies = [
        ("Строгая (threshold=0.7)", RefusalPolicy(threshold=0.7, min_docs=1)),
        ("Средняя (threshold=0.55)", RefusalPolicy(threshold=0.55, min_docs=1)),
        ("Мягкая (threshold=0.3)", RefusalPolicy(threshold=0.3, min_docs=1)),
    ]

    test_query_for_comparison = "what are the main trading concepts?"

    for policy_name, policy_obj in policies:
        print(f"\n{'=' * 80}")
        print(f"POLICY: {policy_name}")
        print(f"{'=' * 80}")

        qa_temp = QAWithConfidence(
            retriever=hybrid_retriever,
            policy=policy_obj,
            llm=llm,
            fallback=FallbackHandlers.suggest_rephrase,
            k_context=3
        )

        result = qa_temp.answer_with_confidence(test_query_for_comparison)
        print(f"Status: {result['status']}")
        print(f"Confidence: {result['stats']['conf']:.4f}")
        print(f"Answer: {result['answer'][:200]}...")