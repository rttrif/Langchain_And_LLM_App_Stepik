import logging
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from DocumentPreprocessor import DocumentPreprocessor
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

llm = ChatOllama(
    model="deepseek-v3.1:671b-cloud",
    base_url="https://ollama.com",
    temperature=0.0
)

embed_model = OllamaEmbeddings(
    model="qwen3-embedding:8b",
    num_gpu=1,
    num_thread=4
)

qdrant_client = QdrantClient("http://localhost:6333")

configs = [
    {"name": "small_recursive", "chunk_size": 200, "chunk_overlap": 20, "strategy": "recursive"},
    {"name": "small_token", "chunk_size": 250, "chunk_overlap": 25, "strategy": "token"},
    {"name": "sparse_optimized", "chunk_size": 300, "chunk_overlap": 30, "strategy": "recursive"},
    {"name": "medium_no_overlap", "chunk_size": 400, "chunk_overlap": 0, "strategy": "recursive"},
    {"name": "medium_token", "chunk_size": 500, "chunk_overlap": 50, "strategy": "token"},
    {"name": "hybrid_balanced", "chunk_size": 500, "chunk_overlap": 50, "strategy": "recursive"},
    {"name": "medium_character", "chunk_size": 600, "chunk_overlap": 60, "strategy": "character"},
    {"name": "large_recursive", "chunk_size": 800, "chunk_overlap": 100, "strategy": "recursive"},
    {"name": "dense_optimized", "chunk_size": 800, "chunk_overlap": 100, "strategy": "token"},
    {"name": "large_high_overlap", "chunk_size": 1000, "chunk_overlap": 200, "strategy": "recursive"},
    {"name": "xlarge_token", "chunk_size": 1200, "chunk_overlap": 150, "strategy": "token"},
    {"name": "xlarge_no_overlap", "chunk_size": 1200, "chunk_overlap": 0, "strategy": "recursive"},
    {"name": "xxlarge_recursive", "chunk_size": 1500, "chunk_overlap": 100, "strategy": "recursive"},
    {"name": "xxlarge_token", "chunk_size": 1500, "chunk_overlap": 100, "strategy": "token"},
]

questions = [
    "какие есть торговые сетапы",
    "какие есть подтверждающие сетапы",
    "как их верно использовать вместе"
]

file_path = "../ORDER_FLOW_Trading_Setups.pdf"


def batch_documents(docs: List[Document], batch_size: int = 50):
    for i in range(0, len(docs), batch_size):
        yield docs[i:i + batch_size]


def test_config(config: dict, docs: List[Document]):
    config_start = time.time()

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Конфигурация: {config['name']}")
    logger.info(
        f"Параметры: chunk_size={config['chunk_size']}, overlap={config['chunk_overlap']}, strategy={config['strategy']}")

    split_start = time.time()
    preprocessor = DocumentPreprocessor(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        chunking_strategy=config["strategy"],
        encoding_name="cl100k_base",
        log_metadata_sample=False
    )

    splitted_docs = preprocessor.process_documents(docs)
    split_time = time.time() - split_start
    logger.info(f"  ✓ Чанков: {len(splitted_docs)} ({split_time:.2f}с)")

    collection_name = f"order_flow_{config['name']}"

    try:
        qdrant_client.delete_collection(collection_name)
    except:
        pass

    embed_start = time.time()

    batch_size = 1000
    vector_store = None

    for batch_idx, batch in enumerate(batch_documents(splitted_docs, batch_size)):
        if vector_store is None:
            vector_store = QdrantVectorStore.from_documents(
                documents=batch,
                embedding=embed_model,
                url="http://localhost:6333",
                collection_name=collection_name,
                batch_size=50
            )
        else:
            vector_store.add_documents(batch, batch_size=50)

        processed = min((batch_idx + 1) * batch_size, len(splitted_docs))
        logger.info(f"  Progress: {processed}/{len(splitted_docs)} чанков")

    embed_time = time.time() - embed_start
    logger.info(f"  ✓ Эмбеддинги загружены ({embed_time:.2f}с, {embed_time / len(splitted_docs):.3f}с/чанк)")

    search_start = time.time()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    for q_idx, q in enumerate(questions, 1):
        logger.info(f"\n  Q{q_idx}: {q}")
        q_start = time.time()
        results = retriever.invoke(q)
        q_time = time.time() - q_start
        logger.info(f"    Найдено: {len(results)} ({q_time:.2f}с)")

        for idx, doc in enumerate(results, 1):
            score = doc.metadata.get("score", "N/A")
            page = doc.metadata.get("page", "N/A")
            snippet = doc.page_content[:100].replace("\n", " ")
            logger.info(f"    [{idx}] стр={page}, score={score}")
            logger.info(f"        {snippet}...")

    search_time = time.time() - search_start
    total_time = time.time() - config_start

    logger.info(
        f"\n  Итого '{config['name']}': {total_time:.2f}с (split={split_time:.2f}с, embed={embed_time:.2f}с, search={search_time:.2f}с)")

    return {
        "name": config["name"],
        "chunks": len(splitted_docs),
        "total_time": total_time,
        "split_time": split_time,
        "embed_time": embed_time,
        "search_time": search_time
    }


if __name__ == "__main__":
    total_start = time.time()

    logger.info("=" * 80)
    logger.info("ЭКСПЕРИМЕНТ ПО ОПТИМИЗАЦИИ CHUNKING")
    logger.info("=" * 80)
    logger.info(f"Конфигураций: {len(configs)}, Вопросов: {len(questions)}")

    load_start = time.time()
    loader = PyPDFLoader(file_path)
    base_docs = loader.load()
    load_time = time.time() - load_start

    total_chars = sum(len(doc.page_content) for doc in base_docs)
    logger.info(f"✓ Загружено: {len(base_docs)} страниц, {total_chars:,} символов ({load_time:.2f}с)")

    results = []
    for config_idx, config in enumerate(configs, 1):
        logger.info(f"\n{'#' * 80}")
        logger.info(f"КОНФИГУРАЦИЯ {config_idx}/{len(configs)}")
        logger.info(f"{'#' * 80}")
        result = test_config(config, base_docs)
        results.append(result)

    total_time = time.time() - total_start

    logger.info(f"\n{'=' * 80}")
    logger.info("РЕЗУЛЬТАТЫ")
    logger.info("=" * 80)

    results_sorted = sorted(results, key=lambda x: x["embed_time"])
    logger.info("\nТоп-5 по скорости эмбеддингов:")
    for r in results_sorted[:5]:
        logger.info(
            f"  {r['name']}: {r['embed_time']:.2f}с ({r['chunks']} чанков, {r['embed_time'] / r['chunks']:.3f}с/чанк)")

    logger.info(f"\nВсего: {total_time:.2f}с ({total_time / 60:.1f} мин)")
    logger.info(f"Среднее: {total_time / len(configs):.2f}с/конфиг")
    logger.info("=" * 80)