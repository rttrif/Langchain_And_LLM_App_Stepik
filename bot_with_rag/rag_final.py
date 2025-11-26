import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.preprocess import DocumentPreprocessor
from src.retriever import HybridRetriever
from src.rag_engine import QAWithConfidence, RefusalPolicy, FallbackHandlers
from src.metrics import QualityEvaluationPipeline
from src.utils import load_prompts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    LLM_MODEL = "deepseek-v3.1:671b-cloud"
    LLM_BASE_URL = "https://ollama.com"
    EMBED_MODEL = "qwen3-embedding:8b"
    QDRANT_URL = "http://localhost:6333"
    COLLECTION_NAME = "bot_with_rag_collection"
    DATA_DIR = "data"
    USE_QDRANT = False  # Set to True to use Qdrant, False for FAISS

def setup_environment():
    load_dotenv()
    logging.info("Environment variables loaded")

def initialize_components():
    llm = ChatOllama(
        model=Config.LLM_MODEL,
        base_url=Config.LLM_BASE_URL,
        temperature=0.0
    )
    logging.info("LLM initialized")

    embed_model = OllamaEmbeddings(model=Config.EMBED_MODEL, num_gpu=1)
    logging.info("Embedding model initialized")

    return llm, embed_model

def load_and_process_data(data_dir: str, preprocessor: DocumentPreprocessor):
    all_docs = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logging.warning(f"Data directory {data_dir} does not exist.")
        return []

    for file_path in data_path.glob('*.*'):
        if file_path.is_file() and not file_path.name.startswith('.'):
            logging.info(f"Processing file: {file_path}")
            try:
                docs = preprocessor.process_file(file_path=str(file_path))
                all_docs.extend(docs)
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
    
    logging.info(f"Total documents processed: {len(all_docs)}")
    return all_docs

def setup_vectorstore(docs, embed_model, use_qdrant=False):
    if use_qdrant:
        logging.info("Using Qdrant Vector Store")
        qdrant_client = QdrantClient(Config.QDRANT_URL)
        
        if qdrant_client.collection_exists(Config.COLLECTION_NAME):
             logging.info(f"Collection {Config.COLLECTION_NAME} exists.")
        
        vectorstore = QdrantVectorStore.from_documents(
            documents=docs,
            embedding=embed_model,
            url=Config.QDRANT_URL,
            collection_name=Config.COLLECTION_NAME,
            force_recreate=True 
        )
    else:
        logging.info("Using FAISS Vector Store")
        vectorstore = FAISS.from_documents(docs, embed_model)
        vectorstore.save_local("faiss_index")
        logging.info("FAISS index saved to 'faiss_index'")
        
        vectorstore = FAISS.load_local("faiss_index", embed_model, allow_dangerous_deserialization=True)
        logging.info("FAISS index loaded successfully")
        
    return vectorstore

def run_evaluation(llm, retriever, embed_model, prompts=None):
    print("\nRunning Evaluation...")
    test_questions = [
        "What is RAG?",
        "How does the hybrid retriever work?",
        "What is the meaning of life?" # Should trigger refusal
    ]
    
    pipeline = QualityEvaluationPipeline(
        topic="RAG System",
        test_questions=test_questions,
        llm=llm,
        retriever=retriever,
        embed_model=embed_model,
        prompts=prompts
    )
    pipeline.evaluate()

def main():
    setup_environment()
    
    llm, embed_model = initialize_components()
    preprocessor = DocumentPreprocessor()
    
    docs = load_and_process_data(Config.DATA_DIR, preprocessor)
    
    if not docs:
        logging.error("No documents found. Please add documents to the 'data' directory.")
        from langchain_core.documents import Document
        docs = [Document(page_content="This is a placeholder document about RAG systems.", metadata={"source": "dummy"})]
    
    vectorstore = setup_vectorstore(docs, embed_model, use_qdrant=Config.USE_QDRANT)
    
    try:
        prompts = load_prompts(os.path.join(os.path.dirname(__file__), 'prompts.yaml'))
        logging.info("Prompts loaded from prompts.yaml")
    except Exception as e:
        logging.warning(f"Failed to load prompts: {e}. Using defaults.")
        prompts = {}

    hybrid_retriever = HybridRetriever(
        vectorstore=vectorstore,
        documents=docs,
        fusion="rrf",
        k_bm25=5,
        k_vector=5,
        cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    
    policy = RefusalPolicy(threshold=0.55, min_docs=1)
    qa_system = QAWithConfidence(
        retriever=hybrid_retriever,
        policy=policy,
        llm=llm,
        fallback=FallbackHandlers.suggest_rephrase,
        k_context=3,
        prompts=prompts
    )
    
    print("\n" + "="*50)
    print("🤖 Bot with RAG initialized! (Type 'exit' to quit)")
    print("="*50)
    
    while True:
        try:
            query = input("\nUser: ")
            if query.lower() in ['exit', 'quit', 'q']:
                break
                
            if not query.strip():
                continue
                
            result = qa_system.answer_with_confidence(query)
            
            print(f"\nBot: {result.get('answer', result.get('status'))}")
            if result.get('fallback'):
                print(f"Tip: {result['fallback']}")
                
            if result.get('status') == 'refusal':
                 print(f"Confidence stats: {result.get('stats')}")
                 
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.error(f"An error occurred: {e}")

    # run_evaluation(llm, hybrid_retriever, embed_model, prompts)

if __name__ == "__main__":
    main()
