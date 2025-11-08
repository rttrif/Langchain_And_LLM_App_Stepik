import logging
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from DocumentPreprocessor import DocumentPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
logging.info("Environment variables loaded")

llm = ChatOllama(model="deepseek-v3.1:671b-cloud",
                 base_url="https://ollama.com",
                 temperature=0.0)
logging.info("LLM initialized")

embed_model = OllamaEmbeddings(model="qwen3-embedding:8b", num_gpu=1)
logging.info("Embedding model initialized")

raw_document_path = '../ORDER_FLOW_Trading_Setups.pdf'

preprocessor = DocumentPreprocessor()
logging.info("DocumentPreprocessor initialized")

docs = preprocessor.process_file(file_path=raw_document_path)
logging.info(f"Document processed: {len(docs)} chunks created")

qdrant_client = QdrantClient("http://localhost:6333")
logging.info("Connected to Qdrant")

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
    logging.info(f"Collection created and populated")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
logging.info("Retriever initialized")

def format_docs(docs):
    return "\n\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)])

template = """Используй следующий контекст для ответа на вопрос. 
Если ответ не находится в контексте, ответь "Не знаю". 
После фактов указывай номер источника в квадратных скобках.

Контекст:
{context}

Вопрос: {question}

Ответ:"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "what kind of trade setups exist?"
logging.info(f"Executing query: '{query}'")

source_docs = retriever.invoke(query)
answer = rag_chain.invoke(query)

logging.info("Answer generated")
print("Ответ:", answer)
print("\nИсточники:")
for i, doc in enumerate(source_docs, 1):
    print(f"[{i}] {doc.metadata}")