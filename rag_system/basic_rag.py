from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from DocumentPreprocessor import DocumentPreprocessor

load_dotenv()

llm = ChatOllama(model="deepseek-v3.1:671b-cloud",
                 base_url="https://ollama.com",
                 temperature=0.0)

embed_model = OllamaEmbeddings(model="qwen3-embedding:8b")

raw_document_path = 'ORDER_FLOW_Trading_Setups.pdf'

preprocessor = DocumentPreprocessor()

docs = preprocessor.process_file(file_path=raw_document_path)

qdrant_client = QdrantClient("http://localhost:6333")

vectorstore = QdrantVectorStore.from_documents(
    documents=docs,
    embedding=embed_model,
    url="http://localhost:6333",
    collection_name="order_flow_trading_setups_collection"
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

query = "what kind of trade setups exist?"
docs = retriever.get_relevant_documents(query)
for doc in docs:
    print(doc.metadata, "->", doc.page_content[:50])