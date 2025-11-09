import logging
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

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
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.4}
)
logging.info("Retriever initialized")

store = {}
last_retrieved_docs = []


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def format_docs(docs):
    if not docs:
        return "Контекст не найден."
    return "\n\n".join([f"[Источник {i + 1}] {doc.page_content}" for i, doc in enumerate(docs)])


def format_chat_history(history):
    if not history:
        return "Нет предыдущих сообщений."
    formatted = []
    for msg in history:
        role = "Пользователь" if msg.type == "human" else "Ассистент"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted[-6:])


def rewrite_query(question: str, history) -> str:
    if not history or len(history) < 2:
        return question

    has_pronouns = any(word in question.lower() for word in
                       ['it', 'this', 'that', 'the first', 'the second', 'them', 'they',
                        'его', 'её', 'это', 'этот', 'первый', 'второй', 'их'])

    if not has_pronouns:
        return question

    rewrite_template = """На основе истории беседы перефразируй вопрос так, чтобы он был понятен без контекста.
Замени местоимения и отсылки на конкретные термины из истории.

История:
{chat_history}

Вопрос: {question}

Перефразированный вопрос (только вопрос):"""

    rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    rewritten = rewrite_chain.invoke({
        "chat_history": format_chat_history(history),
        "question": question
    })

    rewritten = rewritten.strip().strip('"').strip("'")
    logging.info(f"Query rewritten: '{question}' -> '{rewritten}'")
    return rewritten


template = """Используй следующий контекст для ответа на вопрос.

ВАЖНО:
- Если контекст не содержит информации для ответа на вопрос, ответь "Не знаю" или "В предоставленных документах нет информации по этому вопросу"
- НЕ придумывай информацию, которой нет в контексте
- После каждого факта указывай номер источника в квадратных скобках, например [Источник 1]
- Учитывай историю беседы при формулировке ответа

История беседы:
{chat_history}

Контекст:
{context}

Вопрос: {question}

Ответ:"""

prompt = ChatPromptTemplate.from_template(template)


def retrieve_with_rewrite(x):
    global last_retrieved_docs
    question = x["question"]
    history = x.get("history", [])
    rewritten = rewrite_query(question, history)
    docs = retriever.invoke(rewritten)
    last_retrieved_docs = docs
    return format_docs(docs)


rag_chain = (
        {
            "context": retrieve_with_rewrite,
            "question": lambda x: x["question"],
            "chat_history": lambda x: format_chat_history(x.get("history", []))
        }
        | prompt
        | llm
        | StrOutputParser()
)

conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


def chat(query: str, session_id: str = "default", show_sources: bool = True):
    global last_retrieved_docs
    logging.info(f"Executing query: '{query}'")

    input_data = {"question": query}

    answer = conversational_rag.invoke(
        input_data,
        config={"configurable": {"session_id": session_id}}
    )

    logging.info("Answer generated")

    if show_sources:
        return answer, last_retrieved_docs
    return answer


def print_sources(docs):
    if not docs:
        print("\n[Источники не найдены]")
        return

    print("\n" + "=" * 80)
    print("ИСТОЧНИКИ:")
    print("=" * 80)
    for i, doc in enumerate(docs, 1):
        print(f"\n[Источник {i}]")
        print(f"Метаданные: {doc.metadata}")
        print(f"Текст: {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}")
        print("-" * 80)


if __name__ == "__main__":
    session_id = "user_session_1"

    queries = [
        "what kind of trade setups exist?",
        "tell me more about the first one",
        "how to identify it?"
    ]

    for query in queries:
        print(f"\n{'=' * 80}")
        print(f"ВОПРОС: {query}")
        print(f"{'=' * 80}")

        answer, sources = chat(query, session_id, show_sources=True)

        print(f"\nОТВЕТ:\n{answer}")
        print_sources(sources)