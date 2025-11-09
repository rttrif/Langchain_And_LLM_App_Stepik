import logging
from typing import List, Dict
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document

from DocumentPreprocessor import DocumentPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MultiCollectionRAG:
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        load_dotenv()

        self.llm = ChatOllama(
            model="deepseek-v3.1:671b-cloud",
            base_url="https://ollama.com",
            temperature=0.0
        )

        self.embed_model = OllamaEmbeddings(model="qwen3-embedding:8b", num_gpu=1)
        self.qdrant_client = QdrantClient(qdrant_url)
        self.qdrant_url = qdrant_url
        self.preprocessor = DocumentPreprocessor()
        self.collections: Dict[str, QdrantVectorStore] = {}
        self.store = {}
        self.last_retrieved_docs = []

        logging.info("MultiCollectionRAG initialized")

    def add_collection(self, collection_name: str, file_path: str = None, docs: List[Document] = None):
        collection_exists = self.qdrant_client.collection_exists(collection_name)

        if collection_exists:
            points_count = self.qdrant_client.get_collection(collection_name).points_count
            if points_count > 0:
                vectorstore = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=collection_name,
                    embedding=self.embed_model
                )
                logging.info(f"Using existing collection '{collection_name}' with {points_count} points")
                self.collections[collection_name] = vectorstore
                return

        if docs is None and file_path:
            docs = self.preprocessor.process_file(file_path=file_path)
            logging.info(f"Processed {len(docs)} chunks from {file_path}")

        if docs:
            vectorstore = QdrantVectorStore.from_documents(
                documents=docs,
                embedding=self.embed_model,
                url=self.qdrant_url,
                collection_name=collection_name,
                batch_size=512
            )
            self.collections[collection_name] = vectorstore
            logging.info(f"Created collection '{collection_name}' with {len(docs)} documents")

    def retrieve_from_multiple_collections(self, query: str, k: int = 5, score_threshold: float = 0.4) -> List[
        Document]:
        all_docs = []
        seen_content = set()

        for name, vectorstore in self.collections.items():
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold}
            )
            docs = retriever.invoke(query)

            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    if 'source' not in doc.metadata:
                        doc.metadata['source'] = name
                    all_docs.append(doc)

        return all_docs[:k]

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def format_docs(self, docs):
        if not docs:
            return "Контекст не найден."

        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            formatted.append(f"[Источник {i} - {source}] {doc.page_content}")

        return "\n\n".join(formatted)

    def format_chat_history(self, history):
        if not history:
            return "Нет предыдущих сообщений."
        formatted = []
        for msg in history:
            role = "Пользователь" if msg.type == "human" else "Ассистент"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted[-6:])

    def rewrite_query(self, question: str, history) -> str:
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
        rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()

        rewritten = rewrite_chain.invoke({
            "chat_history": self.format_chat_history(history),
            "question": question
        })

        rewritten = rewritten.strip().strip('"').strip("'")
        logging.info(f"Query rewritten: '{question}' -> '{rewritten}'")
        return rewritten

    def build_chain(self, k: int = 5, score_threshold: float = 0.4):
        template = """Используй следующий контекст для ответа на вопрос.

ВАЖНО:
- Если контекст не содержит информации для ответа, ответь "Не знаю" или "В предоставленных документах нет информации по этому вопросу"
- НЕ придумывай информацию, которой нет в контексте
- После каждого факта указывай номер источника в квадратных скобках
- Если информация из разных источников противоречит, укажи это явно
- Учитывай историю беседы

История беседы:
{chat_history}

Контекст:
{context}

Вопрос: {question}

Ответ:"""

        prompt = ChatPromptTemplate.from_template(template)

        def retrieve_with_rewrite(x):
            question = x["question"]
            history = x.get("history", [])
            rewritten = self.rewrite_query(question, history)
            docs = self.retrieve_from_multiple_collections(rewritten, k=k, score_threshold=score_threshold)
            self.last_retrieved_docs = docs
            return self.format_docs(docs)

        rag_chain = (
                {
                    "context": retrieve_with_rewrite,
                    "question": lambda x: x["question"],
                    "chat_history": lambda x: self.format_chat_history(x.get("history", []))
                }
                | prompt
                | self.llm
                | StrOutputParser()
        )

        conversational_rag = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        return conversational_rag

    def chat(self, query: str, session_id: str = "default", k: int = 5,
             score_threshold: float = 0.4, show_sources: bool = True):
        if not self.collections:
            return "Коллекции не загружены. Добавьте коллекции через add_collection()"

        chain = self.build_chain(k=k, score_threshold=score_threshold)

        logging.info(f"Query: '{query}' | Session: {session_id}")

        input_data = {"question": query}

        answer = chain.invoke(
            input_data,
            config={"configurable": {"session_id": session_id}}
        )

        if show_sources:
            return answer, self.last_retrieved_docs
        return answer

    def print_sources(self, docs):
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
    rag = MultiCollectionRAG()

    rag.add_collection(
        collection_name="order_flow_trading",
        file_path="../ORDER_FLOW_Trading_Setups.pdf"
    )

    session_id = "trading_session_1"

    queries = [
        "what kind of trade setups exist?",
        "tell me more about the first one",
        "what are the key indicators?"
    ]

    for query in queries:
        print(f"\n{'=' * 80}")
        print(f"ВОПРОС: {query}")
        print(f"{'=' * 80}")

        answer, sources = rag.chat(query, session_id, k=5, score_threshold=0.4, show_sources=True)

        print(f"\nОТВЕТ:\n{answer}")
        rag.print_sources(sources)