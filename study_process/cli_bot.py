import os
import logging
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

logging.basicConfig(
    filename="chat_session.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def setup_bot():
    llm = ChatOllama(
        model="gpt-oss:120b-cloud",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Ты — полезный, вежливый и точный ассистент."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm

    conversation = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    return conversation


def chat_loop(conversation):
    while True:
        try:
            user_input = input("Вы: ")
        except (KeyboardInterrupt, EOFError):
            print("\n[Прерывание пользователем]")
            logging.info("User interrupted. Session ended.")
            break

        user_input = user_input.strip()
        if user_input == "":
            continue

        if user_input.lower() in ("выход", "quit", "exit", "стоп", "конец"):
            print("Бот: До свидания!")
            logging.info("User initiated exit. Session ended.")
            break

        if user_input.lower() in ("сброс", "reset"):
            store["default"].clear()
            print("Бот: Контекст диалога очищен.")
            logging.info("Context cleared by user.")
            continue

        logging.info(f"User: {user_input}")

        try:
            response = conversation.invoke(
                {"input": user_input},
                {"configurable": {"session_id": "default"}}
            )
            bot_reply = response.content.strip()
            logging.info(f"Bot: {bot_reply}")
            print(f"Бот: {bot_reply}")
        except Exception as e:
            logging.error(f"Error: {e}")
            print(f"Бот: [Ошибка] {e}")
            continue


def main():
    logging.info("=== New session ===")
    conversation = setup_bot()
    print("Чат-бот запущен! Можете задавать вопросы. Для выхода введите 'выход'.\n")
    chat_loop(conversation)


if __name__ == "__main__":
    main()