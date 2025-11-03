from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama

load_dotenv()

store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Вы полезный ассистент."),
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

print("Начинаем диалог с ботом (для выхода введите 'выход')")

while True:
    user_input = input("Вы: ")

    if user_input.lower() in ("выход", "exit", "quit"):
        print("Бот: До свидания!")
        break

    try:
        response = conversation.invoke(
            {"input": user_input},
            {"configurable": {"session_id": "default"}}
        )
        print(f"Бот: {response.content}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        continue
