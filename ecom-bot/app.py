import os
import json
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def load_json_file(filename):
    filepath = DATA_DIR / filename
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load {filename}: {e}")
        return None


def search_faq(user_input, faq_data):
    user_lower = user_input.lower()
    for item in faq_data:
        if any(word in user_lower for word in item["q"].lower().split()):
            return item["a"]
    return None


def get_order_status(order_id, orders_data):
    order = orders_data.get(order_id)
    if not order:
        return f"Заказ #{order_id} не найден. Проверьте номер."

    status_map = {
        "in_transit": "в пути",
        "delivered": "доставлен",
        "processing": "обрабатывается"
    }

    status = status_map.get(order["status"], order["status"])
    result = f"Заказ #{order_id}: {status}"

    if "eta_days" in order:
        result += f", прибудет через {order['eta_days']} дн."
    if "delivered_at" in order:
        result += f", доставлен {order['delivered_at']}"
    if "note" in order:
        result += f". {order['note']}"
    if "carrier" in order:
        result += f" (перевозчик: {order['carrier']})"

    return result


def setup_bot():
    brand_name = "Shoply"
    model = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")

    llm = ChatOllama(
        model=model,
        temperature=0,
    )

    system_prompt = f"""Ты — вежливый и краткий ассистент поддержки интернет-магазина {brand_name}.
Отвечай по делу, не фантазируй информацию.
Если не знаешь ответ — так и скажи.
Будь дружелюбным, но профессиональным."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
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


def log_to_jsonl(session_id, role, content, usage=None):
    log_file = LOGS_DIR / f"session_{session_id}.jsonl"
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "content": content
    }
    if usage:
        log_entry["usage"] = usage

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def chat_loop(conversation, session_id):
    faq_data = load_json_file("faq.json") or []
    orders_data = load_json_file("orders.json") or {}

    print(f"Бот {os.getenv('BRAND_NAME', 'Shoply')} запущен!")
    print("Команды: /order <id> - проверить заказ, /exit - выход, /reset - сброс контекста\n")

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[Завершение работы]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "выход"):
            print("Бот: До свидания!")
            log_to_jsonl(session_id, "system", "Session ended")
            break

        if user_input.lower() == "/reset":
            if session_id in store:
                store[session_id].clear()
            print("Бот: Контекст очищен.")
            log_to_jsonl(session_id, "system", "Context reset")
            continue

        if user_input.startswith("/order "):
            order_id = user_input.split(maxsplit=1)[1].strip()
            response = get_order_status(order_id, orders_data)
            print(f"Бот: {response}")
            log_to_jsonl(session_id, "user", user_input)
            log_to_jsonl(session_id, "assistant", response)
            continue

        log_to_jsonl(session_id, "user", user_input)

        faq_answer = search_faq(user_input, faq_data)
        if faq_answer:
            print(f"Бот: {faq_answer}")
            log_to_jsonl(session_id, "assistant", faq_answer, usage={"type": "faq"})
            continue

        try:
            response = conversation.invoke(
                {"input": user_input},
                {"configurable": {"session_id": session_id}}
            )

            bot_reply = response.content.strip()

            usage_info = {
                "prompt_tokens": getattr(response, "usage_metadata", {}).get("input_tokens", 0),
                "completion_tokens": getattr(response, "usage_metadata", {}).get("output_tokens", 0),
                "total_tokens": getattr(response, "usage_metadata", {}).get("total_tokens", 0)
            }

            print(f"Бот: {bot_reply}")
            log_to_jsonl(session_id, "assistant", bot_reply, usage=usage_info)

        except Exception as e:
            error_msg = f"[Ошибка] {e}"
            print(f"Бот: {error_msg}")
            log_to_jsonl(session_id, "error", str(e))


def main():
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_to_jsonl(session_id, "system", "Session started")

    conversation = setup_bot()
    chat_loop(conversation, session_id)


if __name__ == "__main__":
    main()