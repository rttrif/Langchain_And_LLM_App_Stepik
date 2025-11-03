#!/usr/bin/env python3
import os
import sys
import argparse
from dotenv import load_dotenv
from ollama import Client


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Первый запрос к LLM")
    parser.add_argument(
        "-q", "--query",
        default="Привет, бот! Назови столицу Франции.",
        help="Текст запроса к модели (по умолчанию — простой вопрос)"
    )
    args = parser.parse_args()

    client = Client()

    system_message = "Ты дружелюбный ассистент. Отвечай кратко и по делу."

    resp = client.chat(
        model=os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud"),
        messages=[
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': args.query}
        ],
        stream=False
    )

    print(resp['message']['content'].strip())

    prompt_tokens = resp.get('prompt_eval_count', 0)
    response_tokens = resp.get('eval_count', 0)
    total_tokens = prompt_tokens + response_tokens

    if total_tokens > 0:
        print(f"\n(токены: {total_tokens} = {prompt_tokens} вход + {response_tokens} выход)")

    return 0

if __name__ == "__main__":
    sys.exit(main())