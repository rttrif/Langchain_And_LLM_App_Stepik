from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.0)

test_cases = [
    {"user": "Здравствуйте, у меня проблема с заказом.",
     "assistant": "Привет! Давай посмотрим, что случилось с твоим заказом."},
    {"user": "Добрый день. Не могли бы вы подсказать статус заявки 324?",
     "assistant": "Добрый день! С удовольствием помогу вам узнать статус вашей заявки №324."},
    {"user": "Чё по доставке, где мой товар?",
     "assistant": "Добрый день. Пожалуйста, уточните номер вашего заказа, чтобы я мог проверить информацию."}
]

def style_rules_check(text: str) -> dict:
    issues = {}
    if " ты " in text.lower() or text.lower().startswith("ты"):
        issues["second_person_informal"] = True
    if "чё" in text.lower():
        issues["slang_detected"] = True
    return issues

evaluation_prompt = (
    "Ты — эксперт по стилю текста ассистента. Проанализируй ответ ассистента и оцени, соблюдает ли он требования:\n"
    "1) Тон официально-вежливый (да/нет);\n2) Нет ли сленга или жаргона (да/нет);\n3) Обращение на Вы (да/нет).\n"
    "Если есть нарушения, перечисли какие.\n\n"
    "Ответ ассистента: \"{answer}\"\n"
    "Вывод:"
)

def evaluate_with_llm(answer: str) -> dict:
    prompt = evaluation_prompt.format(answer=answer)
    try:
        response = client.invoke([HumanMessage(content=prompt)])
        eval_text = response.content
    except Exception as e:
        eval_text = f"Ошибка при оценке LLM: {e}"
    return {"evaluation": eval_text.strip()}

report = {"cases": []}
summary_violations = {"second_person_informal": 0, "slang_detected": 0}

for case in test_cases:
    assistant_answer = case["assistant"]
    rule_issues = style_rules_check(assistant_answer)
    for issue in rule_issues:
        if rule_issues[issue]:
            summary_violations[issue] = summary_violations.get(issue, 0) + 1
    llm_eval = evaluate_with_llm(assistant_answer)
    case_report = {
        "user": case["user"],
        "assistant": assistant_answer,
        "rule_issues": rule_issues,
        "llm_evaluation": llm_eval["evaluation"]
    }
    report["cases"].append(case_report)

report["summary"] = {
    "total_cases": len(test_cases),
    "violations": summary_violations
}

with open("style_eval.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("Отчёт сохранён в style_eval.json")