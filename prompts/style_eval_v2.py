import json
import re
import argparse
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


def check_no_first_person(text: str) -> bool:
    return (re.search(r"\bя\b", text, flags=re.IGNORECASE) is None
            and re.search(r"\bмы\b", text, flags=re.IGNORECASE) is None)


def check_sentence_count(text: str, max_sentences: int = 3) -> bool:
    sentences = re.split(r"[.!?]+", text)
    num_sentences = sum(1 for s in sentences if s.strip())
    return num_sentences <= max_sentences


def get_llm_style_evaluation(text: str) -> dict:
    llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.0)

    system_msg = SystemMessage(content=(
        "Ты – помощник, оценивающий стиль ответа по заданным правилам. "
        "Критерии: (1) тон ответа – деловой, вежливый; (2) речь от третьего лица, "
        "избегать местоимения 'я'; (3) нет жаргона, только профессиональные термины. "
        "Ответь в формате JSON с полями: formality_score (0-10), no_first_person (True/False), comment."
    ))
    user_msg = HumanMessage(content=f"Оцени стиль следующего ответа:\n\"\"\"\n{text}\n\"\"\"")

    try:
        response = llm.invoke([system_msg, user_msg])
        content = response.content
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        return {"error": f"LLM API failed: {str(e)}"}


def evaluate_answer(answer_text: str) -> dict:
    llm_scores = get_llm_style_evaluation(answer_text)

    violations = {
        "no_first_person": not check_no_first_person(answer_text),
        "too_many_sentences": not check_sentence_count(answer_text, max_sentences=3)
    }

    return {"llm_scores": llm_scores, "violations": violations}


def main(input_path: str, output_path: str):
    import os

    if not os.path.exists(input_path):
        default_data = {
            "answers": [
                {"id": 1, "text": "Здравствуйте! Я рад помочь. Вот решение..."},
                {"id": 2, "text": "Привет, думаю, тебе стоит попробовать это..."}
            ]
        }
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(default_data, f, ensure_ascii=False, indent=2)
        print(f"Created default input file: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    answers = data.get("answers", data) if isinstance(data, dict) else data

    report = {"answers": [], "summary": {}}
    violations_count = {"no_first_person": 0, "too_many_sentences": 0}
    formality_score_sum = 0
    total = len(answers)

    for idx, ans in enumerate(answers, start=1):
        text = ans["text"] if isinstance(ans, dict) else str(ans)
        answer_id = ans.get("id", idx) if isinstance(ans, dict) else idx

        result = evaluate_answer(text)

        entry = {
            "id": answer_id,
            "text": text,
            "llm_scores": result["llm_scores"],
            "violations": result["violations"]
        }
        report["answers"].append(entry)

        if result["violations"]["no_first_person"]:
            violations_count["no_first_person"] += 1
        if result["violations"]["too_many_sentences"]:
            violations_count["too_many_sentences"] += 1

        score = result["llm_scores"].get("formality_score")
        if isinstance(score, (int, float)):
            formality_score_sum += score

        print(f"[{idx}/{total}] Answer {answer_id} evaluated")

    report["summary"]["total_answers"] = total
    report["summary"]["violations_count"] = violations_count
    report["summary"]["average_formality_score"] = round(formality_score_sum / total, 2) if total else None

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved style evaluation report to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate style of answers and generate report.")
    parser.add_argument("--input", "-i", default="answers.json", help="Path to input JSON with answers.")
    parser.add_argument("--output", "-o", default="style_eval.json", help="Path to output report JSON.")
    args = parser.parse_args()
    main(args.input, args.output)