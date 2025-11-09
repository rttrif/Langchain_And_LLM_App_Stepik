import logging
import pandas as pd
import re
import os
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

from basic_rag import llm, retriever, prompt, StrOutputParser, embed_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class QualityEvaluationPipeline:
    def __init__(self, topic: str, test_questions: List[str], k: int = 5):
        self.topic = topic
        self.test_questions = test_questions
        self.k = k
        self.llm = llm
        self.retriever = retriever
        self.rag_prompt = prompt
        self.embed_model = embed_model

    def _generate_ground_truth(self, question: str, contexts: List[str]) -> str:
        gt_template = """На основе предоставленного контекста дай точный и полный ответ на вопрос.

Контекст:
{context}

Вопрос: {question}

Ответ:"""

        gt_prompt = ChatPromptTemplate.from_template(gt_template)
        gt_chain = gt_prompt | self.llm | StrOutputParser()

        return gt_chain.invoke({
            "context": "\n\n".join(contexts),
            "question": question
        })

    def _get_rag_answer(self, question: str):
        docs = self.retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        doc_ids = [doc.metadata.get('source', f'doc_{i}') for i, doc in enumerate(docs)]

        rag_chain = (
                {
                    "context": lambda x: "\n\n".join([f"[Источник {i + 1}] {ctx}" for i, ctx in enumerate(contexts)]),
                    "question": lambda x: x["question"],
                    "chat_history": lambda x: "Нет предыдущих сообщений."
                }
                | self.rag_prompt
                | self.llm
                | StrOutputParser()
        )

        answer = rag_chain.invoke({"question": question})

        return answer, contexts, doc_ids

    def _calculate_citation_rate(self, answer: str) -> float:
        sentences = re.split(r'[.?!]\s+', answer.strip())
        sentences = [s for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        cited = sum(1 for sent in sentences if re.search(r'\[.*?\d+.*?\]', sent))
        return cited / len(sentences)

    def _calculate_hallucination_rate(self, answer: str, contexts: List[str]) -> float:
        check_template = """Проверь, содержит ли ответ информацию, которой нет в контексте.

Контекст:
{context}

Ответ:
{answer}

Содержит ли ответ информацию не из контекста (галлюцинацию)? Ответь только "Да" или "Нет"."""

        check_prompt = ChatPromptTemplate.from_template(check_template)
        check_chain = check_prompt | self.llm | StrOutputParser()

        result = check_chain.invoke({
            "context": "\n\n".join(contexts),
            "answer": answer
        }).strip().lower()

        return 1.0 if 'да' in result or 'yes' in result else 0.0

    def _calculate_precision_recall(self, retrieved_doc_ids: List[str], ground_truth_doc_ids: List[str]) -> Dict[
        str, float]:
        retrieved_set = set(retrieved_doc_ids[:self.k])
        ground_truth_set = set(ground_truth_doc_ids)

        if not retrieved_set:
            return {"precision_at_k": 0.0, "recall_at_k": 0.0, "f1_at_k": 0.0}

        relevant_found = len(retrieved_set.intersection(ground_truth_set))

        precision = relevant_found / len(retrieved_set) if retrieved_set else 0.0
        recall = relevant_found / len(ground_truth_set) if ground_truth_set else 0.0

        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision_at_k": precision,
            "recall_at_k": recall,
            "f1_at_k": f1
        }

    def evaluate(self) -> pd.DataFrame:
        logging.info(f"Начинаю оценку RAG системы по теме: {self.topic}")

        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
            "retrieved_doc_ids": []
        }

        for question in self.test_questions:
            logging.info(f"Обработка вопроса: {question}")

            answer, contexts, doc_ids = self._get_rag_answer(question)
            ground_truth = self._generate_ground_truth(question, contexts)

            data["question"].append(question)
            data["answer"].append(answer)
            data["contexts"].append(contexts)
            data["ground_truth"].append(ground_truth)
            data["retrieved_doc_ids"].append(doc_ids)

        dataset = Dataset.from_dict({
            "question": data["question"],
            "answer": data["answer"],
            "contexts": data["contexts"],
            "ground_truth": data["ground_truth"]
        })

        logging.info("Запуск оценки RAGAS метрик")

        results = evaluate(
            dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy
            ],
            llm=self.llm,
            embeddings=self.embed_model
        )

        results_df = results.to_pandas()

        logging.info("Расчет дополнительных метрик")

        citation_rates = []
        hallucination_rates = []
        precision_values = []
        recall_values = []
        f1_values = []

        for i, question in enumerate(self.test_questions):
            answer = data["answer"][i]
            contexts = data["contexts"][i]
            retrieved_doc_ids = data["retrieved_doc_ids"][i]

            citation_rate = self._calculate_citation_rate(answer)
            citation_rates.append(citation_rate)

            hallucination_rate = self._calculate_hallucination_rate(answer, contexts)
            hallucination_rates.append(hallucination_rate)

            pr_metrics = self._calculate_precision_recall(retrieved_doc_ids, retrieved_doc_ids)
            precision_values.append(pr_metrics["precision_at_k"])
            recall_values.append(pr_metrics["recall_at_k"])
            f1_values.append(pr_metrics["f1_at_k"])

        results_df['citation_rate'] = citation_rates
        results_df['hallucination_rate'] = hallucination_rates
        results_df[f'precision_at_{self.k}'] = precision_values
        results_df[f'recall_at_{self.k}'] = recall_values
        results_df[f'f1_at_{self.k}'] = f1_values

        output_file = f"rag_evaluation_{self.topic.replace(' ', '_')}.xlsx"
        results_df.to_excel(output_file, index=False)
        logging.info(f"Результаты сохранены в {output_file}")

        logging.info("Оценка завершена")

        return results_df


if __name__ == "__main__":
    topic = "volume_trading"
    queries = [
        "what kind of trade setups exist?",
        "what kind of confirmation setups exist",
        "how use volume trading"
    ]

    pipeline = QualityEvaluationPipeline(topic=topic, test_questions=queries)
    results = pipeline.evaluate()

    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ:")
    print("=" * 80)
    print(results.to_string())
    print("\n" + "=" * 80)