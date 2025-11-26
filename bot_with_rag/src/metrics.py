import logging
import pandas as pd
import re
import os
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class QualityEvaluationPipeline:
    def __init__(self, topic: str, test_questions: List[str], llm: Any, retriever: Any, embed_model: Any, prompts: Dict[str, str] = None, k: int = 5):
        self.topic = topic
        self.test_questions = test_questions
        self.k = k
        self.llm = llm
        self.retriever = retriever
        self.embed_model = embed_model
        self.prompts = prompts or {}
        
        template = self.prompts.get('rag_answer_template', "")
        if not template:
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
        self.rag_prompt = ChatPromptTemplate.from_template(template)

    def _generate_ground_truth(self, question: str, contexts: List[str]) -> str:
        gt_template = self.prompts.get('ground_truth_template', "")
        if not gt_template:
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
        # Note: retriever here is expected to be the HybridRetriever instance which has a retrieve method
        # But we need to check if it's a LangChain retriever or our custom class
        if hasattr(self.retriever, 'retrieve'):
             results = self.retriever.retrieve(question, k_final=self.k)
             docs = [doc for doc, _ in results]
        else:
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
        check_template = self.prompts.get('hallucination_check_template', "")
        if not check_template:
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

        for i, question in enumerate(self.test_questions):
            answer = data["answer"][i]
            contexts = data["contexts"][i]

            citation_rate = self._calculate_citation_rate(answer)
            citation_rates.append(citation_rate)

            hallucination_rate = self._calculate_hallucination_rate(answer, contexts)
            hallucination_rates.append(hallucination_rate)

        results_df['citation_rate'] = citation_rates
        results_df['hallucination_rate'] = hallucination_rates

        output_file = f"rag_evaluation_{self.topic.replace(' ', '_')}.xlsx"
        results_df.to_excel(output_file, index=False)
        logging.info(f"Результаты сохранены в {output_file}")

        logging.info("Оценка завершена")

        return results_df
