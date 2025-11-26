import logging
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional, Protocol

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


@dataclass
class RetrievedDoc:
    """Retrieved document with relevance score."""
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float


class Retriever(Protocol):
    """Protocol for retriever interface."""

    def retrieve(self, query: str, k_final: int) -> List[tuple[Any, float]]:
        ...


class RefusalPolicy:
    """Policy for determining when to refuse answering based on confidence metrics."""

    def __init__(
            self,
            threshold: float = 0.55,
            min_docs: int = 1,
            refusal_phrases: Optional[List[str]] = None,
    ):
        self.threshold = float(threshold)
        self.min_docs = int(min_docs)
        self.refusal_phrases = refusal_phrases or [
            "Извините, я не нашёл надёжной информации по вашему вопросу.",
            "Не знаю. В текущей базе знаний нет достаточного контекста.",
            "Не могу ответить уверенно — недостаточно релевантных источников.",
        ]

    def check_confidence(
            self, query: str, docs: List[RetrievedDoc]
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Calculate confidence metric from top-N documents.

        Formula: conf = 0.7 * top1_norm + 0.3 * max(margin, 0)
        where top1_norm is normalized top-1 score and margin is difference between top-1 and top-2.
        """
        if not docs or len(docs) < self.min_docs:
            return False, {"reason": "no_docs", "conf": 0.0, "n_docs": len(docs)}

        scores = np.array([d.score for d in docs], dtype=float)

        if np.allclose(scores.max(), scores.min()):
            top1_norm, margin = 1.0, 0.0
        else:
            normalized = (scores - scores.min()) / (scores.max() - scores.min())
            top1_norm = float(normalized[0])
            margin = float(normalized[0] - (normalized[1] if len(normalized) > 1 else 0.0))

        conf = 0.7 * top1_norm + 0.3 * max(margin, 0.0)
        ok = conf >= self.threshold

        return ok, {
            "conf": conf,
            "top1_norm": top1_norm,
            "margin": margin,
            "n_docs": len(docs)
        }

    def refusal_text(self) -> str:
        """Get random refusal phrase."""
        return random.choice(self.refusal_phrases)


class FallbackHandlers:
    """Collection of fallback strategies when answer confidence is low."""

    @staticmethod
    def suggest_rephrase(query: str, stats: Dict[str, Any]) -> str:
        """Suggest query rephrasing."""
        return (
            "Попробуйте уточнить запрос: добавьте ключевые термины, даты или названия разделов. "
            'Напр.: «…в отчёте 2020 года», «…определение из главы "Архитектура RAG"».'
        )

    @staticmethod
    def to_web_router(query: str, stats: Dict[str, Any]) -> str:
        """Fallback to web search."""
        return "Могу попробовать поиск в веб-источниках (если он включён в конфигурации)."

    @staticmethod
    def log_only(query: str, stats: Dict[str, Any]) -> None:
        """Silent logging fallback."""
        return None


class QAWithConfidence:
    """QA system with confidence checking and refusal capability."""

    def __init__(
            self,
            retriever: Retriever,
            policy: RefusalPolicy,
            llm: Optional[Any] = None,
            fallback: Optional[Callable[[str, Dict[str, Any]], str]] = None,
            k_context: int = 3,
            prompts: Dict[str, str] = None
    ):
        self.retriever = retriever
        self.policy = policy
        self.llm = llm
        self.fallback = fallback
        self.k_context = int(k_context)
        self.prompts = prompts or {}
        self.log = logging.getLogger("qa_confidence")

    def _llm_answer(self, query: str, ctx_docs: List[RetrievedDoc]) -> str:
        """Generate answer using LLM with provided context."""
        context = "\n\n---\n\n".join([d.text for d in ctx_docs])

        template = self.prompts.get('rag_answer_template', "")
        if not template:
             # Fallback
             template = """Ты помощник по внутренней базе знаний. Отвечай ТОЛЬКО по приведённому контексту.
Если информации нет — ответь буквально: «Не знаю».

Контекст:
{context}

Вопрос: {question}

Ответ:"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        answer = chain.invoke({"context": context, "question": query})
        return answer.strip()

    def retrieve_docs(self, query: str) -> List[RetrievedDoc]:
        """Retrieve documents and convert to RetrievedDoc format."""
        raw = self.retriever.retrieve(query, k_final=self.k_context)
        docs = []

        for raw_doc, score in raw:
            doc_id = getattr(raw_doc, "id", None) or raw_doc.metadata.get("id", "unknown")
            text = getattr(raw_doc, "page_content", None) or getattr(raw_doc, "text", str(raw_doc))
            metadata = getattr(raw_doc, "metadata", {})

            docs.append(RetrievedDoc(
                id=str(doc_id),
                text=text,
                metadata=metadata,
                score=float(score),
            ))

        return docs

    def answer_with_confidence(self, query: str) -> Dict[str, Any]:
        """
        Process query with confidence checking.

        Returns:
            Dict with keys: status, answer, stats, docs, fallback (optional)
        """
        docs = self.retrieve_docs(query)
        ok, stats = self.policy.check_confidence(query, docs)

        if not ok:
            text = self.policy.refusal_text()
            extra = self.fallback(query, stats) if self.fallback else None
            self.log.info({"event": "refusal", "query": query, **stats})

            return {
                "status": "refusal",
                "answer": text,
                "fallback": extra,
                "stats": stats,
                "docs": docs
            }

        if not self.llm:
            self.log.info({"event": "confident_no_llm", "query": query, **stats})
            return {
                "status": "confident",
                "answer": None,
                "stats": stats,
                "docs": docs
            }

        answer = self._llm_answer(query, docs)
        self.log.info({"event": "answer", "query": query, **stats})

        return {
            "status": "answer",
            "answer": answer,
            "stats": stats,
            "docs": docs
        }
