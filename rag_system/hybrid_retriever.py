import re
from collections import defaultdict
from typing import List, Dict, Tuple, Literal

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


class LangChainHybridRetriever(BaseRetriever):
    hybrid_retriever: "HybridRetriever"
    k: int

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        results = self.hybrid_retriever.retrieve(query, k_final=self.k)
        return [doc for doc, _ in results]


class HybridRetriever:
    """
    Гибридный ретривер: BM25 + Vector search + Cross-encoder reranking
    """

    TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9_]+", re.UNICODE)

    def __init__(
            self,
            vectorstore: QdrantVectorStore,
            documents: List[Document],
            *,
            fusion: Literal["rrf", "weighted"] = "rrf",
            alpha: float = 0.5,
            k_bm25: int = 12,
            k_vector: int = 12,
            rrf_k: int = 60,
            cross_encoder_name: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2",
            ce_threshold: float | None = None,
    ):
        self.vectorstore = vectorstore
        self.documents = documents
        self.fusion = fusion
        self.alpha = alpha
        self.k_bm25 = k_bm25
        self.k_vector = k_vector
        self.rrf_k = rrf_k
        self.ce_threshold = ce_threshold

        self.doc_id_map = {i: doc for i, doc in enumerate(documents)}

        tokenized_corpus = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.cross_encoder = CrossEncoder(cross_encoder_name) if cross_encoder_name else None

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        return [t.lower() for t in cls.TOKEN_RE.findall(text)]

    def _normalize_scores(self, scores: Dict[int, float], reverse: bool = False) -> Dict[int, float]:
        if not scores:
            return {}

        vals = np.array(list(scores.values()), dtype=float)
        if reverse:
            vals = -vals

        lo, hi = float(vals.min()), float(vals.max())
        if hi == lo:
            return {k: 1.0 for k in scores}

        norm = (vals - lo) / (hi - lo)
        return {k: float(v) for k, v in zip(scores.keys(), norm)}

    def _bm25_search(self, query: str, k: int) -> Dict[int, float]:
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return {int(i): float(scores[i]) for i in top_idx}

    def _vector_search(self, query: str, k: int) -> Dict[int, float]:
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)

        tmp = {}
        for doc, distance in docs_with_scores:
            doc_idx = next((i for i, d in self.doc_id_map.items()
                            if d.page_content == doc.page_content), None)
            if doc_idx is not None:
                tmp[doc_idx] = 1.0 / (1.0 + float(distance))

        return self._normalize_scores(tmp)

    def _fuse_rrf(self, bm25_scores: Dict[int, float], vec_scores: Dict[int, float]) -> Dict[int, float]:
        def to_ranks(scores: Dict[int, float], higher_better: bool = True) -> Dict[int, int]:
            items = sorted(scores.items(), key=lambda x: x[1], reverse=higher_better)
            return {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(items)}

        ranks_bm25 = to_ranks(bm25_scores, higher_better=True)
        ranks_vec = to_ranks(vec_scores, higher_better=True)

        all_ids = set(ranks_bm25) | set(ranks_vec)
        fused = defaultdict(float)

        for doc_id in all_ids:
            if doc_id in ranks_bm25:
                fused[doc_id] += 1.0 / (self.rrf_k + ranks_bm25[doc_id])
            if doc_id in ranks_vec:
                fused[doc_id] += 1.0 / (self.rrf_k + ranks_vec[doc_id])

        return dict(fused)

    def _fuse_weighted(self, bm25_scores: Dict[int, float], vec_scores: Dict[int, float]) -> Dict[int, float]:
        bm25_norm = self._normalize_scores(bm25_scores)
        vec_norm = self._normalize_scores(vec_scores)

        all_ids = set(bm25_norm) | set(vec_norm)

        return {
            doc_id: self.alpha * bm25_norm.get(doc_id, 0.0) + (1.0 - self.alpha) * vec_norm.get(doc_id, 0.0)
            for doc_id in all_ids
        }

    def retrieve(self, query: str, k_final: int = 5) -> List[Tuple[Document, float]]:
        bm25_scores = self._bm25_search(query, k=self.k_bm25)
        vec_scores = self._vector_search(query, k=self.k_vector)

        if self.fusion == "rrf":
            fused_scores = self._fuse_rrf(bm25_scores, vec_scores)
        else:
            fused_scores = self._fuse_weighted(bm25_scores, vec_scores)

        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)

        if self.cross_encoder and ranked_ids:
            cand_texts = [self.doc_id_map[i].page_content for i in ranked_ids]
            pairs = [(query, text) for text in cand_texts]
            ce_scores = self.cross_encoder.predict(pairs).tolist()

            tmp = [(doc_id, score) for doc_id, score in zip(ranked_ids, ce_scores)
                   if self.ce_threshold is None or score >= self.ce_threshold]

            tmp.sort(key=lambda x: x[1], reverse=True)
            ranked_ids = [doc_id for doc_id, _ in tmp]

        ranked_ids = ranked_ids[:k_final]
        return [(self.doc_id_map[i], float(fused_scores.get(i, 0.0))) for i in ranked_ids]

    def as_langchain_retriever(self, k: int = 5):
        return LangChainHybridRetriever(hybrid_retriever=self, k=k)


class AdvancedHybridRetriever:
    """
    Продвинутый гибридный ретривер:
    - Parent-Child chunking (ищем по мелким, возвращаем крупные)
    - BM25 + Vector search
    - Cross-encoder reranking
    - Contextual compression (LLM фильтрация)
    """

    TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9_]+", re.UNICODE)

    def __init__(
            self,
            child_vectorstore: QdrantVectorStore,
            child_documents: List[Document],
            parent_documents: Dict[str, Document],
            llm: BaseLanguageModel,
            *,
            fusion: Literal["rrf", "weighted"] = "rrf",
            alpha: float = 0.5,
            k_bm25: int = 12,
            k_vector: int = 12,
            rrf_k: int = 60,
            cross_encoder_name: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2",
            ce_threshold: float | None = None,
            use_contextual_compression: bool = True,
            compression_threshold: float = 0.5,
    ):
        self.child_vectorstore = child_vectorstore
        self.child_documents = child_documents
        self.parent_documents = parent_documents
        self.llm = llm
        self.fusion = fusion
        self.alpha = alpha
        self.k_bm25 = k_bm25
        self.k_vector = k_vector
        self.rrf_k = rrf_k
        self.ce_threshold = ce_threshold
        self.use_contextual_compression = use_contextual_compression
        self.compression_threshold = compression_threshold

        self.child_id_map = {i: doc for i, doc in enumerate(child_documents)}

        tokenized_corpus = [self._tokenize(doc.page_content) for doc in child_documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.cross_encoder = CrossEncoder(cross_encoder_name) if cross_encoder_name else None

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        return [t.lower() for t in cls.TOKEN_RE.findall(text)]

    def _normalize_scores(self, scores: Dict[int, float], reverse: bool = False) -> Dict[int, float]:
        if not scores:
            return {}

        vals = np.array(list(scores.values()), dtype=float)
        if reverse:
            vals = -vals

        lo, hi = float(vals.min()), float(vals.max())
        if hi == lo:
            return {k: 1.0 for k in scores}

        norm = (vals - lo) / (hi - lo)
        return {k: float(v) for k, v in zip(scores.keys(), norm)}

    def _bm25_search(self, query: str, k: int) -> Dict[int, float]:
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return {int(i): float(scores[i]) for i in top_idx}

    def _vector_search(self, query: str, k: int) -> Dict[int, float]:
        docs_with_scores = self.child_vectorstore.similarity_search_with_score(query, k=k)

        tmp = {}
        for doc, distance in docs_with_scores:
            doc_idx = next((i for i, d in self.child_id_map.items()
                            if d.page_content == doc.page_content), None)
            if doc_idx is not None:
                tmp[doc_idx] = 1.0 / (1.0 + float(distance))

        return self._normalize_scores(tmp)

    def _fuse_rrf(self, bm25_scores: Dict[int, float], vec_scores: Dict[int, float]) -> Dict[int, float]:
        def to_ranks(scores: Dict[int, float], higher_better: bool = True) -> Dict[int, int]:
            items = sorted(scores.items(), key=lambda x: x[1], reverse=higher_better)
            return {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(items)}

        ranks_bm25 = to_ranks(bm25_scores, higher_better=True)
        ranks_vec = to_ranks(vec_scores, higher_better=True)

        all_ids = set(ranks_bm25) | set(ranks_vec)
        fused = defaultdict(float)

        for doc_id in all_ids:
            if doc_id in ranks_bm25:
                fused[doc_id] += 1.0 / (self.rrf_k + ranks_bm25[doc_id])
            if doc_id in ranks_vec:
                fused[doc_id] += 1.0 / (self.rrf_k + ranks_vec[doc_id])

        return dict(fused)

    def _fuse_weighted(self, bm25_scores: Dict[int, float], vec_scores: Dict[int, float]) -> Dict[int, float]:
        bm25_norm = self._normalize_scores(bm25_scores)
        vec_norm = self._normalize_scores(vec_scores)

        all_ids = set(bm25_norm) | set(vec_norm)

        return {
            doc_id: self.alpha * bm25_norm.get(doc_id, 0.0) + (1.0 - self.alpha) * vec_norm.get(doc_id, 0.0)
            for doc_id in all_ids
        }

    def _get_parent_documents(self, child_ids: List[int]) -> List[Tuple[Document, str]]:
        parent_docs = []
        seen_parents = set()

        for child_id in child_ids:
            child_doc = self.child_id_map[child_id]
            parent_id = child_doc.metadata.get("parent_id")

            if parent_id and parent_id not in seen_parents:
                parent_doc = self.parent_documents.get(parent_id)
                if parent_doc:
                    parent_docs.append((parent_doc, parent_id))
                    seen_parents.add(parent_id)

        return parent_docs

    def _contextual_compression(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        compression_prompt = ChatPromptTemplate.from_template("""
Given the following question and document, determine if the document contains relevant information to answer the question.
Return ONLY a relevance score from 0.0 to 1.0, where:
- 0.0 = completely irrelevant
- 0.5 = somewhat relevant
- 1.0 = highly relevant

Question: {query}

Document: {document}

Relevance score (just the number):""")

        compression_chain = compression_prompt | self.llm | StrOutputParser()

        compressed_docs = []
        for doc in documents:
            try:
                score_str = compression_chain.invoke({
                    "query": query,
                    "document": doc.page_content[:2000]
                }).strip()

                score = float(score_str)

                if score >= self.compression_threshold:
                    compressed_docs.append((doc, score))
            except (ValueError, Exception):
                compressed_docs.append((doc, 0.5))

        return compressed_docs

    def retrieve(self, query: str, k_final: int = 5) -> List[Tuple[Document, float]]:
        bm25_scores = self._bm25_search(query, k=self.k_bm25)
        vec_scores = self._vector_search(query, k=self.k_vector)

        if self.fusion == "rrf":
            fused_scores = self._fuse_rrf(bm25_scores, vec_scores)
        else:
            fused_scores = self._fuse_weighted(bm25_scores, vec_scores)

        ranked_child_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)

        if self.cross_encoder and ranked_child_ids:
            cand_texts = [self.child_id_map[i].page_content for i in ranked_child_ids]
            pairs = [(query, text) for text in cand_texts]
            ce_scores = self.cross_encoder.predict(pairs).tolist()

            tmp = [(child_id, score) for child_id, score in zip(ranked_child_ids, ce_scores)
                   if self.ce_threshold is None or score >= self.ce_threshold]

            tmp.sort(key=lambda x: x[1], reverse=True)
            ranked_child_ids = [child_id for child_id, _ in tmp]

        parent_docs = self._get_parent_documents(ranked_child_ids)

        if self.use_contextual_compression and parent_docs:
            parent_docs_only = [doc for doc, _ in parent_docs]
            compressed = self._contextual_compression(query, parent_docs_only)
            compressed.sort(key=lambda x: x[1], reverse=True)
            return compressed[:k_final]

        return [(doc, float(fused_scores.get(ranked_child_ids[i], 0.0)))
                for i, (doc, _) in enumerate(parent_docs[:k_final])]
