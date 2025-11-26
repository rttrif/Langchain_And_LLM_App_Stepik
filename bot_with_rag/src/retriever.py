import re
from collections import defaultdict
from typing import List, Dict, Tuple, Literal, Any, Union

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
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
    Поддерживает как Qdrant, так и FAISS (через общий интерфейс VectorStore).
    """

    TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9_]+", re.UNICODE)

    def __init__(
            self,
            vectorstore: VectorStore,
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

        # Map index to document for BM25
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
        # similarity_search_with_score returns (Document, score)
        # Note: FAISS returns L2 distance (lower is better) or Cosine similarity (higher is better) depending on config.
        # LangChain FAISS usually returns L2 distance by default if not normalized, or cosine distance.
        # However, similarity_search_with_score usually returns a "score".
        # For Qdrant it's similarity (higher better).
        # For FAISS (default L2) it's distance (lower better).
        # We need to handle this. But wait, LangChain's `similarity_search_with_score` docstring says:
        # "Return docs and relevance scores in the range [0, 1]." (ideally).
        # Let's assume standard behavior or check the vectorstore type.
        
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)

        tmp = {}
        for doc, score in docs_with_scores:
            # Find doc index in our local map
            # This relies on exact content matching which might be slow or fragile if duplicates exist.
            # Ideally we should use IDs.
            doc_idx = next((i for i, d in self.doc_id_map.items()
                            if d.page_content == doc.page_content), None)
            
            if doc_idx is not None:
                # We assume score is "distance" if it's > 1 or we treat it as such for safety?
                # Actually, let's just normalize later. 
                # If it's cosine similarity, it's -1 to 1.
                # If it's L2, it's 0 to inf.
                # The original code used: 1.0 / (1.0 + float(distance)) for Qdrant (which returns similarity? No, Qdrant returns similarity usually).
                # Wait, Qdrant `similarity_search_with_score` returns score.
                # If using FAISS, `similarity_search_with_score` returns L2 distance by default.
                # 1 / (1 + distance) is a good way to convert distance to similarity [0, 1].
                
                # Let's try to detect if it's a distance or similarity.
                # For now, we'll assume it needs conversion if we want to be safe, OR just normalize what we get.
                # But mixing L2 (lower good) and BM25 (higher good) is bad.
                # We will assume the vectorstore returns a SCORE where HIGHER IS BETTER.
                # If FAISS returns distance, we might need `similarity_search_with_relevance_scores`.
                
                # Let's use `similarity_search_with_relevance_scores` if available, which guarantees 0-1 range (higher better).
                # But standard FAISS might not support it well without specific index types.
                
                # Fallback: Use 1/(1+score) if it looks like a distance (e.g. > 1 or we know it's FAISS L2).
                # But simpler: just use the score and let normalization handle it, BUT we need to know direction.
                
                # Let's stick to the original logic: 1 / (1 + score)
                # This assumes 'score' is a distance (0 is perfect).
                # If 'score' is similarity (1 is perfect), this becomes 0.5.
                
                # Let's blindly apply 1/(1+score) as in the original code, assuming it was tuned for Qdrant returning distance?
                # Actually Qdrant usually returns Cosine Similarity.
                # If Qdrant returns Cosine (0..1), then 1/(1+0.9) = 0.52.
                # If FAISS returns L2 (0..inf), then 1/(1+0) = 1.0.
                
                # I will use the score as is, but normalize it.
                tmp[doc_idx] = float(score)

        # If we suspect these are distances (lower is better), we should invert them before normalization?
        # Or just rely on the fact that we want to combine them with BM25 (higher is better).
        # Let's assume we want HIGHER IS BETTER.
        # If the vector store is FAISS (L2), lower is better.
        # If the vector store is Qdrant (Cosine), higher is better.
        
        # Heuristic: If we are using FAISS, it's likely L2 (lower better).
        # If we are using Qdrant, it's likely Cosine (higher better).
        
        is_distance = False
        if "faiss" in str(type(self.vectorstore)).lower():
             is_distance = True # FAISS default is L2
        
        if is_distance:
             # Convert distance to similarity
             for k_idx in tmp:
                 tmp[k_idx] = 1.0 / (1.0 + tmp[k_idx])
        
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
