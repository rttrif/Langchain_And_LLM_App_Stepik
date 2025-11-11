System persona: 
 Ты – senior python developer, эксперт по работе с последней версией Langchain, который знает и умеет применять
лучшие практики в работе с LLM. https://docs.langchain.com/oss/python/langchain/overview

System tone:
Отвечай коротко и по сути, о чем тебя попросили. Если не знаешь или не уверен, то так и скажи. 

Context:
Я изучаю Langchain на курсе и сейчас урок - Гибридный поиск и реранк
Шаг 4: Артефакт – гибридный ретривер с реранком
Цель шага: Завершить работу над модулем RAG, собрав полностью рабочий гибридный ретривер. Получить код-артефакт, который можно протестировать на запросах, сравнить с обычным ретривером и убедиться в улучшении качества ответов.

Объяснение:
В качестве итогового результата мы создаём модуль (класс или функцию) для гибридного поиска. Артефакт представляет собой реализованный HybridRetriever с поддержкой реранка, который интегрируется в цепочку вопрос-ответ. Он включает:

Настройку подключения к хранилищу эмбеддингов (например, FAISS или Qdrant с заранее заложенными эмбеддингами документов).

# requirements (минимум):
# pip install langchain-community langchain-openai rank-bm25 sentence-transformers faiss-cpu tiktoken

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from rank_bm25 import BM25Okapi
import re

# ===== данные =====
@dataclass
class RawDoc:
    id: str
    text: str
    metadata: Dict[str, Any]

corpus: List[RawDoc] = [
    RawDoc("d1", "Компания XYZ основана в 1999 году. Миссия — сделать ИИ доступным.", {"source":"ustav_company","page":1}),
    RawDoc("d2", "В 2020 году компания выпустила продукт Alpha. Он ориентирован на B2B.", {"source":"report_2020","page":3}),
    RawDoc("d3", "Наше видение: безопасные и объяснимые модели машинного обучения.", {"source":"vision","page":2}),
]

# ===== эмбеддинги + FAISS =====
emb = OpenAIEmbeddings(
    # при необходимости укажите свой proxy/base_url
    # base_url="https://api.essayai.ru/v1",
    model="text-embedding-3-large"
)
faiss = FAISS.from_documents(
    [Document(page_content=d.text, metadata={"id": d.id, **d.metadata}) for d in corpus],
    embedding=emb
)

# ===== BM25 =====
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9_]+", re.UNICODE)
def tokenize(txt: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(txt)]

tokenized_corpus = [tokenize(d.text) for d in corpus]
bm25 = BM25Okapi(tokenized_corpus)

id2doc = {d.id: d for d in corpus}

                  
Настройку подключения к поисковому индексу (например, Elasticsearch, или простой TF-IDF индекс через Langchain.vectorstores.TextualInvertedIndex или аналог).

from collections import defaultdict
import numpy as np
from sentence_transformers import CrossEncoder

class HybridRetriever:
    """
    Гибридный ретривер:
      • BM25 (лексический) + Vector (семантический, FAISS)
      • Фьюжн: RRF или взвешенная сумма (после нормализации)
      • Опционально: cross-encoder реранк (например, 'cross-encoder/ms-marco-MiniLM-L-6-v2')
    """
    def __init__(
        self,
        faiss_store: FAISS,
        bm25_index: BM25Okapi,
        raw_docs: Dict[str, RawDoc],
        *,
        fusion: str = "rrf",         # "rrf" | "weighted"
        alpha: float = 0.5,          # для weighted
        k1: int = 12,                # кандидаты из BM25
        k2: int = 12,                # кандидаты из Vector
        rrf_k: int = 60,             # параметр RRF
        cross_encoder_name: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ce_threshold: float | None = None,  # отбрасывать документы с низким CE-скором
    ):
        self.faiss = faiss_store
        self.bm25 = bm25_index
        self.raw = raw_docs
        self.fusion = fusion
        self.alpha = alpha
        self.k1 = k1
        self.k2 = k2
        self.rrf_k = rrf_k

        self.cross_encoder = CrossEncoder(cross_encoder_name) if cross_encoder_name else None
        self.ce_threshold = ce_threshold

    # --- вспомогательные ---

    def _normalize(self, scores: Dict[str, float], *, reverse: bool=False) -> Dict[str, float]:
        """min-max нормализация в [0,1]. reverse=True — инвертировать (когда меньшие — лучше)."""
        if not scores:
            return {}
        vals = np.array(list(scores.values()), dtype=float)
        if reverse:
            vals = -vals
        lo, hi = float(vals.min()), float(vals.max())
        if hi == lo:
            return {k: 1.0 for k in scores}
        it = iter(scores.items())
        keys = [k for k,_ in scores.items()]
        norm = (vals - lo) / (hi - lo)
        return {k: float(v) for k, v in zip(keys, norm)}

    def _rrf(self, ranks: Dict[str, int]) -> Dict[str, float]:
        """RRF score(d) = 1 / (rrf_k + rank(d))"""
        return {doc_id: 1.0 / (self.rrf_k + r) for doc_id, r in ranks.items()}

    # --- кандидаты из BM25 / Vector ---

    def _bm25_candidates(self, query: str, k: int) -> Dict[str, float]:
        toks = tokenize(query)
        scores = self.bm25.get_scores(toks)  # чем выше — тем лучше
        top_idx = np.argsort(scores)[::-1][:k]
        res = {}
        for i in top_idx:
            doc_id = list(self.raw.keys())[i]
            res[doc_id] = float(scores[i])
        return res

    def _vector_candidates(self, query: str, k: int) -> Dict[str, float]:
        # LangChain FAISS similarity_search_with_score возвращает (doc, distance)
        # Часто distance ~ L2, где МЕНЬШЕ — ЛУЧШЕ → инвертируем в «похожесть».
        docs_scores: List[Tuple[Document, float]] = self.faiss.similarity_search_with_score(query, k=k)
        # преобразуем к id -> similarity_in_[0,1]
        # сначала distance->similarity: s = 1/(1+d), затем min-max
        tmp = {d.metadata.get("id"): 1.0 / (1.0 + float(dist)) for d, dist in docs_scores}
        return self._normalize(tmp)  # уже [0..1]

    # --- слияние кандидатов ---

    def _fuse_rrf(self, bm25: Dict[str, float], vec: Dict[str, float]) -> Dict[str, float]:
        # превращаем оценки в ранги (1 — лучший)
        def to_ranks(scores: Dict[str, float], higher_better=True) -> Dict[str, int]:
            items = sorted(scores.items(), key=lambda kv: kv[1], reverse=higher_better)
            return {doc_id: rank+1 for rank, (doc_id, _) in enumerate(items)}
        ranks_bm25 = to_ranks(bm25, higher_better=True)
        ranks_vec  = to_ranks(vec,  higher_better=True)
        # объединяем носители
        all_ids = set(ranks_bm25) | set(ranks_vec)
        fused = defaultdict(float)
        for doc_id in all_ids:
            if doc_id in ranks_bm25:
                fused[doc_id] += 1.0 / (self.rrf_k + ranks_bm25[doc_id])
            if doc_id in ranks_vec:
                fused[doc_id] += 1.0 / (self.rrf_k + ranks_vec[doc_id])
        return dict(fused)

    def _fuse_weighted(self, bm25: Dict[str, float], vec: Dict[str, float]) -> Dict[str, float]:
        bm25_n = self._normalize(bm25)            # выше — лучше
        # vec уже нормализован в [0..1]
        vec_n = self._normalize(vec)
        all_ids = set(bm25_n) | set(vec_n)
        fused = {}
        a = float(self.alpha)
        for doc_id in all_ids:
            fused[doc_id] = a * bm25_n.get(doc_id, 0.0) + (1.0 - a) * vec_n.get(doc_id, 0.0)
        return fused

    # --- публичный метод ---

    def retrieve(self, query: str, k_final: int = 5) -> List[Tuple[RawDoc, float]]:
        bm25_scores = self._bm25_candidates(query, k=self.k1)        # выше — лучше
        vec_scores  = self._vector_candidates(query, k=self.k2)      # [0..1], выше — лучше

        if self.fusion == "rrf":
            fused = self._fuse_rrf(bm25_scores, vec_scores)
        else:
            fused = self._fuse_weighted(bm25_scores, vec_scores)

        # сортируем по fusion score
        ranked_ids = sorted(fused, key=fused.get, reverse=True)

        # --- опциональный реранк cross-encoder ---
        if self.cross_encoder and ranked_ids:
            cand_texts = [self.raw[i].text for i in ranked_ids]
            pairs = [(query, t) for t in cand_texts]
            ce_scores = self.cross_encoder.predict(pairs).tolist()  # выше — лучше
            # применим порог (если задан)
            tmp = [(doc_id, s) for doc_id, s in zip(ranked_ids, ce_scores)
                   if (self.ce_threshold is None or s >= self.ce_threshold)]
            # сортировка по CE
            tmp.sort(key=lambda x: x[1], reverse=True)
            ranked_ids = [doc_id for doc_id, _ in tmp]

        ranked_ids = ranked_ids[:k_final]
        return [(self.raw[i], float(fused.get(i, 0.0))) for i in ranked_ids]

                  
Код объединения результатов и их ранжирования с помощью выбранного кросс-энкодера.

# Создаём гибридный ретривер
hr = HybridRetriever(
    faiss_store=faiss,
    bm25_index=bm25,
    raw_docs=id2doc,
    fusion="rrf",          # попробуйте "weighted"
    alpha=0.6,             # работает, если fusion="weighted"
    k1=8, k2=8,            # кандидаты с каждой стороны
    rrf_k=60,
    cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",  # или None, если без реранка
    ce_threshold=None
)

queries = [
    "В каком году основана компания XYZ?",
    "Какова миссия компании?",
    "Как называется продукт, выпущенный в 2020 году?",
]

for q in queries:
    print(f"\nQ: {q}")
    results = hr.retrieve(q, k_final=3)
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [{doc.id}] score={score:.3f} → {doc.text[:90]}")

# --- простейшая метрика: contains-трюк (precision@5 surrogate) ---
ground_truth = {
    queries[0]: ["1999"],
    queries[1]: ["миссия", "доступным"],
    queries[2]: ["Alpha", "2020"],
}

def precision_at_k_contains(q: str, results: List[Tuple[RawDoc, float]], k: int = 5) -> float:
    want = ground_truth.get(q, [])
    if not want:
        return np.nan
    take = [t.lower() for t in want]
    hits = 0
    for doc, _ in results[:k]:
        text = doc.text.lower()
        if any(w in text for w in take):
            hits += 1
    return hits / min(k, len(results))

print("\n— Мини-оценка (contains surrogate):")
for q in queries:
    pa5 = precision_at_k_contains(q, hr.retrieve(q, k_final=5), k=5)
    print(f"{q} → precision@5≈{pa5:.2f}")

                  
Мы также добавили конфигурационные параметры: размер списков кандидатов от каждого источника (k1, k2), параметр α для взвешивания скоринга (если не используем RRF), порог отсечения для реранкера (например, можно отбросить документы, получившие очень низкий скор от кросс-энкодера, как нерелевантные).

Тестирование: Запустив наш гибридный ретривер на примере вопросов, мы увидим, что в сложных случаях он находит более полезную информацию. Например, на запрос, содержащий редкие термины, сработает BM25, а на запрос с обобщающей формулировкой – эмбеддинги. Реранкер же отфильтрует случайные неточные попадания. В логах или отчёте качества (например, сравнении метрик precision@5 до и после) можно отразить выигрыш. Артефакт hybrid_retriever.py готов к использованию в дальнейшем – мы будем применять его при построении комплексных цепочек, где нужен надежный поиск по своим данным.

Task:
Изучи контекст и напиши класс который реализует гибридный поиск в соответветсвии с текущими лучшими практиками построения rag и последней документацией Langchain
в качестве основы llm и эмбеденгов исользуй пример из basic_naive_rag



Constraint: 
    - отвечай только на тот вопрос, что тебя попросили 
    - не добавляй в ответ, ничего лишнего 
    - не создавай примеров использования, readme и прочего, если тебя явно об этом не попросили
    - НИКОГДА не пиши демонстрационных примеров, примеров использования, инструкций и readme, пока я сам тебя об этом не попрошу 
    - Используй последнюю версию документации Langchain 
