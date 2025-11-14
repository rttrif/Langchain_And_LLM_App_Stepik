import hashlib
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================
# TTL + LRU кэш (в памяти)
# =========================

@dataclass
class TTLCache:
    ttl_sec: float = 3600.0
    max_items: int = 256
    _store: Dict[str, Tuple[float, Any]] = field(default_factory=dict)
    _order: List[str] = field(default_factory=list)

    def _now(self) -> float:
        return time.time()

    def _prune(self) -> None:
        # удалить протухшие
        now = self._now()
        expired = [k for k, (ts, _) in self._store.items() if now > ts]
        for k in expired:
            self._store.pop(k, None)
            if k in self._order:
                self._order.remove(k)
        # LRU — если переполнено
        while len(self._store) > self.max_items:
            old = self._order.pop(0)
            self._store.pop(old, None)

    def get(self, key: str) -> Optional[Any]:
        self._prune()
        if key not in self._store:
            return None
        exp, val = self._store[key]
        if self._now() > exp:
            self._store.pop(key, None)
            if key in self._order:
                self._order.remove(key)
            return None
        # LRU bump
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)
        return val

    def set(self, key: str, value: Any) -> None:
        self._prune()
        exp = self._now() + self.ttl_sec
        self._store[key] = (exp, value)
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def clear(self) -> None:
        self._store.clear()
        self._order.clear()


CACHE = TTLCache(ttl_sec=3600, max_items=512)

def _cache_key(query: str, k: int) -> str:
    h = hashlib.sha256(f"{query}|{k}".encode("utf-8")).hexdigest()[:16]
    return f"websearch:{h}"


# =========================
# Провайдер поиска (DDG)
# =========================

def _ddg_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    DuckDuckGo text search. Возвращает [{title,url,snippet}, ...].
    """
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    out: List[Dict[str, str]] = []
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results, region="wt-wt", safesearch="off")
        for r in results:
            out.append({
                "title": r.get("title") or "",
                "url": r.get("href") or r.get("url") or "",
                "snippet": r.get("body") or r.get("snippet") or "",
            })
    return out

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=0.4, max=3.0))
def _search_with_retries(query: str, k: int) -> List[Dict[str, str]]:
    return _ddg_search(query, max_results=k)


# =========================
# Инструмент для LangChain
# =========================

@tool("web_search", return_direct=False)
def web_search(query: str, k: int = 5, use_cache: bool = True) -> str:
    """
    Поиск в интернете (DuckDuckGo). Ввод: query (строка), k (число результатов), use_cache (bool).
    Вывод: JSON {"ok":bool, "cached":bool, "results":[{title,url,snippet,rank}], "query":str}.
    """
    try:
        cached = False
        if use_cache:
            key = _cache_key(query, k)
            hit = CACHE.get(key)
            if hit is not None:
                cached = True
                results = hit
            else:
                results = _search_with_retries(query, k)
                CACHE.set(key, results)
        else:
            results = _search_with_retries(query, k)

        # пронумеруем для удобного цитирования [1], [2], ...
        norm = []
        for i, r in enumerate(results, start=1):
            norm.append({
                "rank": i,
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("snippet", "")
            })

        return json.dumps({"ok": True, "cached": cached, "query": query, "results": norm}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"search_failed: {e}", "query": query}, ensure_ascii=False)


# =========================
# Регистрация в агент
# =========================

def make_web_search_tools() -> List:
    return [web_search]


# =========================
# LLM-агент с веб-поиском
# =========================

class WebSearchAssistant:
    """
    Ассистент с веб-поиском на основе Ollama.
    Использует упрощенную схему: анализ вопроса -> поиск -> ответ с источниками.
    """

    def __init__(
        self,
        model: str = "deepseek-v3.1:671b-cloud",
        base_url: str = "https://ollama.com",
        temperature: float = 0.0
    ):
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature
        )

    def ask(self, query: str, k: int = 5) -> str:
        """
        Задать вопрос ассистенту.
        Автоматически выполняет поиск и формирует ответ.
        """
        try:
            # Выполняем поиск
            search_result = web_search.invoke({"query": query, "k": k})
            search_data = json.loads(search_result)

            if not search_data.get("ok"):
                return f"Ошибка поиска: {search_data.get('error', 'unknown')}"

            results = search_data["results"]

            # Если результатов нет
            if not results:
                return "Поиск не вернул результатов. Попробуйте переформулировать вопрос."

            # Форматируем результаты для LLM
            formatted_results = "\n\n".join([
                f"[{r['rank']}] {r['title']}\n{r['snippet']}\nURL: {r['url']}"
                for r in results
            ])

            # Создаем промпт для ответа
            template = """Ты - экспертный ассистент. На основе результатов веб-поиска дай полный и точный ответ на вопрос.

ВАЖНО:
- Используй только информацию из результатов поиска
- Указывай номера источников [1], [2] после каждого факта
- Если информация неполная или противоречива, укажи это
- Структурируй ответ логично
- Не придумывай информацию

Вопрос: {question}

Результаты поиска:
{search_results}

Ответ:"""

            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | StrOutputParser()

            answer = chain.invoke({
                "question": query,
                "search_results": formatted_results
            })

            # Добавляем список источников
            sources = "\n\nИсточники:\n" + "\n".join([
                f"[{r['rank']}] {r['title']} - {r['url']}"
                for r in results
            ])

            return answer + sources

        except Exception as e:
            logging.error(f"Error in ask: {e}")
            return f"Ошибка при обработке запроса: {e}"


def create_search_chain():
    """
    Создает простую цепочку: поиск → форматирование → LLM → ответ.
    Без агента, для более простых случаев.
    """
    llm = ChatOllama(
        model="deepseek-v3.1:671b-cloud",
        base_url="https://ollama.com",
        temperature=0.0
    )

    template = """На основе результатов веб-поиска дай точный и полный ответ на вопрос.

ВАЖНО:
- Используй только информацию из результатов поиска
- Указывай номера источников [1], [2] после каждого факта
- Если информации недостаточно, так и скажи
- Не придумывай информацию

Вопрос: {question}

Результаты поиска:
{search_results}

Ответ:"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain


def answer_with_search(question: str, k: int = 5) -> str:
    """
    Простая функция: выполняет поиск и формирует ответ через LLM.
    """
    search_result = web_search.invoke({"query": question, "k": k})
    search_data = json.loads(search_result)

    if not search_data.get("ok"):
        return f"Ошибка поиска: {search_data.get('error', 'unknown')}"

    results = search_data["results"]
    formatted_results = "\n\n".join([
        f"[{r['rank']}] {r['title']}\n{r['snippet']}\nURL: {r['url']}"
        for r in results
    ])

    chain = create_search_chain()
    answer = chain.invoke({
        "question": question,
        "search_results": formatted_results
    })

    sources = "\n\nИсточники:\n" + "\n".join([
        f"[{r['rank']}] {r['title']} - {r['url']}"
        for r in results
    ])

    return answer + sources


# =========================
# Пример использования
# =========================

if __name__ == "__main__":
    print("=== Тест 1: Простая цепочка (поиск + LLM) ===\n")
    question1 = "Что такое график Footprint в тейдинге?"
    print(f"Вопрос: {question1}\n")
    answer1 = answer_with_search(question1, k=3)
    print(f"Ответ:\n{answer1}\n")
    print("=" * 80)

    print("\n=== Тест 2: Web Search Assistant (агент) ===\n")
    assistant = WebSearchAssistant()
    question2 = "Как написать код для графика Footprint на python?"
    print(f"Вопрос: {question2}\n")
    answer2 = assistant.ask(question2)
    print(f"Ответ:\n{answer2}\n")
    print("=" * 80)

    print("\n=== Тест 3: Кэширование ===\n")
    print("Первый запрос (MISS):")
    out1 = web_search.invoke({"query": "latest AI news 2025", "k": 3})
    data1 = json.loads(out1)
    print(f"Cached: {data1.get('cached', False)}")

    print("\nВторой запрос (HIT):")
    out2 = web_search.invoke({"query": "latest AI news 2025", "k": 3})
    data2 = json.loads(out2)
    print(f"Cached: {data2.get('cached', False)}")
