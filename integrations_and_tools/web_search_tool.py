import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

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
    from duckduckgo_search import DDGS  # импорт внутри, чтобы не требовать его, если инструмент не используется
    out: List[Dict[str, str]] = []
    # timelimit="y" — за последний год; можно убрать/поменять
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results, safesearch="moderate", timelimit="y"):
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
# Пример использования
# =========================

if __name__ == "__main__":
    # 1) голый вызов инструмента (2 запроса подряд → второй из кэша)
    print("— First call (MISS):")
    out1 = web_search.run("president of Brazil 2025", k=3)
    print(out1[:300], "...\n")

    print("— Second call (HIT):")
    out2 = web_search.run("president of Brazil 2025", k=3)
    print(out2[:300], "...\n")

    # 2) мини-«диалог» в стиле ReAct (симулируем логи)
    # В реальном агенте эти шаги делаются автоматически, здесь — иллюстрация.
    print("=== Demo: assistant with web_search ===")
    user_q = "Кто сейчас является президентом Бразилии?"
    print(f"User: {user_q}")

    print('Assistant (thinking): локальной информации нет → Action: web_search')
    ws_json = web_search.run("президент Бразилии 2025", k=5)
    ws = json.loads(ws_json)
    if ws.get("ok"):
        results = ws["results"]
        # Сформируем краткий ответ + цитаты [1], [2]
        # (В реальном ассистенте ответ генерирует LLM, здесь просто пример форматирования)
        guess_name = None
        for r in results:
            t = (r["title"] + " " + r["snippet"]).lower()
            if "lula" in t or "лула" in t or "luiz inácio" in t or "луис инасиу" in t:
                guess_name = "Луис Инасиу Лула да Силва"
                break

        if guess_name:
            # сослаться на первый/лучший источник
            cite = f"[1] {results[0]['title']} — {results[0]['url']}"
            print(f"Assistant: По данным веб-поиска, президент Бразилии сейчас — {guess_name}. См. источники ниже.")
            print("Sources:")
            for r in results[:3]:
                print(f"  [{r['rank']}] {r['title']} — {r['url']}")
        else:
            print("Assistant: Похоже, результаты неоднозначны. Нужен дополнительный уточняющий запрос.")
    else:
        print("Assistant: Не удалось выполнить веб-поиск. Попробуйте позже.")
