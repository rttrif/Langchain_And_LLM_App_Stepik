
import ast
import math
import os
import pathlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from langchain.tools import tool
from langchain_ollama import ChatOllama

llm = ChatOllama(model="deepseek-v3.1:671b-cloud",
                 base_url="https://ollama.com",
                 temperature=0.0)

# ---------------------------
# Общие настройки/логгер
# ---------------------------

@dataclass
class ToolConfig:
    weather_timeout_s: float = 10.0
    search_timeout_s: float = 12.0
    calculator_timeout_s: float = 2.0
    file_root: str = "./"
    file_max_mb: int = 2
    file_allow_ext: tuple = (".txt", ".md", ".csv", ".json", ".yaml", ".yml")

CFG = ToolConfig()

# ---------------------------
# 1) CalculatorTool
# ---------------------------

_ALLOWED_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
_ALLOWED_NAMES.update({"abs": abs, "round": round, "min": min, "max": max})

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub,
    ast.Call, ast.Load, ast.Name,
)

def _safe_eval(expr: str) -> float:
    """Безопасная оценка арифметического выражения через AST."""
    node = ast.parse(expr, mode="eval")
    def _check(n: ast.AST):
        if not isinstance(n, _ALLOWED_NODES):
            raise ValueError(f"Недопустимая конструкция: {type(n).__name__}")
        for child in ast.iter_child_nodes(n):
            _check(child)
    _check(node)

    def _eval(n: ast.AST):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Num):        # Py<3.8
            return n.n
        if isinstance(n, ast.Constant):   # Py3.8+
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("Разрешены только числа")
        if isinstance(n, ast.BinOp):
            l, r = _eval(n.left), _eval(n.right)
            if isinstance(n.op, ast.Add):  return l + r
            if isinstance(n.op, ast.Sub):  return l - r
            if isinstance(n.op, ast.Mult): return l * r
            if isinstance(n.op, ast.Div):  return l / r
            if isinstance(n.op, ast.FloorDiv): return l // r
            if isinstance(n.op, ast.Mod):  return l % r
            if isinstance(n.op, ast.Pow):  return l ** r
        if isinstance(n, ast.UnaryOp):
            v = _eval(n.operand)
            if isinstance(n.op, ast.UAdd): return +v
            if isinstance(n.op, ast.USub): return -v
        if isinstance(n, ast.Call):
            func = _eval(n.func)
            args = [_eval(a) for a in n.args]
            if not all(isinstance(a, (int, float)) for a in args):
                raise ValueError("Функции принимают только числа")
            return func(*args)
        if isinstance(n, ast.Name):
            if n.id in _ALLOWED_NAMES:
                return _ALLOWED_NAMES[n.id]
            raise ValueError(f"Запрещено имя: {n.id}")
        raise ValueError("Недопустимое выражение")
    return float(_eval(node))

@tool("calculator", return_direct=False)
def calculator(expression: str) -> str:
    """
    Выполняет безопасные математические вычисления.
    Поддержка: +, -, *, /, //, %, **, а также функции из math: sin, cos, sqrt и т.п.
    Пример: "2*(3+4) + sqrt(16)"
    """
    start = time.time()
    try:
        # грубый таймаут на уровне инструмента
        res = _safe_eval(expression)
        if time.time() - start > CFG.calculator_timeout_s:
            return json.dumps({"ok": False, "error": "timeout"})
        return json.dumps({"ok": True, "result": res})
    except ZeroDivisionError:
        return json.dumps({"ok": False, "error": "division by zero"})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})

# ---------------------------
# 2) WebSearchTool (скелет + ретраи)
# ---------------------------

# По умолчанию используем DuckDuckGo (через библиотеку duckduckgo_search).
# Можно заменить на Tavily/Serper и т.п., просто поменяв клиент в _ddg_search().
def _ddg_search(query: str, max_results: int = 5, timeout: float = 8.0) -> List[Dict[str, Any]]:
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        # возвращаем title, href, body
        out = []
        for i, r in enumerate(ddgs.text(query, max_results=max_results, timelimit="y")):
            out.append({"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")})
        return out

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=0.5, max=3.0))
def _web_search_impl(query: str, k: int = 5) -> List[Dict[str, Any]]:
    return _ddg_search(query, max_results=k, timeout=CFG.search_timeout_s)

@tool("web_search", return_direct=False)
def web_search(query: str, k: int = 5) -> str:
    """
    Выполняет веб-поиск (DuckDuckGo). Возвращает top-k результатов: title, url, snippet.
    Встроены ретраи с экспоненциальным backoff.
    """
    try:
        results = _web_search_impl(query, k=k)
        return json.dumps({"ok": True, "results": results})
    except Exception as e:
        return json.dumps({"ok": False, "error": f"search_failed: {e}"})

# ---------------------------
# 3) WeatherTool (Open-Meteo, без ключей)
# ---------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=0.3, max=2.0))
def _geocode_city(city: str) -> Optional[Dict[str, float]]:
    # Open-Meteo Geocoding API
    url = "https://geocoding-api.open-meteo.com/v1/search"
    with httpx.Client(timeout=CFG.weather_timeout_s) as client:
        r = client.get(url, params={"name": city, "count": 1, "language": "ru", "format": "json"})
        r.raise_for_status()
        data = r.json()
        if data.get("results"):
            item = data["results"][0]
            return {"lat": item["latitude"], "lon": item["longitude"], "name": item["name"]}
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=0.3, max=2.0))
def _fetch_weather(lat: float, lon: float) -> Dict[str, Any]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "current_weather": True, "timezone": "auto"}
    with httpx.Client(timeout=CFG.weather_timeout_s) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        return r.json()

@tool("weather", return_direct=False)
def weather(city: str) -> str:
    """
    Погода по городу (Open-Meteo): текущая температура, скорость ветра и т.п.
    Пример: city="Brisbane"
    """
    try:
        loc = _geocode_city(city)
        if not loc:
            return json.dumps({"ok": False, "error": "city_not_found"})
        data = _fetch_weather(loc["lat"], loc["lon"])
        cur = data.get("current_weather", {})
        return json.dumps({"ok": True, "city": loc["name"], "lat": loc["lat"], "lon": loc["lon"], "current": cur})
    except Exception as e:
        return json.dumps({"ok": False, "error": f"weather_failed: {e}"})

# ---------------------------
# 4) FileTool (ограниченный доступ к файловой системе)
# ---------------------------

def _is_safe_path(root: str, path: str) -> bool:
    root_path = pathlib.Path(root).resolve()
    target = (root_path / path).resolve()
    return str(target).startswith(str(root_path))

@tool("read_file", return_direct=False)
def read_file(relative_path: str, encoding: str = "utf-8") -> str:
    """
    Читает файл из ограниченной папки (CFG.file_root).
    Ограничения: допустимые расширения, размер (<= file_max_mb).
    """
    try:
        if not _is_safe_path(CFG.file_root, relative_path):
            return json.dumps({"ok": False, "error": "path_not_allowed"})
        fp = pathlib.Path(CFG.file_root) / relative_path
        if not fp.exists() or not fp.is_file():
            return json.dumps({"ok": False, "error": "not_found"})
        if fp.suffix.lower() not in CFG.file_allow_ext:
            return json.dumps({"ok": False, "error": "extension_not_allowed"})
        if fp.stat().st_size > CFG.file_max_mb * 1024 * 1024:
            return json.dumps({"ok": False, "error": "file_too_large"})
        text = fp.read_text(encoding=encoding, errors="replace")
        return json.dumps({"ok": True, "path": str(fp), "content": text})
    except Exception as e:
        return json.dumps({"ok": False, "error": f"read_failed: {e}"})

# ---------------------------
# 5) CurrencyTool (exchangerate.host)
# ---------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=0.3, max=2.0))
def _fx_rate(base: str, quote: str) -> Optional[float]:
    url = "https://api.exchangerate.host/latest"
    with httpx.Client(timeout=8.0) as client:
        r = client.get(url, params={"base": base.upper(), "symbols": quote.upper()})
        r.raise_for_status()
        data = r.json()
        rates = data.get("rates") or {}
        return rates.get(quote.upper())

@tool("fx_rate", return_direct=False)
def fx_rate(base: str, quote: str) -> str:
    """
    Курс валют: сколько 1 {base} стоит в {quote}. Источник: exchangerate.host
    Пример: base="USD", quote="RUB"
    """
    try:
        val = _fx_rate(base, quote)
        if val is None:
            return json.dumps({"ok": False, "error": "rate_not_available"})
        return json.dumps({"ok": True, "pair": f"{base.upper()}/{quote.upper()}", "rate": val})
    except Exception as e:
        return json.dumps({"ok": False, "error": f"fx_failed: {e}"})


# ---------------------------
# Регистрация инструментов
# ---------------------------

def make_tools() -> List:
    """
    Возвращает список зарегистрированных инструментов для агента.
    Пример использования:
        tools = make_tools()
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = hub.pull("hwchase17/openai-tools-agent")
        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_exec = AgentExecutor(agent=agent, tools=tools, verbose=True)
    """
    return [calculator, web_search, weather, read_file, fx_rate]


# Тест «в лоб» (ручная проверка без LLM)
if __name__ == "__main__":
    print("Calculator:", calculator.run("2*(3+4) + sqrt(16)"))
    print("FX:", fx_rate.run("USD", "RUB"))
    print("Weather:", weather.run("Brisbane")[:200], "...")
    # Создайте ./sandbox и положите туда файл test.txt для примера:
    # print("Read:", read_file.run("test.txt")[:200])
    print("Search:", web_search.run("LangChain RAG hybrid search")[:200], "...")
