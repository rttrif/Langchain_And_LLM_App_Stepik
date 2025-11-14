# llm_client.py
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dotenv import load_dotenv

load_dotenv()

# =========================
# Исключения
# =========================
class LLMError(Exception):
    """Базовая ошибка LLM-вызова."""

class CircuitOpen(LLMError):
    """Circuit breaker открыт: вызовы временно блокируются."""

# =========================
# Провайдеры (стратегии вызова)
# =========================
# Провайдер должен быть вызываемым: (prompt: str) -> str
LLMProvider = Callable[[str], str]

# Пример реального провайдера через LangChain (опционально)
# pip install langchain-openai
def make_openai_chat_provider(model: str = "gpt-4o-mini", temperature: float = 0.0) -> LLMProvider:
    """
    Возвращает провайдер, который вызывает OpenAI/совместимые API через LangChain.
    Требуется переменная окружения OPENAI_API_KEY (или укажите base_url для прокси).
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
    except Exception as e:
        raise RuntimeError("langchain-openai не установлен. Установите: pip install langchain-openai") from e

    llm = ChatOpenAI(model=model, temperature=temperature)

    def _call(prompt: str) -> str:
        msgs = [SystemMessage(content="Отвечай кратко и по делу."),
                HumanMessage(content=prompt)]
        out = llm.invoke(msgs)
        return out.content.strip()

    return _call

def make_ollama_provider(
    model: str = "deepseek-v3.1:671b-cloud",
    base_url: str = "https://ollama.com",
    temperature: float = 0.0
) -> LLMProvider:
    """
    Возвращает провайдер, который вызывает Ollama через LangChain.

    Для облачных моделей (с суффиксом -cloud):
        base_url="https://ollama.com"
        model="deepseek-v3.1:671b-cloud"
        OLLAMA_API_KEY берётся автоматически из .env

    Для локального Ollama:
        base_url="http://localhost:11434"
        model="llama3.2" (или другая установленная модель)
    """
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import SystemMessage, HumanMessage
    except Exception as e:
        raise RuntimeError("langchain-ollama не установлен. Установите: pip install langchain-ollama") from e

    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)

    def _call(prompt: str) -> str:
        msgs = [SystemMessage(content="Отвечай кратко и по делу."),
                HumanMessage(content=prompt)]
        out = llm.invoke(msgs)
        return out.content.strip()

    return _call

# =========================
# Конфиг надёжности
# =========================
@dataclass
class ReliabilityConfig:
    timeout_s: float = 15.0             # общий таймаут одного запроса
    retry_attempts: int = 3             # сколько попыток на один провайдер
    retry_sleep_base_s: float = 1.0     # старт ожидания между попытками
    retry_sleep_max_s: float = 8.0      # максимум ожидания
    retry_jitter_s: float = 0.25        # небольшой "джиттер"
    circuit_threshold: int = 5          # сколько подряд фейлов, чтобы открыть брейкер
    circuit_open_base_s: float = 30.0   # как долго держать "open" в первый раз
    circuit_multiplier: float = 2.0     # во сколько раз увеличивается hold при повторных триггерах
    circuit_open_max_s: float = 10 * 60 # абсолютный максимум hold

# =========================
# Клиент с ретраями и брейкером
# =========================
@dataclass
class LLMClient:
    providers: List[LLMProvider]
    config: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("llm_client"))

    # состояние circuit breaker'a
    failure_count: int = 0
    circuit_open_until: float = 0.0
    breaker_strikes: int = 0            # сколько раз уже открывали
    provider_index: int = 0             # текущий активный провайдер (для простого фолбэка)

    # --- публичный метод ---
    def send_request(self, prompt: str) -> str:
        now = time.time()
        if now < self.circuit_open_until:
            # брейкер открыт → либо сразу фейлим, либо можно попробовать "фолбэк-провайдеры"
            raise CircuitOpen(f"Circuit open until {time.ctime(self.circuit_open_until)}")

        # half-open: если период закончился, даём попытку и по результату закрываем/открываем
        try:
            out = self._attempt_with_fallbacks(prompt)
            # успех: сбросим счётчики
            self._reset_breaker()
            return out
        except CircuitOpen:
            raise
        except Exception as e:
            # финальный провал — учтём в брейкере
            self._register_failure()
            raise

    # --- внутренняя логика ---
    def _attempt_with_fallbacks(self, prompt: str) -> str:
        """
        Пытаемся вызвать текущего провайдера с ретраями; если не вышло — переключаемся
        на запасного и продолжаем. Возвращаем первый успешный ответ, иначе прокидываем исключение.
        """
        last_error: Optional[Exception] = None

        start_provider = self.provider_index
        tried = 0
        while tried < len(self.providers):
            idx = (start_provider + tried) % len(self.providers)
            prov = self.providers[idx]
            try:
                self.logger.debug(f"LLM call via provider[{idx}]")
                res = self._retry_call(prov, prompt)
                # если ходили не на первичном — можно вернуть указатель назад
                self.provider_index = idx
                return res
            except Exception as e:
                last_error = e
                self.logger.warning(f"Provider[{idx}] failed: {e}")
                tried += 1
                # переключаемся на следующего провайдера
                continue

        # все провайдеры провалились
        assert last_error is not None
        raise last_error

    def _retry_call(self, provider: LLMProvider, prompt: str) -> str:
        delay = self.config.retry_sleep_base_s
        attempts_left = self.config.retry_attempts
        while attempts_left > 0:
            attempts_left -= 1
            try:
                return self._call_with_timeout(provider, prompt, self.config.timeout_s)
            except (FuturesTimeout, TimeoutError) as e:
                self.logger.info(f"Timeout: {e}. Attempts left: {attempts_left}")
                if attempts_left <= 0:
                    raise LLMError("Timeout after retries") from e
            except Exception as e:
                # любые другие сетевые/API-ошибки
                self.logger.info(f"LLM error: {e}. Attempts left: {attempts_left}")
                if attempts_left <= 0:
                    raise LLMError("LLM failed after retries") from e

            # экспоненциальная пауза с джиттером
            time.sleep(min(delay, self.config.retry_sleep_max_s))
            delay = min(delay * 2, self.config.retry_sleep_max_s) + self.config.retry_jitter_s

        # формально unreachable
        raise LLMError("Unreachable retry state")

    def _call_with_timeout(self, provider: LLMProvider, prompt: str, timeout_s: float) -> str:
        # вызываем провайдера в отдельном потоке, чтобы enforceить таймаут
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(provider, prompt)
            return fut.result(timeout=timeout_s)

    # --- circuit breaker helpers ---
    def _reset_breaker(self) -> None:
        if self.failure_count > 0 or self.circuit_open_until > 0 or self.breaker_strikes > 0:
            self.logger.info("Breaker reset")
        self.failure_count = 0
        self.circuit_open_until = 0.0
        self.breaker_strikes = 0

    def _register_failure(self) -> None:
        self.failure_count += 1
        self.logger.debug(f"Failure count = {self.failure_count}")
        if self.failure_count >= self.config.circuit_threshold:
            # открыть брейкер
            self.breaker_strikes += 1
            hold = self.config.circuit_open_base_s * (self.config.circuit_multiplier ** (self.breaker_strikes - 1))
            hold = min(hold, self.config.circuit_open_max_s)
            self.circuit_open_until = time.time() + hold
            self.logger.warning(f"Circuit OPEN for {int(hold)}s (until {time.ctime(self.circuit_open_until)})")
            # не обнуляем failure_count, чтобы серия оставалась видимой
        else:
            self.logger.debug("Threshold not reached yet")

# =========================
# Тестовые симуляторы
# =========================
def sleepy_provider(delay_s: float) -> LLMProvider:
    """Провайдер, который просто спит и возвращает моковый ответ (для симуляции таймаута)."""
    def _call(prompt: str) -> str:
        time.sleep(delay_s)
        return f"[sleepy ok after {delay_s:.1f}s] {prompt[:40]}..."
    return _call

def flaky_provider(fail_times: int, succeed_text: str = "OK") -> LLMProvider:
    """Провайдер, который первые N раз падает исключением, потом начинает работать."""
    state = {"left": fail_times}
    def _call(prompt: str) -> str:
        if state["left"] > 0:
            state["left"] -= 1
            raise RuntimeError("simulated failure")
        return succeed_text
    return _call

def always_fail_provider() -> LLMProvider:
    def _call(prompt: str) -> str:
        raise RuntimeError("always fails")
    return _call

# =========================
# Демонстрация
# =========================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # --- Сценарий 1: таймаут и ретраи ---
    print("\n=== Scenario 1: Timeout & retries ===")
    cfg = ReliabilityConfig(timeout_s=1.5, retry_attempts=2, retry_sleep_base_s=0.5, retry_sleep_max_s=1.0)
    # sleepy 2s → будет таймаутиться (1.5s) и ретраиться
    client = LLMClient(providers=[sleepy_provider(2.0)], config=cfg)
    try:
        print("Response:", client.send_request("Hello timeout"))
    except Exception as e:
        print("Final error:", type(e).__name__, e)

    # --- Сценарий 2: серия ошибок → circuit open ---
    print("\n=== Scenario 2: Failures → circuit open ===")
    cfg2 = ReliabilityConfig(timeout_s=1.0, retry_attempts=1, circuit_threshold=3, circuit_open_base_s=3, circuit_multiplier=2)
    client2 = LLMClient(providers=[always_fail_provider()], config=cfg2)
    for i in range(5):
        try:
            print(f"Call {i+1} → ", client2.send_request("boom"))
        except Exception as e:
            print(f"Call {i+1} error → {type(e).__name__}: {e}")

    # Попробуем позвонить во время open
    try:
        print("During open:", client2.send_request("boom again"))
    except Exception as e:
        print("During open error →", type(e).__name__, e)

    # Подождём пока half-open позволит попробовать
    print("Sleeping 4s to pass open window...")
    time.sleep(4)
    try:
        print("Half-open attempt →", client2.send_request("still boom?"))
    except Exception as e:
        print("Half-open error →", type(e).__name__, e)

    # --- Сценарий 3: фолбэк провайдера ---
    print("\n=== Scenario 3: Fallback provider ===")
    # Первый всегда падает, второй — флакит (2 раза упадёт, потом OK)
    client3 = LLMClient(
        providers=[always_fail_provider(), flaky_provider(2, succeed_text="SECOND OK")],
        config=ReliabilityConfig(timeout_s=2.0, retry_attempts=2, retry_sleep_base_s=0.3)
    )
    try:
        print("Response:", client3.send_request("use fallback"))
    except Exception as e:
        print("Final error:", type(e).__name__, e)

    # --- Сценарий 4: Ollama Cloud (облачные модели) ---
    print("\n=== Scenario 4: Ollama Cloud provider ===")
    print("(OLLAMA_API_KEY берётся из .env)")
    try:
        ollama_cloud = make_ollama_provider(
            model="deepseek-v3.1:671b-cloud",
            base_url="https://ollama.com",
            temperature=0.0
        )
        client4 = LLMClient(
            providers=[ollama_cloud],
            config=ReliabilityConfig(timeout_s=60.0, retry_attempts=2)
        )
        response = client4.send_request("Что такое circuit breaker pattern? Ответь одним предложением.")
        print(f"Ollama Cloud response: {response}")
    except Exception as e:
        print(f"Ollama Cloud error: {type(e).__name__}: {e}")

    # --- Сценарий 5: фолбэк с облачного Ollama на mock ---
    print("\n=== Scenario 5: Ollama Cloud with mock fallback ===")
    try:
        mock_provider = flaky_provider(0, succeed_text="MOCK: Retry pattern - повтор операции после ошибки.")
        ollama_cloud2 = make_ollama_provider(
            model="deepseek-v3.1:671b-cloud",
            base_url="https://ollama.com",
            temperature=0.0
        )
        client5 = LLMClient(
            providers=[ollama_cloud2, mock_provider],
            config=ReliabilityConfig(timeout_s=60.0, retry_attempts=1)
        )
        response5 = client5.send_request("Объясни retry pattern в одном предложении.")
        print(f"Response with fallback: {response5}")
    except Exception as e:
        print(f"Fallback test error: {type(e).__name__}: {e}")

    # --- Сценарий 7: реальный OpenAI (если хотите раскомментировать) ---
    # print("\n=== Scenario 7: Real OpenAI provider ===")
    # provider = make_openai_chat_provider(model="gpt-4o-mini", temperature=0.0)
    # client7 = LLMClient(providers=[provider])
    # print("Real model:", client7.send_request("Скажи одно слово: 'готово'"))
