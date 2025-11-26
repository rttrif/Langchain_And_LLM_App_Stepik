# llm_client.py
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dotenv import load_dotenv

load_dotenv()


class LLMError(Exception):
    """Базовая ошибка LLM-вызова."""


class CircuitOpen(LLMError):
    """Circuit breaker открыт: вызовы временно блокируются."""


LLMProvider = Callable[[str], str]


def make_openai_chat_provider(model: str = "gpt-4o-mini", temperature: float = 0.0) -> LLMProvider:
    """Возвращает провайдер, который вызывает OpenAI/совместимые API через LangChain."""
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


@dataclass
class LLMClient:
    providers: List[LLMProvider]
    config: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("llm_client"))

    failure_count: int = 0
    circuit_open_until: float = 0.0
    breaker_strikes: int = 0            # сколько раз уже открывали
    provider_index: int = 0             # текущий активный провайдер (для простого фолбэка)

    def send_request(self, prompt: str) -> str:
        now = time.time()
        if now < self.circuit_open_until:
            raise CircuitOpen(f"Circuit open until {time.ctime(self.circuit_open_until)}")

        try:
            out = self._attempt_with_fallbacks(prompt)
            self._reset_breaker()
            return out
        except CircuitOpen:
            raise
        except Exception as e:
            self._register_failure()
            raise

    def _attempt_with_fallbacks(self, prompt: str) -> str:
        last_error: Optional[Exception] = None
        start_provider = self.provider_index
        tried = 0
        
        while tried < len(self.providers):
            idx = (start_provider + tried) % len(self.providers)
            prov = self.providers[idx]
            try:
                self.logger.debug(f"LLM call via provider[{idx}]")
                res = self._retry_call(prov, prompt)
                self.provider_index = idx
                return res
            except Exception as e:
                last_error = e
                self.logger.warning(f"Provider[{idx}] failed: {e}")
                tried += 1
                continue

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
                self.logger.info(f"LLM error: {e}. Attempts left: {attempts_left}")
                if attempts_left <= 0:
                    raise LLMError("LLM failed after retries") from e

            time.sleep(min(delay, self.config.retry_sleep_max_s))
            delay = min(delay * 2, self.config.retry_sleep_max_s) + self.config.retry_jitter_s

        raise LLMError("Unreachable retry state")

    def _call_with_timeout(self, provider: LLMProvider, prompt: str, timeout_s: float) -> str:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(provider, prompt)
            return fut.result(timeout=timeout_s)

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
            self.breaker_strikes += 1
            hold = self.config.circuit_open_base_s * (self.config.circuit_multiplier ** (self.breaker_strikes - 1))
            hold = min(hold, self.config.circuit_open_max_s)
            self.circuit_open_until = time.time() + hold
            self.logger.warning(f"Circuit OPEN for {int(hold)}s (until {time.ctime(self.circuit_open_until)})")
        else:
            self.logger.debug("Threshold not reached yet")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Сценарий 1: Базовый вызов с одним провайдером
    print("\n=== Сценарий 1: Базовый вызов Ollama ===")
    try:
        ollama_provider = make_ollama_provider(
            model="deepseek-v3.1:671b-cloud",
            base_url="https://ollama.com",
            temperature=0.0
        )
        
        client = LLMClient(
            providers=[ollama_provider],
            config=ReliabilityConfig(timeout_s=60.0, retry_attempts=2)
        )
        
        response = client.send_request("Что такое circuit breaker pattern? Ответь одним предложением.")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

    # Сценарий 2: Fallback между двумя провайдерами
    print("\n=== Сценарий 2: Fallback между Ollama моделями ===")
    try:
        primary = make_ollama_provider(model="deepseek-v3.1:671b-cloud", base_url="https://ollama.com")
        fallback = make_ollama_provider(model="qwen2.5:7b-cloud", base_url="https://ollama.com")
        
        client2 = LLMClient(
            providers=[primary, fallback],
            config=ReliabilityConfig(timeout_s=30.0, retry_attempts=1)
        )
        
        response2 = client2.send_request("Объясни retry pattern в одном предложении.")
        print(f"Response: {response2}")
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

    # Сценарий 3: Агрессивные таймауты для быстрых ответов
    print("\n=== Сценарий 3: Быстрый ответ с коротким таймаутом ===")
    try:
        fast_provider = make_ollama_provider(model="qwen2.5:7b-cloud", base_url="https://ollama.com")
        
        client3 = LLMClient(
            providers=[fast_provider],
            config=ReliabilityConfig(
                timeout_s=10.0,
                retry_attempts=2,
                retry_sleep_base_s=0.5,
                retry_sleep_max_s=2.0
            )
        )
        
        response3 = client3.send_request("Скажи 'готово'")
        print(f"Response: {response3}")
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

    # Сценарий 4: Высоконагруженная система с circuit breaker
    print("\n=== Сценарий 4: Circuit breaker при высокой нагрузке ===")
    try:
        prod_provider = make_ollama_provider(model="deepseek-v3.1:671b-cloud", base_url="https://ollama.com")
        
        client4 = LLMClient(
            providers=[prod_provider],
            config=ReliabilityConfig(
                timeout_s=45.0,
                retry_attempts=3,
                circuit_threshold=3,
                circuit_open_base_s=10.0,
                circuit_multiplier=1.5
            )
        )
        
        response4 = client4.send_request("Перечисли 3 паттерна надёжности для LLM.")
        print(f"Response: {response4}")
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
