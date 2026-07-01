"""Multi-provider LLM service with unified interface and smart model selection."""
import asyncio
import base64
import logging
import os
from abc import ABC, abstractmethod
from functools import wraps
from typing import AsyncIterator, Optional

from app.config import (
    ANTHROPIC_DEFAULT_MODEL,
    DEEPSEEK_DEFAULT_MODEL,
    OPENAI_DEFAULT_MODEL,
    OLLAMA_BASE_URL, OLLAMA_DEFAULT_MODEL,
    LLM_MAX_TOKENS,
    get_default_provider,
)

logger = logging.getLogger(__name__)

DEFAULT_STUDY_SYSTEM = "你是清，一个智能学习助手，擅长总结资料、解释概念、制定复习计划并帮助用户查漏补缺。默认使用中文回答。"
DEFAULT_VISION_SYSTEM = "你负责准确提取图片或扫描资料中的学习内容，并整理成适合总结和复习的文本。"

# ── Retry helper ──

def _retry(max_retries: int = 3, base_delay: float = 1.0):
    """Async retry decorator with exponential backoff for transient errors."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()
                    # Only retry on transient errors
                    is_transient = any(k in err_str for k in (
                        "rate", "timeout", "connection", "overloaded",
                        "429", "500", "502", "503", "504", "server error",
                    ))
                    if not is_transient or attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
            raise last_error
        return wrapper
    return decorator

# ── Key helpers — always read from os.environ so runtime changes take effect ──

def _get_key(provider_id: str) -> str:
    """Get API key from environment (always live, never cached)."""
    keys = {
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "ollama": "",
    }
    env_var = keys.get(provider_id, "")
    return os.environ.get(env_var, "") if env_var else "local"


class LLMProvider(ABC):
    """Unified interface for all LLM backends."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def chat(self, system: str, messages: list[dict]) -> str: ...

    @abstractmethod
    async def chat_stream(self, system: str, messages: list[dict]) -> AsyncIterator[str]: ...

    @abstractmethod
    async def chat_with_image(self, system: str, messages: list[dict], image_b64: str, mime: str) -> str: ...

    @abstractmethod
    def has_vision(self) -> bool: ...

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...


# ── Anthropic ──

class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = None):
        super().__init__(model or ANTHROPIC_DEFAULT_MODEL)
        import anthropic
        key = _get_key("anthropic")
        self.client = anthropic.AsyncAnthropic(api_key=key)

    def has_vision(self) -> bool:
        return True

    def count_tokens(self, text: str) -> int:
        try:
            return self.client.count_tokens(text)
        except Exception:
            return len(text) // 4

    @_retry()
    async def chat(self, system: str, messages: list[dict]) -> str:
        try:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=LLM_MAX_TOKENS,
                system=system or DEFAULT_STUDY_SYSTEM,
                messages=self._convert_messages(messages),
            )
            return resp.content[0].text
        except Exception as e:
            logger.error(f"Anthropic chat error: {e}")
            raise

    async def chat_stream(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        try:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=LLM_MAX_TOKENS,
                system=system or DEFAULT_STUDY_SYSTEM,
                messages=self._convert_messages(messages),
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Anthropic stream error: {e}")
            yield f"\n\n[Error: {e}]"

    @_retry()
    async def chat_with_image(self, system: str, messages: list[dict], image_b64: str, mime: str) -> str:
        content = []
        for m in messages:
            content.append({"type": "text", "text": m["content"]})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": image_b64},
        })
        try:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=LLM_MAX_TOKENS,
                system=system or DEFAULT_VISION_SYSTEM,
                messages=[{"role": "user", "content": content}],
            )
            return resp.content[0].text
        except Exception as e:
            logger.error(f"Anthropic vision error: {e}")
            raise

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        converted = []
        for m in messages:
            role = m.get("role", "user")
            if role == "system":
                continue
            if role == "assistant":
                role = "assistant"
            converted.append({"role": role, "content": m.get("content", "")})
        if not converted:
            converted.append({"role": "user", "content": "Hello"})
        return converted


# ── DeepSeek (OpenAI-compatible) ──

class DeepSeekProvider(LLMProvider):
    def __init__(self, model: str = None):
        super().__init__(model or "deepseek-v4-pro")
        from openai import AsyncOpenAI
        key = _get_key("deepseek")
        self.client = AsyncOpenAI(api_key=key, base_url="https://api.deepseek.com")

    def has_vision(self) -> bool:
        return False

    def count_tokens(self, text: str) -> int:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return len(text) // 4

    @_retry()
    async def chat(self, system: str, messages: list[dict]) -> str:
        try:
            msgs = self._build_messages(system, messages)
            resp = await self.client.chat.completions.create(
                model=self.model, messages=msgs, max_tokens=LLM_MAX_TOKENS,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"DeepSeek chat error: {e}")
            raise

    async def chat_stream(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        try:
            msgs = self._build_messages(system, messages)
            stream = await self.client.chat.completions.create(
                model=self.model, messages=msgs, max_tokens=LLM_MAX_TOKENS, stream=True,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"DeepSeek stream error: {e}")
            yield f"\n\n[Error: {e}]"

    @_retry()
    async def chat_with_image(self, system: str, messages: list[dict], image_b64: str, mime: str) -> str:
        raise NotImplementedError("DeepSeek does not support vision.")

    def _build_messages(self, system: str, messages: list[dict]) -> list[dict]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        for m in messages:
            msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        if not msgs:
            msgs.append({"role": "user", "content": "Hello"})
        return msgs


# ── OpenAI ──

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = None):
        super().__init__(model or OPENAI_DEFAULT_MODEL)
        from openai import AsyncOpenAI
        key = _get_key("openai")
        self.client = AsyncOpenAI(api_key=key)

    def has_vision(self) -> bool:
        return "gpt-4o" in self.model or "gpt-4.1" in self.model or "vision" in self.model.lower()

    def count_tokens(self, text: str) -> int:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return len(text) // 4

    @_retry()
    async def chat(self, system: str, messages: list[dict]) -> str:
        try:
            msgs = self._build_messages(system, messages)
            resp = await self.client.chat.completions.create(model=self.model, messages=msgs, max_tokens=LLM_MAX_TOKENS)
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            raise

    async def chat_stream(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        try:
            msgs = self._build_messages(system, messages)
            stream = await self.client.chat.completions.create(model=self.model, messages=msgs, max_tokens=LLM_MAX_TOKENS, stream=True)
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI stream error: {e}")
            yield f"\n\n[Error: {e}]"

    @_retry()
    async def chat_with_image(self, system: str, messages: list[dict], image_b64: str, mime: str) -> str:
        content = []
        for m in messages:
            content.append({"type": "text", "text": m["content"]})
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}})
        try:
            msgs = [{"role": "system", "content": system}] if system else []
            msgs.append({"role": "user", "content": content})
            resp = await self.client.chat.completions.create(model=self.model, messages=msgs, max_tokens=LLM_MAX_TOKENS)
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI vision error: {e}")
            raise

    def _build_messages(self, system: str, messages: list[dict]) -> list[dict]:
        msgs = [{"role": "system", "content": system}] if system else []
        for m in messages:
            msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        return msgs


# ── Ollama ──

class OllamaProvider(LLMProvider):
    def __init__(self, model: str = None):
        super().__init__(model or OLLAMA_DEFAULT_MODEL)
        import httpx
        self.http = httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=120.0)

    def has_vision(self) -> bool:
        return "llava" in self.model.lower() or "bakllava" in self.model.lower()

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    @_retry()
    async def chat(self, system: str, messages: list[dict]) -> str:
        try:
            msgs = [{"role": "system", "content": system}] if system else []
            for m in messages:
                msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
            resp = await self.http.post("/api/chat", json={"model": self.model, "messages": msgs, "stream": False})
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            raise

    async def chat_stream(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        try:
            msgs = [{"role": "system", "content": system}] if system else []
            for m in messages:
                msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
            async with self.http.stream("POST", "/api/chat", json={"model": self.model, "messages": msgs, "stream": True}) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama stream error: {e}")
            yield f"\n\n[Error: {e}]"

    async def chat_with_image(self, system: str, messages: list[dict], image_b64: str, mime: str) -> str:
        raise NotImplementedError("This Ollama model does not support vision.")


# ── Factory ──

_PROVIDER_CLASSES = {
    "anthropic": AnthropicProvider,
    "deepseek": DeepSeekProvider,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
}


def get_available_providers() -> list[dict]:
    """Return list of configured providers with their models."""
    providers = []
    if _get_key("anthropic"):
        providers.append({"id": "anthropic", "name": "Anthropic", "models": ["claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5"], "vision": True})
    if _get_key("deepseek"):
        providers.append({"id": "deepseek", "name": "DeepSeek", "models": [
            "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner",
        ], "vision": False})
    if _get_key("openai"):
        providers.append({"id": "openai", "name": "OpenAI", "models": ["gpt-4.1", "gpt-4.1-mini", "o3"], "vision": True})
    providers.append({"id": "ollama", "name": "Ollama (local)", "models": ["llama3.2", "llava"], "vision": False, "local": True})
    return providers


def create_llm(provider_id: str = None, model: str = None) -> LLMProvider:
    """Create LLM provider instance. Falls back to first available."""
    pid = provider_id or get_default_provider()
    key = _get_key(pid)

    # If not configured, find first available
    if not key and pid != "ollama":
        for alt in ["anthropic", "deepseek", "openai"]:
            if _get_key(alt):
                pid = alt
                key = _get_key(alt)
                break

    if not key and pid != "ollama":
        raise ValueError(f"No API key configured. Please add at least one in Settings.")

    cls = _PROVIDER_CLASSES.get(pid, AnthropicProvider)
    return cls(model)


# ── Cost-aware model selection for background tasks ──

_CHEAP_TASKS = {"classify", "rerank", "compress", "mnemonic", "flashcard"}
_QUALITY_TASKS = {"summary", "answer", "plan", "mindmap", "compare", "formula", "practice", "sprint"}


def get_task_llm(task: str, provider_id: str = None, model: str = None) -> LLMProvider:
    """Select appropriate LLM for a task — cheap for simple, quality for complex.

    For quality tasks: respects user's provider/model choice if given.
    For cheap tasks: always uses the cheapest available model to save cost.
    """
    if task in _CHEAP_TASKS:
        # Cheap tasks: always use the cheapest model regardless of user preference
        if _get_key("deepseek"):
            return DeepSeekProvider("deepseek-v4-pro")
        if _get_key("anthropic"):
            return AnthropicProvider("claude-haiku-4-5")
        if _get_key("openai"):
            return OpenAIProvider("gpt-4.1-mini")
        return OllamaProvider()

    # Quality tasks: prefer user's choice, then fall back to best available
    pid = provider_id or get_default_provider()

    # If user specified a provider that has a key, use it
    if pid and _get_key(pid):
        cls = _PROVIDER_CLASSES.get(pid)
        if cls:
            return cls(model)

    # Fallback: find the best available provider
    if _get_key("anthropic"):
        return AnthropicProvider(model)
    if _get_key("openai"):
        return OpenAIProvider(model)
    if _get_key("deepseek"):
        return DeepSeekProvider(model)
    return OllamaProvider(model)
