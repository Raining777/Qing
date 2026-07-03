"""Multi-provider LLM service: Anthropic + OpenAI-compatible (OpenAI / DeepSeek / Ollama)."""
import asyncio
import base64
import logging
import os
from abc import ABC, abstractmethod
from functools import wraps
from typing import AsyncIterator

from app.config import (
    ANTHROPIC_DEFAULT_MODEL, DEEPSEEK_DEFAULT_MODEL,
    OPENAI_DEFAULT_MODEL, OLLAMA_BASE_URL, OLLAMA_DEFAULT_MODEL,
    LLM_MAX_TOKENS, get_default_provider,
)

logger = logging.getLogger(__name__)

DEFAULT_STUDY_PROMPT = "你是清，一个智能学习助手，擅长总结资料、解释概念、制定复习计划并帮助用户查漏补缺。默认使用中文回答。"
DEFAULT_VISION_PROMPT = "你负责准确提取图片或扫描资料中的学习内容，并整理成适合总结和复习的文本。"


# ── Retry helper ──

def _retry(max_retries: int = 3, base_delay: float = 1.0):
    """Async retry with exponential backoff for transient errors."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    err = str(e).lower()
                    transient = any(k in err for k in (
                        "rate", "timeout", "connection", "overloaded",
                        "429", "500", "502", "503", "504", "server error",
                    ))
                    if not transient or attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
            raise last_error
        return wrapper
    return decorator


# ── Key helpers ──

def _key(provider: str) -> str:
    """Get API key from environment (live read)."""
    keys = {"anthropic": "ANTHROPIC_API_KEY", "deepseek": "DEEPSEEK_API_KEY",
            "openai": "OPENAI_API_KEY", "ollama": ""}
    env = keys.get(provider, "")
    return os.environ.get(env, "") if env else "local"


# ── Abstract base ──

class LLMProvider(ABC):
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
        self.client = anthropic.AsyncAnthropic(api_key=_key("anthropic"))

    def has_vision(self) -> bool: return True

    def count_tokens(self, text: str) -> int:
        try:
            return self.client.count_tokens(text)
        except Exception:
            return len(text) // 4

    @_retry()
    async def chat(self, system: str, messages: list[dict]) -> str:
        resp = await self.client.messages.create(
            model=self.model, max_tokens=LLM_MAX_TOKENS,
            system=system or DEFAULT_STUDY_PROMPT,
            messages=self._convert(messages),
        )
        return resp.content[0].text

    async def chat_stream(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        try:
            async with self.client.messages.stream(
                model=self.model, max_tokens=LLM_MAX_TOKENS,
                system=system or DEFAULT_STUDY_PROMPT,
                messages=self._convert(messages),
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Anthropic stream error: {e}")
            yield f"\n\n[回答生成出错，请重试]"

    @_retry()
    async def chat_with_image(self, system: str, messages: list[dict], image_b64: str, mime: str) -> str:
        content = [{"type": "text", "text": m["content"]} for m in messages]
        content.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": image_b64}})
        resp = await self.client.messages.create(
            model=self.model, max_tokens=LLM_MAX_TOKENS,
            system=system or DEFAULT_VISION_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        return resp.content[0].text

    def _convert(self, messages: list[dict]) -> list[dict]:
        result = []
        for m in messages:
            role = m.get("role", "user")
            if role == "system": continue
            result.append({"role": "assistant" if role == "assistant" else "user",
                           "content": m.get("content", "")})
        return result or [{"role": "user", "content": "Hello"}]


# ── OpenAI-compatible (OpenAI / DeepSeek / Ollama) ──

class OpenAICompatProvider(LLMProvider):
    """Single class for OpenAI, DeepSeek, and Ollama — all use chat/completions API."""

    PROVIDER_CONFIG = {
        "openai":   {"default_model": OPENAI_DEFAULT_MODEL,   "key_env": "OPENAI_API_KEY",   "base_url": None,                       "vision": lambda m: "gpt-4o" in m or "gpt-4.1" in m or "vision" in m.lower()},
        "deepseek": {"default_model": "deepseek-v4-pro",       "key_env": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com",  "vision": lambda m: False},
        "ollama":   {"default_model": OLLAMA_DEFAULT_MODEL,    "key_env": "",                  "base_url": OLLAMA_BASE_URL + "/v1",    "vision": lambda m: "llava" in m.lower()},
    }

    def __init__(self, provider_id: str = None, model: str = None):
        self.provider_id = provider_id or get_default_provider()
        cfg = self.PROVIDER_CONFIG.get(self.provider_id, self.PROVIDER_CONFIG["deepseek"])
        super().__init__(model or cfg["default_model"])

        from openai import AsyncOpenAI
        kwargs = {}
        if cfg["base_url"]:
            kwargs["base_url"] = cfg["base_url"]
        if cfg["key_env"]:
            kwargs["api_key"] = os.environ.get(cfg["key_env"], "")
        else:
            kwargs["api_key"] = "local"  # Ollama doesn't need a real key
        self.client = AsyncOpenAI(**kwargs)
        self._vision_check = cfg["vision"]

    def has_vision(self) -> bool:
        return self._vision_check(self.model)

    def count_tokens(self, text: str) -> int:
        try:
            import tiktoken
            return len(tiktoken.get_encoding("cl100k_base").encode(text))
        except Exception:
            return len(text) // 4

    @_retry()
    async def chat(self, system: str, messages: list[dict]) -> str:
        kwargs = dict(model=self.model, messages=self._build(system, messages), max_tokens=LLM_MAX_TOKENS)
        if self.provider_id == "deepseek" and "reasoner" in self.model:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        resp = await self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    async def chat_stream(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        try:
            kwargs = dict(model=self.model, messages=self._build(system, messages),
                          max_tokens=LLM_MAX_TOKENS, stream=True)
            if self.provider_id == "deepseek" and "reasoner" in self.model:
                kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            stream = await self.client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"{self.provider_id} stream error: {e}")
            yield f"\n\n[回答生成出错，请重试]"

    @_retry()
    async def chat_with_image(self, system: str, messages: list[dict], image_b64: str, mime: str) -> str:
        if not self.has_vision():
            raise NotImplementedError(f"{self.provider_id} does not support vision.")
        content = [{"type": "text", "text": m["content"]} for m in messages]
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}})
        msgs = [{"role": "system", "content": system}] if system else []
        msgs.append({"role": "user", "content": content})
        resp = await self.client.chat.completions.create(
            model=self.model, messages=msgs, max_tokens=LLM_MAX_TOKENS)
        return resp.choices[0].message.content

    def _build(self, system: str, messages: list[dict]) -> list[dict]:
        msgs = [{"role": "system", "content": system}] if system else []
        for m in messages:
            msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        return msgs or [{"role": "user", "content": "Hello"}]


# ── Factory ──

_PROVIDER_CLASSES = {
    "anthropic": AnthropicProvider,
    "deepseek": OpenAICompatProvider,
    "openai": OpenAICompatProvider,
    "ollama": OpenAICompatProvider,
}


def get_available_providers() -> list[dict]:
    """Return list of configured providers."""
    providers = []
    if _key("anthropic"):
        providers.append({"id": "anthropic", "name": "Anthropic",
                          "models": ["claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5"],
                          "vision": True})
    if _key("deepseek"):
        providers.append({"id": "deepseek", "name": "DeepSeek",
                          "models": ["deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"],
                          "vision": False})
    if _key("openai"):
        providers.append({"id": "openai", "name": "OpenAI",
                          "models": ["gpt-4.1", "gpt-4.1-mini", "o3"], "vision": True})
    providers.append({"id": "ollama", "name": "Ollama (local)",
                      "models": ["llama3.2", "llava"], "vision": False, "local": True})
    return providers


def create_llm(provider_id: str = None, model: str = None) -> LLMProvider:
    """Create an LLM provider. Falls back to first available if not configured."""
    pid = provider_id or get_default_provider()

    # Validate: if no key and not ollama, find fallback
    if not _key(pid) and pid != "ollama":
        for alt in ["anthropic", "deepseek", "openai"]:
            if _key(alt):
                pid = alt
                break

    if not _key(pid) and pid != "ollama":
        raise ValueError("No API key configured. Add at least one in Settings.")

    cls = _PROVIDER_CLASSES.get(pid, OpenAICompatProvider)
    return cls(pid, model)
