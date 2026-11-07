"""LLM Abstraction Layer — unified interface for multiple LLM providers.

Uses LiteLLM to provide a single API across OpenAI, Anthropic, Google, and
local models. Handles token counting, cost tracking, and error recovery.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from atlas_rag.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM completion call."""

    content: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    finish_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    """Unified LLM client supporting multiple providers via LiteLLM.

    Supports:
      - Synchronous completion
      - Async completion
      - Streaming completion (async generator)
      - Token counting and cost estimation
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.default_model = self.settings.default_llm_model

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion from the LLM.

        Args:
            messages: Chat messages in OpenAI format.
            model: Model to use (defaults to configured default).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with content and usage metadata.
        """
        import litellm

        model = model or self.default_model
        provider = litellm.get_llm_provider(model=model)[1]
        api_key = {
            "openai": self.settings.openai_api_key,
            "anthropic": self.settings.anthropic_api_key,
        }.get(provider)
        if api_key:
            kwargs.setdefault("api_key", api_key)
        start = time.perf_counter()

        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            elapsed = (time.perf_counter() - start) * 1000
            usage = response.usage

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=model,
                tokens_in=usage.prompt_tokens if usage else 0,
                tokens_out=usage.completion_tokens if usage else 0,
                latency_ms=round(elapsed, 2),
                finish_reason=response.choices[0].finish_reason or "",
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("LLM completion failed (%s, %.0fms): %s", model, elapsed, e)
            raise

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion from the LLM, yielding text chunks.

        Args:
            messages: Chat messages in OpenAI format.
            model: Model to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Yields:
            Text chunks as they are generated.
        """
        import litellm

        model = model or self.default_model

        provider = litellm.get_llm_provider(model=model)[1]
        api_key = {
            "openai": self.settings.openai_api_key,
            "anthropic": self.settings.anthropic_api_key,
        }.get(provider)
        if api_key:
            kwargs.setdefault("api_key", api_key)

        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

        except Exception as e:
            logger.error("LLM streaming failed (%s): %s", model, e)
            raise
