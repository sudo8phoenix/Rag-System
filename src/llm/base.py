"""Shared LLM response models and domain errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class LLMError(RuntimeError):
    """Base error for LLM operations."""


class LLMConnectionError(LLMError):
    """Raised when an LLM endpoint cannot be reached."""


class LLMProviderError(LLMError):
    """Raised when the provider returns an unexpected error."""


@dataclass(frozen=True)
class LLMStreamToken:
    """A single streamed token/chunk from an LLM response."""

    token: str
    done: bool = False


@dataclass(frozen=True)
class LLMResponse:
    """Final non-streaming LLM response payload."""

    text: str
    model: str
    prompt: str
    done_reason: str | None = None
    raw: dict[str, Any] | None = None
