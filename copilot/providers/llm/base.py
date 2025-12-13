from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict

class LLMProvider(ABC):
    """Stable port/interface for all LLM providers."""

    @classmethod
    @abstractmethod
    def from_env(cls) -> "LLMProvider":
        """Create provider instance using environment variables."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str, *, meta: Dict[str, Any] | None = None) -> str:
        """Return a single natural-language answer based on prompt (+ optional metadata)."""
        raise NotImplementedError
