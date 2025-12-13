from __future__ import annotations
from typing import Dict, Type
from .base import LLMProvider

_REGISTRY: Dict[str, Type[LLMProvider]] = {}

def register_provider(name: str):
    """Decorator used by provider modules to register themselves."""
    name = name.strip().lower()

    def _decorator(cls: Type[LLMProvider]):
        if name in _REGISTRY:
            raise ValueError(f"LLM provider '{name}' already registered by {_REGISTRY[name]}")
        _REGISTRY[name] = cls
        return cls

    return _decorator

def get_registered() -> Dict[str, Type[LLMProvider]]:
    return dict(_REGISTRY)
