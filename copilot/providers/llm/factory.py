from __future__ import annotations
import os
import pkgutil
import importlib
from typing import Optional
from .base import LLMProvider
from .registry import get_registered

def _autodiscover_providers():
    """
    Import all modules in this package so they can register via @register_provider.
    This keeps the system Open/Closed: adding a new provider = adding a new file only.
    """
    pkg_name = __package__  # "copilot.providers.llm"
    pkg = importlib.import_module(pkg_name)

    for m in pkgutil.iter_modules(pkg.__path__):
        mod_name = m.name
        # skip internal modules
        if mod_name in {"base", "registry", "factory"}:
            continue
        try:
            importlib.import_module(f"{pkg_name}.{mod_name}")
        except Exception:
            # Don't crash app because an optional provider dependency isn't installed.
            # If the user selects that provider, we will error clearly later.
            pass

def create_llm_provider() -> Optional[LLMProvider]:
    _autodiscover_providers()

    name = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not name:
        return None  # LLM disabled

    providers = get_registered()
    if name not in providers:
        available = ", ".join(sorted(providers.keys())) or "(none discovered)"
        raise RuntimeError(f"Unknown LLM_PROVIDER='{name}'. Available: {available}")

    return providers[name].from_env()
