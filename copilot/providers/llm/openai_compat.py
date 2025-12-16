from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from .base import LLMProvider
from .registry import register_provider


def _env(*names: str) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v and v.strip():
            return v.strip()
    return None


def _post_json(url: str, api_key: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)


def _extract_text(resp: Dict[str, Any]) -> str:
    # OpenAI Chat Completions shape
    choices = resp.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
        # Some providers may return list-of-blocks
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return str(first["text"]).strip()

    # Fallbacks
    if "output_text" in resp:
        return str(resp["output_text"]).strip()
    if "text" in resp:
        return str(resp["text"]).strip()

    return json.dumps(resp)[:4000]


class _OpenAICompatChatProvider(LLMProvider):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_s: float = 30.0,
        temperature: float = 0.2,
        max_tokens: int = 700,
        retries: int = 2,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries

    def generate(self, prompt: str, *, meta: Dict[str, Any] | None = None) -> str:
        url = f"{self.base_url}/chat/completions"

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        delay = 0.8
        last_err: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                resp = _post_json(url, self.api_key, payload, self.timeout_s)
                return _extract_text(resp)

            except urllib.error.HTTPError as e:
                # Try to read provider error body (useful for debugging)
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    pass

                # Retry on overload/rate-limit
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(delay + random.random() * 0.25)
                    delay *= 2
                    last_err = RuntimeError(f"HTTP {e.code}: {body[:300]}")
                    continue

                raise RuntimeError(f"HTTP {e.code}: {body[:800]}") from e

            except Exception as e:
                # Network/timeout/etc.
                if attempt < self.retries:
                    time.sleep(delay + random.random() * 0.25)
                    delay *= 2
                    last_err = e
                    continue
                raise

        raise RuntimeError(f"LLM failed after retries: {last_err}")  # should never hit


@register_provider("openai_compat")
class OpenAICompatProvider(_OpenAICompatChatProvider):
    """
    Generic OpenAI-compatible provider.
    Configure with:
      OPENAI_COMPAT_API_KEY
      OPENAI_COMPAT_BASE_URL
      OPENAI_COMPAT_MODEL
    """
    @classmethod
    def from_env(cls) -> "OpenAICompatProvider":
        api_key = _env("OPENAI_COMPAT_API_KEY")
        base_url = _env("OPENAI_COMPAT_BASE_URL")
        model = _env("OPENAI_COMPAT_MODEL")

        if not api_key or not base_url or not model:
            raise RuntimeError(
                "openai_compat requires OPENAI_COMPAT_API_KEY, OPENAI_COMPAT_BASE_URL, OPENAI_COMPAT_MODEL"
            )

        timeout_s = float(_env("OPENAI_COMPAT_TIMEOUT_S") or "30")
        retries = int(_env("OPENAI_COMPAT_RETRIES") or "2")
        return cls(api_key=api_key, base_url=base_url, model=model, timeout_s=timeout_s, retries=retries)


@register_provider("groq")
class GroqProvider(_OpenAICompatChatProvider):
    @classmethod
    def from_env(cls) -> "GroqProvider":
        api_key = _env("GROQ_API_KEY", "OPENAI_COMPAT_API_KEY")
        model = _env("GROQ_MODEL") or "llama-3.1-8b-instant"
        base_url = _env("GROQ_BASE_URL") or "https://api.groq.com/openai/v1"

        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")

        return cls(api_key=api_key, base_url=base_url, model=model, retries=2)


@register_provider("cerebras")
class CerebrasProvider(_OpenAICompatChatProvider):
    @classmethod
    def from_env(cls) -> "CerebrasProvider":
        api_key = _env("CEREBRAS_API_KEY", "OPENAI_COMPAT_API_KEY")
        model = _env("CEREBRAS_MODEL") or "llama-3.3-70b"
        base_url = _env("CEREBRAS_BASE_URL") or "https://api.cerebras.ai/v1"

        if not api_key:
            raise RuntimeError("CEREBRAS_API_KEY is not set")

        return cls(api_key=api_key, base_url=base_url, model=model, retries=2)


@register_provider("cloudflare")
class CloudflareProvider(_OpenAICompatChatProvider):
    @classmethod
    def from_env(cls) -> "CloudflareProvider":
        api_key = _env("CLOUDFLARE_API_TOKEN", "OPENAI_COMPAT_API_KEY")
        acct = _env("CLOUDFLARE_ACCOUNT_ID")
        model = _env("CLOUDFLARE_MODEL") or "@cf/meta/llama-3.1-8b-instruct"

        if not api_key or not acct:
            raise RuntimeError("cloudflare requires CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID")

        base_url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/v1"
        return cls(api_key=api_key, base_url=base_url, model=model, retries=2)
