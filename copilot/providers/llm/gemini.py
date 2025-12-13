from __future__ import annotations
import os
from google import genai
from .base import LLMProvider
from .registry import register_provider

@register_provider("gemini")
class GeminiProvider(LLMProvider):
    def __init__(self, client: genai.Client, model: str):
        self.client = client
        self.model = model

    @classmethod
    def from_env(cls) -> "GeminiProvider":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        client = genai.Client(api_key=api_key)
        return cls(client=client, model=model)

    def generate(self, prompt: str, *, meta=None) -> str:
        resp = self.client.models.generate_content(model=self.model, contents=prompt)
        return (resp.text or "").strip()
