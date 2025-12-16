from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any, Dict, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

# ---- Limits / safety knobs (override via env) ----
MAX_TEXT_CHARS = int(os.getenv("COPILOT_MAX_TEXT_CHARS", "20000"))
MAX_LOG_LINES = int(os.getenv("COPILOT_MAX_LOG_LINES", "200"))

KUBECTL_BIN = os.getenv("COPILOT_KUBECTL_BIN", "kubectl")

ALLOWED_NAMESPACES = set(
    x.strip() for x in os.getenv("COPILOT_ALLOWED_NAMESPACES", "oncall,monitoring").split(",") if x.strip()
)
ALLOWED_APP_LABELS = set(
    x.strip() for x in os.getenv("COPILOT_ALLOWED_APP_LABELS", "demo-app").split(",") if x.strip()
)

PROM_URL = os.getenv("COPILOT_PROM_URL", "http://localhost:9090")
QDRANT_URL = os.getenv("COPILOT_QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("COPILOT_QDRANT_COLLECTION", "runbooks")
EMBED_MODEL = os.getenv("COPILOT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class ToolError(BaseModel):
    type: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ToolEnvelope(BaseModel):
    tool: str
    args: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[ToolError] = None
    duration_ms: int = 0


def _dump_model(m: Any) -> Dict[str, Any]:
    if m is None:
        return {}
    if hasattr(m, "model_dump"):  # pydantic v2
        return m.model_dump()
    if hasattr(m, "dict"):  # pydantic v1
        return m.dict()
    if isinstance(m, dict):
        return m
    return json.loads(json.dumps(m, default=str))


def truncate_text(s: str, max_chars: int = MAX_TEXT_CHARS) -> Tuple[str, bool]:
    if not s:
        return "", False
    if len(s) <= max_chars:
        return s, False
    return s[:max_chars] + "\n…(truncated)…", True


def make_envelope(tool: str, args_model: Any, result_model: Any, started_at: float) -> ToolEnvelope:
    env = ToolEnvelope(
        tool=tool,
        args=_dump_model(args_model),
        result=_dump_model(result_model),
        duration_ms=int((time.time() - started_at) * 1000),
    )
    # shallow truncate big strings in result
    for k, v in list((env.result or {}).items()):
        if isinstance(v, str):
            env.result[k], _ = truncate_text(v)
    return env


def make_error_envelope(
    tool: str,
    args_model: Any,
    err_type: str,
    message: str,
    details: Optional[Dict[str, Any]],
    started_at: float,
) -> ToolEnvelope:
    return ToolEnvelope(
        tool=tool,
        args=_dump_model(args_model),
        error=ToolError(type=err_type, message=message, details=details),
        duration_ms=int((time.time() - started_at) * 1000),
    )


def run_kubectl(args: Sequence[str], timeout_s: int = 10) -> subprocess.CompletedProcess:
    # IMPORTANT: no shell=True, caller passes tokenized args
    return subprocess.run(
        [KUBECTL_BIN, *args],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
