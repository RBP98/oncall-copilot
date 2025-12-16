from __future__ import annotations

import time
from typing import Any, Dict

import requests
from pydantic import BaseModel, Field

from .types import PROM_URL, ToolEnvelope, make_envelope, make_error_envelope, truncate_text


class PromQueryArgs(BaseModel):
    promql: str = Field(..., description="PromQL query")


class PromQueryResult(BaseModel):
    promql: str
    result: Any


def query(args: PromQueryArgs, timeout_s: int = 3) -> ToolEnvelope:
    started = time.time()
    try:
        r = requests.get(f"{PROM_URL}/api/v1/query", params={"query": args.promql}, timeout=timeout_s)
        if r.status_code != 200:
            body, _ = truncate_text(r.text)
            return make_error_envelope(
                "prom.query",
                args,
                "PromHttpError",
                f"HTTP {r.status_code}",
                {"body": body},
                started,
            )

        payload = r.json()
        data = (payload.get("data") or {}).get("result")
        return make_envelope("prom.query", args, PromQueryResult(promql=args.promql, result=data), started)

    except Exception as e:
        return make_error_envelope("prom.query", args, type(e).__name__, str(e), {"query": args.promql}, started)
