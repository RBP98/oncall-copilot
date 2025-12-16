from __future__ import annotations

import time
from typing import List, Optional

from pydantic import BaseModel, Field

from .types import (
    EMBED_MODEL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    ToolEnvelope,
    make_envelope,
    make_error_envelope,
    truncate_text,
)

_qdrant = None
_embedder = None


def _get_clients():
    global _qdrant, _embedder
    if _qdrant is None:
        from qdrant_client import QdrantClient  # lazy import
        _qdrant = QdrantClient(url=QDRANT_URL)
    if _embedder is None:
        from sentence_transformers import SentenceTransformer  # lazy import
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _qdrant, _embedder


class DocsSearchArgs(BaseModel):
    question: str
    top_k: int = Field(3, ge=1, le=20)


class DocChunk(BaseModel):
    source: Optional[str] = None
    text: str
    score: float


class DocsSearchResult(BaseModel):
    chunks: List[DocChunk]


def search(args: DocsSearchArgs) -> ToolEnvelope:
    started = time.time()
    try:
        qdrant, embedder = _get_clients()
        vec = embedder.encode(args.question).tolist()

        # match your current usage (query_points)
        res = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vec,
            limit=args.top_k,
            with_payload=True,
        )
        hits = res.points or []

        chunks: List[DocChunk] = []
        for h in hits:
            payload = h.payload or {}
            text = payload.get("text") or ""
            text, _ = truncate_text(text, max_chars=4000)
            chunks.append(
                DocChunk(
                    source=payload.get("source"),
                    text=text,
                    score=float(h.score),
                )
            )

        return make_envelope("docs.search", args, DocsSearchResult(chunks=chunks), started)

    except Exception as e:
        return make_error_envelope("docs.search", args, type(e).__name__, str(e), None, started)
