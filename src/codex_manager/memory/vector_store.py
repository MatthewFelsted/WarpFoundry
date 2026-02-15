"""Per-repository vector memory with optional ChromaDB persistence.

This module is intentionally dependency-light:
- If ``chromadb`` is installed, similarity search uses a persistent Chroma
  collection under ``.codex_manager/memory/vector_db``.
- If ``chromadb`` is unavailable, writes still succeed to JSONL caches and
  reads use a lexical fallback scorer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_EMBED_DIM = 256
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")
_MAX_EVENT_SCAN = 800


@dataclass(frozen=True)
class MemoryHit:
    """One vector-memory retrieval result."""

    memory_id: str
    document: str
    score: float
    metadata: dict[str, Any]


def _tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text or "")]


def _embed_hashed(text: str, dim: int = _EMBED_DIM) -> list[float]:
    """Create a deterministic dense vector from free text.

    This avoids heavyweight embedding model dependencies while still enabling
    cosine-style nearest-neighbor retrieval in Chroma.
    """
    vec = [0.0] * dim
    for token in _tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, byteorder="big", signed=False)
        idx = h % dim
        sign = -1.0 if (h & 1) else 1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        return vec
    return [v / norm for v in vec]


def _clamp_top_k(value: int, *, default: int = 8) -> int:
    try:
        n = int(value)
    except Exception:
        n = default
    return min(30, max(1, n))


def _safe_collection_name(raw: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", (raw or "").strip()).strip("-")
    if not cleaned:
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", fallback).strip("-")
    if not cleaned:
        cleaned = "repo-memory"
    return cleaned[:63]


def _coerce_metadata(payload: dict[str, Any]) -> dict[str, str | int | float | bool]:
    out: dict[str, str | int | float | bool] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[str(key)] = value
            continue
        if isinstance(value, (list, tuple, set)):
            out[str(key)] = ", ".join(str(item) for item in value)
            continue
        out[str(key)] = str(value)
    return out


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


class ProjectVectorMemory:
    """Per-repository memory store for notes, decisions, and research."""

    def __init__(
        self,
        repo_path: str | Path,
        *,
        enabled: bool = False,
        backend: str = "chroma",
        collection_name: str = "",
        default_top_k: int = 8,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.enabled = bool(enabled)
        self.backend = (backend or "chroma").strip().lower()
        self.default_top_k = _clamp_top_k(default_top_k)
        self.memory_dir = self.repo_path / ".codex_manager" / "memory"
        self.vector_db_dir = self.memory_dir / "vector_db"
        self.events_path = self.memory_dir / "vector_events.jsonl"
        self.research_cache_path = self.memory_dir / "deep_research_cache.jsonl"
        self.collection_name = _safe_collection_name(
            collection_name,
            fallback=self.repo_path.name or "repo-memory",
        )

        self.available = False
        self.reason = "disabled"
        self._collection: Any = None

        if self.enabled:
            self._initialize_backend()

    # ------------------------------------------------------------------
    # Backend setup
    # ------------------------------------------------------------------

    def _initialize_backend(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        if self.backend != "chroma":
            self.reason = f"unsupported backend: {self.backend}"
            return

        try:
            import chromadb
        except Exception as exc:
            self.reason = f"chromadb import failed: {exc}"
            return

        try:
            self.vector_db_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.vector_db_dir))
            self._collection = client.get_or_create_collection(name=self.collection_name)
            self.available = True
            self.reason = "ok"
        except Exception as exc:
            self.available = False
            self.reason = f"chroma init failed: {exc}"

    # ------------------------------------------------------------------
    # Basic note ingestion / retrieval
    # ------------------------------------------------------------------

    def add_note(
        self,
        text: str,
        *,
        category: str = "note",
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist one memory note and return its id.

        Notes are always journaled to JSONL. Vector indexing is best-effort.
        """
        content = str(text or "").strip()
        if not self.enabled or not content:
            return ""

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        memory_id = f"mem-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        now = _utc_now().isoformat()
        merged_meta = _coerce_metadata(
            {
                "memory_id": memory_id,
                "category": category or "note",
                "source": source or "",
                "created_at": now,
                **(metadata or {}),
            }
        )

        journal_payload = {
            "id": memory_id,
            "text": content,
            "metadata": merged_meta,
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(journal_payload, ensure_ascii=False) + "\n")

        if self.available and self._collection is not None:
            try:
                self._collection.add(
                    ids=[memory_id],
                    documents=[content],
                    embeddings=[_embed_hashed(content)],
                    metadatas=[merged_meta],
                )
            except Exception as exc:
                logger.warning("Vector memory add failed (%s): %s", memory_id, exc)

        return memory_id

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        categories: list[str] | None = None,
        source_prefix: str = "",
        metadata_equals: dict[str, Any] | None = None,
    ) -> list[MemoryHit]:
        """Return top-k memory hits for a query with optional metadata filters."""
        if not self.enabled:
            return []
        q = str(query or "").strip()
        if not q:
            return []
        limit = _clamp_top_k(top_k or self.default_top_k)

        if self.available and self._collection is not None:
            try:
                out = self._collection.query(
                    query_embeddings=[_embed_hashed(q)],
                    n_results=limit,
                    include=["documents", "metadatas", "distances"],
                )
                docs = (out.get("documents") or [[]])[0]
                metas = (out.get("metadatas") or [[]])[0]
                dists = (out.get("distances") or [[]])[0]
                results: list[MemoryHit] = []
                for idx, doc in enumerate(docs):
                    meta = metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
                    dist = float(dists[idx]) if idx < len(dists) else 1.0
                    score = max(0.0, 1.0 - dist)
                    results.append(
                        MemoryHit(
                            memory_id=str(meta.get("memory_id") or meta.get("id") or ""),
                            document=str(doc or ""),
                            score=score,
                            metadata=dict(meta),
                        )
                    )
                return self._filter_hits(
                    results,
                    categories=categories,
                    source_prefix=source_prefix,
                    metadata_equals=metadata_equals,
                    top_k=limit,
                )
            except Exception as exc:
                logger.warning("Vector memory query failed: %s", exc)

        fallback = self._search_fallback_lexical(q, top_k=max(limit * 4, 12))
        return self._filter_hits(
            fallback,
            categories=categories,
            source_prefix=source_prefix,
            metadata_equals=metadata_equals,
            top_k=limit,
        )

    def _search_fallback_lexical(self, query: str, *, top_k: int) -> list[MemoryHit]:
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for payload in self._iter_recent_event_payloads(limit=_MAX_EVENT_SCAN):
            text = str(payload.get("text") or "")
            t_tokens = set(_tokenize(text))
            if not t_tokens:
                continue
            overlap = len(q_tokens & t_tokens)
            union = len(q_tokens | t_tokens)
            score = overlap / union if union else 0.0
            if score <= 0.0:
                continue
            scored.append((score, payload))

        scored.sort(key=lambda item: item[0], reverse=True)
        hits: list[MemoryHit] = []
        for score, payload in scored[:top_k]:
            meta = payload.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
            hits.append(
                MemoryHit(
                    memory_id=str(payload.get("id") or ""),
                    document=str(payload.get("text") or ""),
                    score=float(score),
                    metadata=dict(meta),
                )
            )
        return hits

    @staticmethod
    def _filter_hits(
        hits: list[MemoryHit],
        *,
        categories: list[str] | None,
        source_prefix: str,
        metadata_equals: dict[str, Any] | None,
        top_k: int,
    ) -> list[MemoryHit]:
        category_set = {str(item).strip().lower() for item in (categories or []) if str(item).strip()}
        source_prefix_clean = str(source_prefix or "").strip().lower()
        expected = {
            str(key).strip(): str(value).strip()
            for key, value in (metadata_equals or {}).items()
            if str(key).strip()
        }
        filtered: list[MemoryHit] = []
        for hit in hits:
            meta = hit.metadata or {}
            category = str(meta.get("category") or "").strip().lower()
            if category_set and category not in category_set:
                continue
            source = str(meta.get("source") or "").strip().lower()
            if source_prefix_clean and not source.startswith(source_prefix_clean):
                continue
            if expected:
                mismatch = False
                for key, expected_value in expected.items():
                    if str(meta.get(key, "")).strip() != expected_value:
                        mismatch = True
                        break
                if mismatch:
                    continue
            filtered.append(hit)
            if len(filtered) >= top_k:
                break
        return filtered

    def _iter_recent_event_payloads(self, *, limit: int) -> list[dict[str, Any]]:
        if not self.events_path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        try:
            for line in self.events_path.read_text(encoding="utf-8").splitlines():
                raw = line.strip()
                if not raw:
                    continue
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    rows.append(parsed)
        except Exception:
            return []
        if len(rows) > limit:
            rows = rows[-limit:]
        return rows

    # ------------------------------------------------------------------
    # Deep-research cache
    # ------------------------------------------------------------------

    def record_deep_research(
        self,
        *,
        topic: str,
        summary: str,
        providers: str = "both",
        source: str = "pipeline:deep_research",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist a deep-research artifact and index it for retrieval."""
        clean_topic = str(topic or "").strip()
        clean_summary = str(summary or "").strip()
        if not self.enabled or not clean_topic or not clean_summary:
            return ""

        now = _utc_now().isoformat()
        payload = {
            "id": f"research-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}",
            "created_at": now,
            "topic": clean_topic,
            "providers": providers,
            "summary": clean_summary,
            "source": source,
            "metadata": metadata or {},
        }

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        with self.research_cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        self.add_note(
            f"Deep research topic: {clean_topic}\n\nSummary:\n{clean_summary}",
            category="deep_research",
            source=source,
            metadata={
                "memory_kind": "deep_research",
                "topic": clean_topic,
                "providers": providers,
                **(metadata or {}),
            },
        )
        return str(payload["id"])

    def lookup_recent_deep_research(
        self,
        topic: str,
        *,
        max_age_hours: int = 168,
        min_similarity: float = 0.72,
    ) -> dict[str, Any] | None:
        """Return a recent matching deep-research payload when available."""
        clean_topic = str(topic or "").strip()
        if not clean_topic or not self.research_cache_path.is_file():
            return None

        now = _utc_now()
        max_age = max(1, int(max_age_hours))
        cutoff = now - timedelta(hours=max_age)
        topic_tokens = set(_tokenize(clean_topic))
        if not topic_tokens:
            return None

        best_score = 0.0
        best_payload: dict[str, Any] | None = None

        lines = self.research_cache_path.read_text(encoding="utf-8").splitlines()
        for raw in reversed(lines):
            row = raw.strip()
            if not row:
                continue
            try:
                payload = json.loads(row)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            created_at = _parse_iso(str(payload.get("created_at") or ""))
            if created_at is None or created_at < cutoff:
                continue

            entry_topic = str(payload.get("topic") or "").strip()
            tokens = set(_tokenize(entry_topic))
            if not tokens:
                continue
            overlap = len(topic_tokens & tokens)
            union = len(topic_tokens | tokens)
            score = overlap / union if union else 0.0
            if score > best_score:
                best_score = score
                best_payload = payload
            if best_score >= 0.98:
                break

        if best_payload is None or best_score < float(min_similarity):
            return None
        return best_payload
