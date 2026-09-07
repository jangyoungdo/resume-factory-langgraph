from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
from mcp.server.fastmcp import FastMCP

from resume_factory.security import safe_path

mcp = FastMCP("dual-brain-readonly")


def _root() -> Path:
    configured = os.getenv("RF_DUAL_BRAIN_ROOT")
    if not configured:
        raise RuntimeError("RF_DUAL_BRAIN_ROOT is not configured")
    return Path(configured).expanduser().resolve()


def _evidence_file() -> Path:
    return safe_path(_root(), _root() / ".resume_factory" / "evidence.jsonl")


def _load_records() -> list[dict[str, Any]]:
    path = _evidence_file()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@mcp.tool()
def search_evidence(
    query: str, capabilities: list[str] | None = None, top_k: int = 5
) -> dict[str, Any]:
    """Search the private, curated evidence index without exposing arbitrary files."""
    records = _load_records()
    semantic = _semantic_rank(query, records, top_k)
    if semantic is not None:
        return {"results": semantic, "retrieval": "bge-m3"}
    terms = {term.lower() for term in query.split() if len(term) > 1}
    required = {item.lower() for item in capabilities or []}
    ranked = []
    for record in records:
        haystack = json.dumps(record, ensure_ascii=False).lower()
        score = sum(term in haystack for term in terms)
        record_caps = {str(item).lower() for item in record.get("capabilities", [])}
        score += 2 * len(required & record_caps)
        if score:
            ranked.append((score, record))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return {
        "results": [record for _, record in ranked[: max(1, min(top_k, 20))]],
        "retrieval": "lexical-fallback",
    }


def _semantic_rank(
    query: str, records: list[dict[str, Any]], top_k: int
) -> list[dict[str, Any]] | None:
    embedding_path = _root() / ".resume_factory" / "evidence_embeddings.npy"
    model_path = _root() / ".resume_factory" / "embedding_model.txt"
    if not records or not embedding_path.exists() or not model_path.exists():
        return None
    embeddings = np.load(embedding_path, allow_pickle=False)
    if len(embeddings) != len(records):
        return None
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path.read_text(encoding="utf-8").strip())
    query_vector = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ query_vector
    order = np.argsort(scores)[::-1][: max(1, min(top_k, 20))]
    return [{**records[int(index)], "retrieval_score": float(scores[index])} for index in order]


@mcp.tool()
def get_evidence_event(event_id: str) -> dict[str, Any]:
    """Return one curated evidence event by immutable ID."""
    for record in _load_records():
        if record.get("event_id") == event_id:
            return record
    return {"error": "not_found", "event_id": event_id}


@mcp.tool()
def get_numeric_authority(authority_id: str) -> dict[str, Any]:
    """Return a verified numeric authority by ID."""
    for record in _load_records():
        for authority in record.get("numeric_authorities", []):
            if authority.get("authority_id") == authority_id:
                return {"event_id": record.get("event_id"), **authority}
    return {"error": "not_found", "authority_id": authority_id}


@mcp.tool()
def get_project_boundary(event_id: str) -> dict[str, Any]:
    """Return contribution boundaries and forbidden event combinations."""
    record = get_evidence_event(event_id)
    if "error" in record:
        return cast(dict[str, Any], record)
    return {
        "event_id": event_id,
        "boundaries": record.get("boundaries", []),
        "forbidden_combinations": record.get("forbidden_combinations", []),
    }


@mcp.tool()
def get_previous_feedback(company: str, question_type: str) -> dict[str, Any]:
    """Read feedback scoped to the same company and question type."""
    feedback_path = safe_path(_root(), _root() / ".resume_factory" / "feedback" / "feedback.jsonl")
    if not feedback_path.exists():
        return {"results": []}
    results: list[dict[str, Any]] = []
    for line in feedback_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("company") == company and item.get("question_type") == question_type:
            results.append(item)
    return {"results": results}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
