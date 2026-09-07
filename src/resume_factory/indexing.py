from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .schemas import EvidencePacket
from .security import safe_path


def build_curated_index(
    source: Path, private_root: Path, *, semantic: bool = True
) -> tuple[Path, int]:
    source = safe_path(private_root, source)
    output_dir = private_root / ".resume_factory"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "evidence.jsonl"
    records: list[EvidencePacket] = []
    for path in sorted(source.glob("*.json")):
        records.append(EvidencePacket.model_validate_json(path.read_text(encoding="utf-8")))
    output.write_text(
        "".join(json.dumps(record.model_dump(), ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    if semantic and records:
        from sentence_transformers import SentenceTransformer

        model_name = os.getenv("RF_EMBEDDING_MODEL", "BAAI/bge-m3")
        model = SentenceTransformer(model_name)
        documents = [_search_document(record) for record in records]
        embeddings = model.encode(documents, normalize_embeddings=True)
        np.save(output_dir / "evidence_embeddings.npy", embeddings, allow_pickle=False)
        (output_dir / "embedding_model.txt").write_text(model_name, encoding="utf-8")
    return output, len(records)


def _search_document(record: EvidencePacket) -> str:
    return " ".join(
        [
            record.title,
            record.problem,
            record.judgment,
            *record.actions,
            *record.results,
            *record.capabilities,
        ]
    )
