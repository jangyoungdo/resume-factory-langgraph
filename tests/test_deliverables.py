import json
from pathlib import Path

import pytest

from resume_factory.deliverables import render_bundle_file


def _bundle(body_length: int) -> dict[str, object]:
    headline = "[합성 소제목]"
    body = "가" * body_length
    return {
        "company": "Sample Steel",
        "job": "Maintenance Engineer",
        "revision": 2,
        "source_run_id": "synthetic-run",
        "answers": [
            {
                "question_id": "Q1",
                "prompt": "합성 질문",
                "character_limit": 600,
                "headline": headline,
                "body": body,
                "evidence_ids": ["SYNTH-PHM-01"],
            }
        ],
    }


def test_render_bundle_uses_canonical_count_and_never_overwrites(tmp_path: Path) -> None:
    input_path = tmp_path / "draft.json"
    output_path = tmp_path / "draft.md"
    payload = _bundle(575)
    input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    render_bundle_file(input_path, output_path)
    rendered = output_path.read_text(encoding="utf-8")
    assert "글자 수: 584/600" in rendered
    assert "[합성 소제목]\n" in rendered
    with pytest.raises(FileExistsError):
        render_bundle_file(input_path, output_path)


def test_render_bundle_rejects_underfilled_submission(tmp_path: Path) -> None:
    input_path = tmp_path / "draft.json"
    input_path.write_text(json.dumps(_bundle(100), ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="최소"):
        render_bundle_file(input_path, tmp_path / "draft.md")
