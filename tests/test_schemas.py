from pathlib import Path

import pytest

from resume_factory.schemas import AgentScore, ApplicationInput, DraftAnswer

FIXTURE = Path(__file__).parent / "fixtures" / "sample_application.json"


def test_application_fixture_is_valid_and_synthetic() -> None:
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    assert application.company == "Sample Steel"
    assert application.evidence[0].event_id.startswith("SYNTH-")


def test_weighted_score_prioritizes_evidence() -> None:
    score = AgentScore(
        relevance=5,
        evidence_fidelity=5,
        company_transfer=4,
        differentiation=4,
        sentence_efficiency=3,
        defensibility=4,
    )
    assert score.weighted == 4.4


def test_application_requires_evidence() -> None:
    payload = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8")).model_dump()
    payload["evidence"] = []
    with pytest.raises(ValueError, match="evidence"):
        ApplicationInput.model_validate(payload)


def test_application_rejects_unknown_supporting_evidence() -> None:
    payload = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8")).model_dump()
    payload["questions"][0]["supporting_evidence_ids"] = ["SYNTH-MISSING"]
    with pytest.raises(ValueError, match="unknown supporting evidence IDs"):
        ApplicationInput.model_validate(payload)


def test_submission_text_counts_headline_and_one_newline() -> None:
    answer = DraftAnswer(
        question_id="Q1",
        headline=" [제목]\r\n ",
        body=" 본문\r\n내용 ",
        sentence_plans=[],
        evidence_ids=[],
        character_limit=20,
    )
    assert answer.submission_text == "[제목]\n본문\n내용"
    assert answer.character_count == len("[제목]\n본문\n내용")
    assert answer.utilization_ratio == round(answer.character_count / 20, 4)
