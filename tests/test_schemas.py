from pathlib import Path

import pytest

from resume_factory.schemas import AgentScore, ApplicationInput

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
