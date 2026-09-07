from pathlib import Path

from resume_factory.agents import DeterministicBackend
from resume_factory.graph import run_resume_graph
from resume_factory.schemas import ApplicationInput, ExecutionMode


async def test_public_golden_case_has_zero_fact_failures() -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "sample_application.json"
    application = ApplicationInput.model_validate_json(fixture.read_text(encoding="utf-8"))
    result = await run_resume_graph(application, DeterministicBackend(), ExecutionMode.BALANCED)
    assert result.validation.metrics["unsupported_claim_count"] == 0
    assert result.validation.metrics["numeric_mismatch_count"] == 0
    assert result.validation.metrics["fact_collision_count"] == 0
