from pathlib import Path

from resume_factory.agents import DeterministicBackend
from resume_factory.graph import run_resume_graph
from resume_factory.schemas import ApplicationInput, ExecutionMode

FIXTURE = Path(__file__).parent / "fixtures" / "sample_application.json"


async def test_full_graph_runs_offline_and_is_grounded() -> None:
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    backend = DeterministicBackend()
    result = await run_resume_graph(application, backend, ExecutionMode.BALANCED)
    assert result.status == "validated"
    assert result.answers[0].evidence_ids == ["SYNTH-PHM-01"]
    assert result.validation.metrics["sentence_role_coverage"] == 1
    assert len(result.team_decisions) == 4
    assert any(call.agent_role.endswith("_anonymous_critic") for call in backend.calls)
    assert any(call.agent_role == "evidence_brand_lead" for call in backend.calls)
    assert 582 <= result.answers[0].character_count <= 588
    assert result.telemetry.metadata["character_rewrite_attempts"] == ["Q1"]


async def test_economy_mode_uses_fewer_agents() -> None:
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    economy_backend = DeterministicBackend()
    balanced_backend = DeterministicBackend()
    await run_resume_graph(application, economy_backend, ExecutionMode.ECONOMY)
    await run_resume_graph(application, balanced_backend, ExecutionMode.BALANCED)
    assert len(economy_backend.calls) < len(balanced_backend.calls)


async def test_questions_can_select_distinct_evidence_packets() -> None:
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    application.questions.append(
        application.questions[0].model_copy(
            update={"question_id": "Q2", "text": "최근 주도 경험을 작성하십시오."}
        )
    )
    second = application.evidence[0].model_copy(
        update={
            "event_id": "SYNTH-LEAD-02",
            "title": "재현 가능한 인계",
            "best_for_questions": ["Q2"],
        }
    )
    application.evidence.append(second)

    result = await run_resume_graph(application, DeterministicBackend(), ExecutionMode.ECONOMY)

    assert result.answers[0].evidence_ids == ["SYNTH-PHM-01"]
    assert result.answers[1].evidence_ids == ["SYNTH-LEAD-02"]
    assert result.transfer_contracts[1].evidence_event_id == "SYNTH-LEAD-02"
