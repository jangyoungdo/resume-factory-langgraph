from pathlib import Path

from resume_factory.agents import DeterministicBackend
from resume_factory.graph import _outside_hard_gate, run_resume_graph
from resume_factory.schemas import ApplicationInput, CallKind, ExecutionMode

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


async def test_graph_persists_sqlite_checkpoints(tmp_path: Path) -> None:
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    checkpoint = tmp_path / "checkpoints.sqlite"
    await run_resume_graph(
        application,
        DeterministicBackend(),
        ExecutionMode.BALANCED,
        run_id="checkpoint-test",
        checkpoint_path=checkpoint,
    )
    assert checkpoint.exists()
    import sqlite3

    with sqlite3.connect(checkpoint) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", ("checkpoint-test",)
        ).fetchone()[0]
    assert count >= 7


def test_character_rewrite_uses_only_hard_gate() -> None:
    assert not _outside_hard_gate(969, 950, 1000)
    assert not _outside_hard_gate(983, 950, 1000)
    assert _outside_hard_gate(949, 950, 1000)
    assert _outside_hard_gate(1001, 950, 1000)


async def test_common_analysis_starts_in_parallel() -> None:
    class StartRecordingBackend(DeterministicBackend):
        def __init__(self) -> None:
            super().__init__()
            self.started_roles: list[str] = []

        async def propose(self, **kwargs):  # type: ignore[no-untyped-def]
            self.started_roles.append(str(kwargs["role"]))
            import asyncio

            await asyncio.sleep(0.01)
            return await super().propose(**kwargs)

    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    backend = StartRecordingBackend()
    await run_resume_graph(application, backend, ExecutionMode.BALANCED)
    assert set(backend.started_roles[:6]) == {
        "business_analyst",
        "job_demand_analyst",
        "literal_question_analyst",
        "recruiter_intent_analyst",
        "evidence_transfer_analyst",
        "positioning_strategist",
    }


async def test_clear_writing_candidates_skip_optional_critic() -> None:
    class ClearMarginBackend(DeterministicBackend):
        async def propose(self, **kwargs):  # type: ignore[no-untyped-def]
            proposal = await super().propose(**kwargs)
            if kwargs["role"] == "recruiter_value_writer":
                proposal.score.relevance = 1.0
                proposal.score.evidence_fidelity = 1.0
            return proposal

    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    backend = ClearMarginBackend()
    result = await run_resume_graph(application, backend, ExecutionMode.BALANCED)
    critics = [call for call in backend.calls if call.call_kind is CallKind.CRITIC]
    assert critics == []
    assert result.telemetry.base_call_budget == 13


async def test_multiple_failed_questions_use_one_character_repair_call() -> None:
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    application.questions.extend(
        [
            application.questions[0].model_copy(
                update={"question_id": "Q2", "text": "두 번째 문항"}
            ),
            application.questions[0].model_copy(
                update={"question_id": "Q3", "text": "세 번째 문항"}
            ),
        ]
    )
    backend = DeterministicBackend()
    result = await run_resume_graph(application, backend, ExecutionMode.BALANCED)
    rewrites = [call for call in backend.calls if call.call_kind is CallKind.CHARACTER_REWRITE]
    assert len(rewrites) == 1
    assert result.telemetry.base_call_budget == 19
    assert result.telemetry.hard_call_cap == 24
    assert len(backend.calls) <= 24
