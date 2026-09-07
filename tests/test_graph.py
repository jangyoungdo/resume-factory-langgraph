from pathlib import Path

import resume_factory.graph as graph_module
from resume_factory.agents import DeterministicBackend
from resume_factory.graph import _outside_hard_gate, _writing_brief, run_resume_graph
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
    assert 1 <= result.telemetry.metadata["character_rewrite_round_count"] <= 3


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
            "experience_key": "SYNTH_HANDOVER",
            "title": "재현 가능한 인계",
            "best_for_questions": ["Q2"],
        }
    )
    application.evidence.append(second)

    result = await run_resume_graph(application, DeterministicBackend(), ExecutionMode.ECONOMY)

    assert result.answers[0].evidence_ids == ["SYNTH-PHM-01"]
    assert result.answers[1].evidence_ids == ["SYNTH-LEAD-02"]
    assert result.transfer_contracts[1].evidence_event_id == "SYNTH-LEAD-02"


async def test_transfer_contract_uses_application_context_without_fa_leakage() -> None:
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    result = await run_resume_graph(application, DeterministicBackend(), ExecutionMode.ECONOMY)

    transfer = result.transfer_contracts[0]
    assert application.company in transfer.first_action
    assert application.job in transfer.first_action
    assert "물류설비" not in transfer.first_action
    assert "배터리" not in result.question_contracts[0].likely_objections[0]


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
    assert set(backend.started_roles[:4]) == {
        "business_analyst",
        "job_demand_analyst",
        "literal_question_analyst",
        "recruiter_intent_analyst",
    }
    evidence_start = backend.started_roles.index("evidence_transfer_analyst")
    assert evidence_start > backend.started_roles.index("answer_architect")
    assert evidence_start > backend.started_roles.index("intelligence_lead")


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
    assert result.telemetry.base_call_budget == 14


async def test_multiple_failed_questions_use_one_successful_character_repair_round() -> None:
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
    application.evidence.extend(
        [
            application.evidence[0].model_copy(
                update={
                    "event_id": "SYNTH-Q2-02",
                    "experience_key": "SYNTH_Q2",
                    "best_for_questions": ["Q2"],
                }
            ),
            application.evidence[0].model_copy(
                update={
                    "event_id": "SYNTH-Q3-03",
                    "experience_key": "SYNTH_Q3",
                    "best_for_questions": ["Q3"],
                }
            ),
        ]
    )
    backend = DeterministicBackend()
    result = await run_resume_graph(application, backend, ExecutionMode.BALANCED)
    rewrites = [call for call in backend.calls if call.call_kind is CallKind.CHARACTER_REWRITE]
    assert len(rewrites) <= 3
    assert result.telemetry.base_call_budget == 22
    assert result.telemetry.metadata["character_rewrite_round_count"] <= 3
    assert result.telemetry.hard_call_cap == 30
    assert len(backend.calls) <= 30


async def test_character_repair_retries_up_to_three_rounds(monkeypatch) -> None:
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    original_rewrite = graph_module.rewrite_to_character_target
    attempts = 0

    def flaky_rewrite(answer, app, evidence, transfer):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return answer, True
        updated, _ = original_rewrite(answer, app, evidence, transfer)
        return original_rewrite(updated, app, evidence, transfer)

    monkeypatch.setattr(graph_module, "rewrite_to_character_target", flaky_rewrite)
    backend = DeterministicBackend()
    result = await run_resume_graph(application, backend, ExecutionMode.BALANCED)

    rewrites = [call for call in backend.calls if call.call_kind is CallKind.CHARACTER_REWRITE]
    assert len(rewrites) == 3
    assert result.telemetry.metadata["character_rewrite_round_count"] == 3
    assert result.status == "validated"


async def test_learning_question_receives_followup_evidence_and_editorial_rules() -> None:
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    primary = application.evidence[0]
    followup = primary.model_copy(
        update={
            "event_id": "SYNTH-FOLLOWUP-02",
            "experience_key": "SYNTH_FOLLOWUP",
            "title": "후행 적용",
        }
    )
    application.evidence.append(followup)
    application.questions[
        0
    ].text = "가장 많은 노력을 쏟은 실패 경험과 그 과정을 통해 무엇을 배웠는지 쓰십시오."
    application.questions[0].supporting_evidence_ids = [followup.event_id]
    backend = DeterministicBackend()
    result = await run_resume_graph(application, backend, ExecutionMode.ECONOMY)

    brief = _writing_brief(
        application,
        application.questions[0].model_dump(),
        result.question_contracts[0],
        result.transfer_contracts[0],
        primary,
        result.positioning_brief,
    )

    assert brief["learning_transfer_required"] is True
    assert brief["evidence_ids"] == [primary.event_id, followup.event_id]
    assert [item["event_id"] for item in brief["supporting_evidence"]] == [followup.event_id]
    joined_rules = " ".join(brief["rules"])
    assert "보험 문장은 본문 금지" in joined_rules
    assert "관찰 가능한 결과" in joined_rules
    assert "후행 프로젝트" in joined_rules
    assert any(call.agent_role == "writing_council_anonymous_critic" for call in backend.calls)
