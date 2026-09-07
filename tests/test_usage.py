import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from resume_factory.agents import DeterministicBackend, OpenAIBackend
from resume_factory.character_budget import bounds_for
from resume_factory.config import Settings
from resume_factory.graph import run_resume_graph
from resume_factory.pricing import PriceCatalog
from resume_factory.schemas import (
    AgentProposal,
    AgentScore,
    ApplicationInput,
    CallKind,
    ExecutionMode,
    FeedbackDecision,
    ModelCallRecord,
    ModelTier,
    UsageStatus,
)
from resume_factory.storage import RunStore
from resume_factory.usage import aggregate_calls, build_feedback

FIXTURE = Path(__file__).parent / "fixtures" / "sample_application.json"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        model_luna="gpt-5.6-luna",
        model_terra="gpt-5.6-terra",
        model_sol="gpt-5.6-sol",
        execution_mode=ExecutionMode.BALANCED,
        dual_brain_root=None,
        notion_snapshot_dir=None,
        notion_token=None,
        openai_api_key="synthetic-key",
        enable_mlflow=False,
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        local_dir=tmp_path / ".local",
    )


def _proposal() -> AgentProposal:
    return AgentProposal(
        agent_role="fact_architect",
        proposal_id="proposal-1",
        recommendation="합성 근거만 사용합니다.",
        score=AgentScore(
            relevance=4,
            evidence_fidelity=5,
            company_transfer=4,
            differentiation=4,
            sentence_efficiency=4,
            defensibility=5,
        ),
        confidence=0.9,
    )


def _call(**updates: object) -> ModelCallRecord:
    payload: dict[str, object] = {
        "run_id": "run-1",
        "call_id": "call-1",
        "sequence": 1,
        "question_id": "Q1",
        "team": "soara",
        "agent_role": "fact_architect",
        "call_kind": CallKind.SPECIALIST,
        "tier": ModelTier.TERRA,
        "model": "test-model",
        "input_tokens": 100,
        "cached_input_tokens": 20,
        "output_tokens": 30,
        "reasoning_tokens": 10,
        "total_tokens": 130,
        "estimated_cost_usd": 0.001,
        "latency_ms": 50,
        "retry_count": None,
        "usage_status": UsageStatus.REPORTED,
        "started_at": datetime.now(UTC),
    }
    payload.update(updates)
    return ModelCallRecord.model_validate(payload)


def test_price_catalog_separates_cached_tokens_and_marks_unknown_model() -> None:
    catalog = PriceCatalog(
        {
            "catalog_version": "test-v1",
            "models": {
                "known": {
                    "input_per_million": 1,
                    "cached_input_per_million": 0.1,
                    "output_per_million": 2,
                }
            },
        }
    )
    estimate = catalog.estimate(
        "known", input_tokens=1000, cached_input_tokens=500, output_tokens=1000
    )
    assert estimate.cost_usd == 0.00255
    assert (
        catalog.estimate(
            "unknown", input_tokens=100, cached_input_tokens=0, output_tokens=10
        ).cost_usd
        is None
    )


def test_character_bounds_cover_600_and_500_character_questions() -> None:
    six_hundred = bounds_for(600)
    five_hundred = bounds_for(500)
    assert (six_hundred.hard_min, six_hundred.target_min, six_hundred.target_max) == (
        570,
        582,
        588,
    )
    assert (five_hundred.hard_min, five_hundred.target_min, five_hundred.target_max) == (
        475,
        485,
        490,
    )


def test_usage_aggregation_keeps_shared_calls_and_missing_price_visible() -> None:
    calls = [
        _call(),
        _call(
            call_id="call-2",
            sequence=2,
            question_id=None,
            team="intelligence",
            estimated_cost_usd=None,
            usage_status=UsageStatus.MISSING,
        ),
    ]
    rows = {row["group"]: row for row in aggregate_calls(calls, "question")}
    assert rows["Q1"]["total_tokens"] == 130
    assert rows["shared"]["unknown_price_calls"] == 1
    assert rows["shared"]["cost_complete"] is False
    assert rows["shared"]["missing_usage_calls"] == 1


async def test_openai_backend_normalizes_reported_usage(monkeypatch, tmp_path: Path) -> None:
    class FakeModel:
        def with_structured_output(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return self

        async def ainvoke(self, prompt: str):  # type: ignore[no-untyped-def]
            assert "evidence_ids" in prompt
            raw = SimpleNamespace(
                usage_metadata={
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "total_tokens": 1200,
                    "input_token_details": {"cache_read": 400, "cache_creation": 50},
                    "output_token_details": {"reasoning": 80},
                }
            )
            return {"parsed": _proposal(), "raw": raw}

    monkeypatch.setattr("resume_factory.agents.ChatOpenAI", lambda **kwargs: FakeModel())
    backend = OpenAIBackend(_settings(tmp_path))
    backend.begin_run("reported-run")
    await backend.propose(
        role="fact_architect",
        team="soara",
        brief={"evidence_ids": ["SYNTH-PHM-01"]},
        tier=ModelTier.TERRA,
        question_id="Q1",
        call_kind=CallKind.SPECIALIST,
    )
    call = backend.calls[0]
    assert call.usage_status is UsageStatus.REPORTED
    assert call.cached_input_tokens == 400
    assert call.cache_creation_input_tokens == 50
    assert call.reasoning_tokens == 80
    assert call.total_tokens == 1200
    assert call.estimated_cost_usd == 0.0044


async def test_openai_backend_records_sanitized_failure(monkeypatch, tmp_path: Path) -> None:
    class FailingModel:
        def with_structured_output(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return self

        async def ainvoke(self, prompt: str):  # type: ignore[no-untyped-def]
            raise RuntimeError("synthetic-secret-must-not-be-stored")

    monkeypatch.setattr("resume_factory.agents.ChatOpenAI", lambda **kwargs: FailingModel())
    backend = OpenAIBackend(_settings(tmp_path))
    backend.begin_run("failed-run")
    with pytest.raises(RuntimeError):
        await backend.propose(
            role="critic",
            team="fact_qa",
            brief={"evidence_ids": ["SYNTH-PHM-01"]},
            tier=ModelTier.LUNA,
            call_kind=CallKind.CRITIC,
        )
    call = backend.calls[0]
    assert not call.success
    assert call.error_code == "RuntimeError"
    assert "synthetic-secret" not in call.model_dump_json()


async def test_graph_records_unique_offline_call_lineage() -> None:
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    result = await run_resume_graph(application, DeterministicBackend(), ExecutionMode.BALANCED)
    calls = result.telemetry.model_calls
    assert len({call.call_id for call in calls}) == len(calls)
    assert [call.sequence for call in calls] == list(range(1, len(calls) + 1))
    assert all(call.usage_status is UsageStatus.OFFLINE for call in calls)
    assert any(call.question_id == "Q1" and call.team == "writing_council" for call in calls)
    assert any(call.question_id is None and call.team == "intelligence" for call in calls)
    assert any(call.call_kind is CallKind.CHARACTER_REWRITE for call in calls)


async def test_storage_migrates_and_persists_usage_rows(tmp_path: Path) -> None:
    local_dir = tmp_path / ".local"
    local_dir.mkdir()
    database = local_dir / "resume_factory.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY, status TEXT, company TEXT, job TEXT,
                total_cost_usd REAL, result_path TEXT, created_at TEXT
            )
            """
        )
    store = RunStore(local_dir)
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    result = await run_resume_graph(application, DeterministicBackend(), ExecutionMode.ECONOMY)
    store.save(result)
    with sqlite3.connect(database) as connection:
        usage_count = connection.execute("SELECT COUNT(*) FROM usage_runs").fetchone()
        call_count = connection.execute("SELECT COUNT(*) FROM model_calls").fetchone()
        migration = connection.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE version = 4"
        ).fetchone()
        usage_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(usage_runs)").fetchall()
        }
        call_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(model_calls)").fetchall()
        }
    assert usage_count == (1,)
    assert call_count == (len(result.telemetry.model_calls),)
    assert migration == (1,)
    assert {"wall_time_ms", "phase_spans_json", "network_status"} <= usage_columns
    assert {"queue_latency_ms", "first_event_latency_ms", "provider_execution_ms"} <= call_columns


async def test_feedback_stores_only_hash_path_and_edit_ratios(tmp_path: Path) -> None:
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    result = await run_resume_graph(application, DeterministicBackend(), ExecutionMode.ECONOMY)
    answer = result.answers[0]
    final_path = tmp_path / "final.json"
    final_payload = {
        "company": "Sample Steel",
        "job": "Maintenance Engineer",
        "revision": 2,
        "source_run_id": result.run_id,
        "answers": [
            {
                "question_id": "Q1",
                "prompt": "합성 질문",
                "character_limit": 600,
                "headline": answer.headline,
                "body": f"{answer.body} 수정",
                "evidence_ids": answer.evidence_ids,
            }
        ],
    }
    final_path.write_text(json.dumps(final_payload, ensure_ascii=False), encoding="utf-8")
    feedback = build_feedback(
        result,
        decision=FeedbackDecision.REVISED,
        rating=4,
        final_draft=final_path,
    )
    assert feedback.final_draft_hash
    assert feedback.overall_edit_ratio is not None
    assert feedback.per_question_edit_ratio["Q1"] > 0
    assert not hasattr(feedback, "final_draft_content")
