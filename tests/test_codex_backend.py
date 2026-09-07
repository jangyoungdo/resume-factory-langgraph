import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from resume_factory.codex_backend import (
    CodexCallBudgetExceeded,
    CodexExecBackend,
    CodexNetworkDegradedError,
)
from resume_factory.config import Settings
from resume_factory.schemas import (
    AgentProposal,
    BackendProvider,
    BillingMode,
    CostStatus,
    ExecutionMode,
    ModelTier,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        model_luna="gpt-5.6-luna",
        model_terra="gpt-5.6-terra",
        model_sol="gpt-5.6-sol",
        execution_mode=ExecutionMode.BALANCED,
        dual_brain_root=None,
        notion_snapshot_dir=None,
        notion_token=None,
        openai_api_key=None,
        enable_mlflow=False,
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        local_dir=tmp_path / ".local",
    )


def _proposal_payload() -> dict[str, object]:
    return {
        "agent_role": "analyst",
        "proposal_id": "p1",
        "recommendation": "검증된 근거만 사용",
        "score": {
            "relevance": 4,
            "evidence_fidelity": 5,
            "company_transfer": 4,
            "differentiation": 4,
            "sentence_efficiency": 4,
            "defensibility": 5,
        },
        "confidence": 0.9,
    }


def test_codex_jsonl_usage_and_subscription_metadata() -> None:
    raw = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "agent_message",
                        "text": json.dumps(_proposal_payload(), ensure_ascii=False),
                    },
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 40,
                        "output_tokens": 30,
                        "reasoning_output_tokens": 10,
                    },
                }
            ),
        ]
    )
    proposal, usage, provider_run_id = CodexExecBackend._parse_events(raw)
    assert proposal.proposal_id == "p1"
    assert usage["reasoning_tokens"] == 10
    assert provider_run_id == "thread-1"
    assert CodexExecBackend.provider is BackendProvider.CODEX_CLI
    assert CodexExecBackend.billing_mode is BillingMode.CHATGPT_SUBSCRIPTION
    assert CodexExecBackend.cost_status is CostStatus.NOT_APPLICABLE


async def test_codex_call_budget_is_hard_capped(tmp_path: Path) -> None:
    backend = CodexExecBackend(_settings(tmp_path), hard_call_cap=1, executable="/usr/bin/true")
    await backend._reserve_call()
    with pytest.raises(CodexCallBudgetExceeded):
        await backend._reserve_call()


async def test_codex_streaming_records_queue_and_first_event(tmp_path: Path) -> None:
    executable = tmp_path / "fake-codex"
    payload = json.dumps(_proposal_payload(), ensure_ascii=False)
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "sys.stdin.read()\n"
        "print(json.dumps({'type':'thread.started','thread_id':'fake-thread'}), flush=True)\n"
        f"print(json.dumps({{'type':'item.completed','item':{{'type':'agent_message',"
        f"'text':{payload!r}}}}}), flush=True)\n"
        "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10,"
        "'cached_input_tokens':2,'output_tokens':3,'reasoning_output_tokens':1}}), flush=True)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    backend = CodexExecBackend(_settings(tmp_path), executable=str(executable))
    backend.begin_run("stream-test")
    proposal = await backend.propose(
        role="analyst",
        team="test",
        brief={"evidence_ids": ["SYNTH-01"]},
        tier=ModelTier.LUNA,
    )
    assert proposal.proposal_id == "p1"
    call = backend.calls[0]
    assert call.provider_run_id == "fake-thread"
    assert call.process_started_at is not None
    assert call.first_event_latency_ms is not None
    assert call.provider_execution_ms is not None


async def test_two_transient_attempts_become_network_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = CodexExecBackend(_settings(tmp_path), executable="/usr/bin/true")
    backend.begin_run("network-test")

    async def fail(*args):  # type: ignore[no-untyped-def]
        raise OSError("temporary network/process failure")

    async def no_wait(seconds: float) -> None:
        assert seconds == 2

    monkeypatch.setattr(backend, "_invoke", fail)
    monkeypatch.setattr("resume_factory.codex_backend.asyncio.sleep", no_wait)
    with pytest.raises(CodexNetworkDegradedError):
        await backend.propose(
            role="analyst",
            team="test",
            brief={},
            tier=ModelTier.LUNA,
        )
    assert backend.calls[0].retry_count == 1


async def test_codex_concurrency_never_exceeds_three(tmp_path: Path, monkeypatch) -> None:
    backend = CodexExecBackend(
        _settings(tmp_path), max_concurrency=3, hard_call_cap=10, executable="/usr/bin/true"
    )
    backend.begin_run("concurrency-test")
    active = 0
    maximum = 0

    async def succeed(*args):  # type: ignore[no-untyped-def]
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return (
            AgentProposal.model_validate(_proposal_payload()),
            {"input_tokens": 10, "output_tokens": 2},
            "thread",
            {
                "process_started_at": datetime.now(UTC),
                "first_event_latency_ms": 1,
                "provider_execution_ms": 10,
            },
        )

    monkeypatch.setattr(backend, "_invoke", succeed)
    await asyncio.gather(
        *(
            backend.propose(
                role=f"role-{index}",
                team="test",
                brief={},
                tier=ModelTier.LUNA,
            )
            for index in range(6)
        )
    )
    assert maximum == 3


def test_codex_child_environment_drops_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-pass")
    monkeypatch.setenv("CODEX_API_KEY", "must-not-pass")
    environment = CodexExecBackend._safe_environment()
    assert "OPENAI_API_KEY" not in environment
    assert "CODEX_API_KEY" not in environment


def test_codex_schema_is_strict_at_every_object() -> None:
    schema = CodexExecBackend._strict_output_schema(
        {"type": "object", "properties": {"nested": {"type": "object", "properties": {}}}}
    )
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["nested"]
    assert schema["properties"]["nested"]["additionalProperties"] is False
