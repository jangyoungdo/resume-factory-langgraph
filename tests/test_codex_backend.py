import json
from pathlib import Path

import pytest

from resume_factory.codex_backend import CodexCallBudgetExceeded, CodexExecBackend
from resume_factory.config import Settings
from resume_factory.schemas import (
    BackendProvider,
    BillingMode,
    CostStatus,
    ExecutionMode,
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
