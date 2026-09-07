from __future__ import annotations

import hashlib
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

from .schemas import (
    CostStatus,
    FeedbackDecision,
    HumanFeedback,
    ModelCallRecord,
    RunResult,
    SubmissionBundle,
)

GroupBy = Literal["question", "team", "agent", "model"]


def aggregate_calls(
    calls: list[ModelCallRecord], group_by: GroupBy | None = None
) -> list[dict[str, Any]]:
    grouped: dict[str, list[ModelCallRecord]] = defaultdict(list)
    if group_by is None:
        grouped["total"] = calls
    else:
        for call in calls:
            key = _group_key(call, group_by)
            grouped[key].append(call)
    return [_aggregate_row(key, items) for key, items in sorted(grouped.items())]


def usage_report(result: RunResult, group_by: GroupBy = "team") -> dict[str, Any]:
    top_agents = sorted(
        aggregate_calls(result.telemetry.model_calls, "agent"),
        key=lambda row: (float(row["estimated_cost_usd"]), int(row["total_tokens"])),
        reverse=True,
    )[:10]
    return {
        "run_id": result.run_id,
        "company": result.input_summary["company"],
        "job": result.input_summary["job"],
        "status": result.status,
        "mode": result.telemetry.mode.value,
        "price_catalog_version": result.telemetry.price_catalog_version,
        "cost_complete": result.telemetry.cost_complete,
        "provider": result.telemetry.provider.value,
        "billing_mode": result.telemetry.billing_mode.value,
        "cost_status": result.telemetry.cost_status.value,
        "graph_version": result.telemetry.graph_version,
        "graph_nodes_completed": result.telemetry.graph_nodes_completed,
        "call_budget": {
            "base": result.telemetry.base_call_budget,
            "optional_used": result.telemetry.optional_calls_used,
            "hard_cap": result.telemetry.hard_call_cap,
        },
        "character_rewrite_count": len(
            result.telemetry.metadata.get("character_rewrite_attempts", [])
        ),
        "total": aggregate_calls(result.telemetry.model_calls)[0],
        "groups": aggregate_calls(result.telemetry.model_calls, group_by),
        "top_agents": top_agents,
        "validation": result.validation.metrics,
    }


def compare_reports(results: list[RunResult], group_by: GroupBy = "team") -> dict[str, Any]:
    return {
        "runs": [usage_report(result, group_by) for result in results],
        "quality_is_vector": True,
    }


def build_feedback(
    result: RunResult,
    *,
    decision: FeedbackDecision,
    rating: int,
    final_draft: Path | None,
) -> HumanFeedback:
    if final_draft is None:
        return HumanFeedback(run_id=result.run_id, decision=decision, rating=rating)
    raw = final_draft.read_bytes()
    bundle = SubmissionBundle.model_validate_json(raw)
    generated = {answer.question_id: answer.submission_text for answer in result.answers}
    final = {answer.question_id: answer.submission_text for answer in bundle.answers}
    ratios = {
        question_id: _edit_ratio(generated[question_id], text)
        for question_id, text in final.items()
        if question_id in generated
    }
    overall = round(sum(ratios.values()) / len(ratios), 4) if ratios else None
    return HumanFeedback(
        run_id=result.run_id,
        decision=decision,
        rating=rating,
        final_draft_path=str(final_draft.resolve()),
        final_draft_hash=hashlib.sha256(raw).hexdigest(),
        overall_edit_ratio=overall,
        per_question_edit_ratio=ratios,
    )


def _edit_ratio(original: str, revised: str) -> float:
    similarity = SequenceMatcher(None, original, revised).ratio()
    return round(1 - similarity, 4)


def _group_key(call: ModelCallRecord, group_by: GroupBy) -> str:
    if group_by == "question":
        return call.question_id or "shared"
    if group_by == "team":
        return call.team
    if group_by == "agent":
        return call.agent_role
    return call.model


def _aggregate_row(key: str, calls: list[ModelCallRecord]) -> dict[str, Any]:
    known_cost = round(sum(call.estimated_cost_usd or 0 for call in calls), 8)
    cost_applicable = any(call.cost_status is not CostStatus.NOT_APPLICABLE for call in calls)
    return {
        "group": key,
        "calls": len(calls),
        "input_tokens": sum(call.input_tokens for call in calls),
        "cached_input_tokens": sum(call.cached_input_tokens for call in calls),
        "output_tokens": sum(call.output_tokens for call in calls),
        "reasoning_tokens": sum(call.reasoning_tokens for call in calls),
        "total_tokens": sum(call.total_tokens for call in calls),
        "estimated_cost_usd": known_cost,
        "cost_display": f"{known_cost:.8f}" if cost_applicable else "N/A",
        "cost_complete": all(
            call.estimated_cost_usd is not None or call.cost_status is CostStatus.NOT_APPLICABLE
            for call in calls
        ),
        "latency_ms": sum(call.latency_ms for call in calls),
        "failures": sum(not call.success for call in calls),
        "missing_usage_calls": sum(call.usage_status.value == "missing" for call in calls),
        "offline_calls": sum(call.usage_status.value == "offline" for call in calls),
        "legacy_calls": sum(call.usage_status.value == "legacy" for call in calls),
        "unknown_price_calls": sum(call.cost_status is CostStatus.UNKNOWN for call in calls),
        "known_retries": sum(call.retry_count or 0 for call in calls),
        "unknown_retry_calls": sum(call.retry_count is None for call in calls),
    }
