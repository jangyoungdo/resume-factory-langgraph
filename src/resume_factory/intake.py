from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from typing import Any, Literal

from .mcp.client import open_tools_persistent
from .schemas import (
    ApplicationInput,
    EvidencePacket,
    MaterialSelectionMode,
    RunPhaseSpan,
)


class IntakeError(RuntimeError):
    pass


async def load_application_from_mcp(
    application_id: str, phase_spans: list[RunPhaseSpan] | None = None
) -> ApplicationInput:
    """Load an application and re-verify every selected evidence item through MCP."""
    spans = phase_spans if phase_spans is not None else []
    startup_at = datetime.now(UTC)
    startup = time.perf_counter()
    startup_metrics: list[dict[str, Any]] = []
    async with open_tools_persistent(timeout_seconds=15, startup_metrics=startup_metrics) as (
        tools,
        health,
    ):
        failed = [name for name, status in health.items() if status != "healthy"]
        spans.append(
            RunPhaseSpan(
                phase="mcp_startup",
                started_at=startup_at,
                completed_at=datetime.now(UTC),
                duration_ms=int((time.perf_counter() - startup) * 1000),
                status="failed" if failed else "completed",
            )
        )
        spans.extend(
            RunPhaseSpan(
                phase="mcp_startup_server",
                started_at=startup_at,
                completed_at=datetime.now(UTC),
                duration_ms=int(item["duration_ms"]),
                status=("completed" if health.get(str(item["server"])) == "healthy" else "failed"),
                detail=str(item["server"]),
            )
            for item in startup_metrics
        )
        if failed:
            raise IntakeError(f"required MCP server unavailable: {', '.join(failed)}")

        async def invoke(suffix: str, arguments: dict[str, Any]) -> dict[str, Any]:
            tool_at = datetime.now(UTC)
            tool_started = time.perf_counter()
            tool_status: Literal["completed", "failed"] = "completed"
            tool = next((item for item in tools if item.name.endswith(suffix)), None)
            if tool is None:
                raise IntakeError(f"required MCP tool missing: {suffix}")
            try:
                result = await tool.ainvoke(arguments)
                if isinstance(result, list) and result:
                    parsed_result = None
                    for block in result:
                        if not isinstance(block, dict) or not isinstance(block.get("text"), str):
                            continue
                        try:
                            candidate = json.loads(block["text"])
                        except json.JSONDecodeError:
                            continue
                        if isinstance(candidate, dict):
                            parsed_result = candidate
                            break
                    if parsed_result is None:
                        raise IntakeError(f"MCP {suffix} returned invalid JSON")
                    result = parsed_result
                if not isinstance(result, dict) or result.get("error"):
                    raise IntakeError(f"MCP {suffix} failed: {result}")
                return result
            except Exception:
                tool_status = "failed"
                raise
            finally:
                spans.append(
                    RunPhaseSpan(
                        phase="mcp_tool",
                        started_at=tool_at,
                        completed_at=datetime.now(UTC),
                        duration_ms=int((time.perf_counter() - tool_started) * 1000),
                        status=tool_status,
                        detail=suffix,
                    )
                )

        read_at = datetime.now(UTC)
        read_started = time.perf_counter()
        try:
            application, jd, question_result, draft_result = await asyncio.gather(
                invoke("get_application", {"application_id": application_id}),
                invoke("get_job_description", {"application_id": application_id}),
                invoke("get_application_questions", {"application_id": application_id}),
                invoke("get_existing_draft", {"application_id": application_id}),
            )
            questions = question_result.get("questions") or []
            legacy_mapping = application.get("evidence_ids_by_question") or {}
            material_selection = application.get("material_selection") or {}
            if isinstance(material_selection, str):
                material_selection = {"mode": material_selection}
            selection_mode = MaterialSelectionMode(
                material_selection.get("mode", MaterialSelectionMode.PINNED)
            )
            pinned_mapping = (
                material_selection.get("pinned_evidence_ids_by_question") or legacy_mapping
            )
            preferred_mapping = (
                material_selection.get("preferred_evidence_ids_by_question")
                or application.get("preferred_evidence_ids_by_question")
                or (legacy_mapping if selection_mode is MaterialSelectionMode.AUTO_UNIQUE else {})
            )
            supporting_mapping = application.get("supporting_evidence_ids_by_question") or {}
            for question in questions:
                question_id = str(question["question_id"])
                question["supporting_evidence_ids"] = [
                    str(item) for item in supporting_mapping.get(question_id, [])
                ]
            search_results = await asyncio.gather(
                *(
                    invoke(
                        "search_evidence",
                        {
                            "query": (
                                f"{jd.get('company')} {jd.get('job')} {question.get('text', '')}"
                            ),
                            "capabilities": application.get("search_capabilities", []),
                            "top_k": 8,
                        },
                    )
                    for question in questions
                )
            )
            requested_ids: list[str] = []
            for question in questions:
                question_id = str(question["question_id"])
                event_id = (
                    pinned_mapping.get(question_id)
                    if selection_mode is MaterialSelectionMode.PINNED
                    else preferred_mapping.get(question_id)
                )
                if selection_mode is MaterialSelectionMode.PINNED and not event_id:
                    raise IntakeError(f"no pinned evidence ID for {question_id}")
                if event_id:
                    requested_ids.append(str(event_id))
                requested_ids.extend(question["supporting_evidence_ids"])
            if selection_mode is MaterialSelectionMode.AUTO_UNIQUE:
                requested_ids.extend(
                    str(item["event_id"])
                    for result in search_results
                    for item in result.get("results", [])
                    if item.get("event_id")
                )

            async def load_evidence(event_id: str) -> EvidencePacket:
                raw, boundary = await asyncio.gather(
                    invoke("get_evidence_event", {"event_id": event_id}),
                    invoke("get_project_boundary", {"event_id": event_id}),
                )
                raw["boundaries"] = boundary.get("boundaries", [])
                raw["forbidden_combinations"] = boundary.get("forbidden_combinations", [])
                if selection_mode is MaterialSelectionMode.AUTO_UNIQUE and not raw.get(
                    "experience_key"
                ):
                    raise IntakeError(f"missing experience_key for auto allocation: {event_id}")
                authorities = {
                    str(item["authority_id"]): item for item in raw.get("numeric_authorities", [])
                }
                verified = await asyncio.gather(
                    *(
                        invoke("get_numeric_authority", {"authority_id": authority_id})
                        for authority_id in authorities
                    )
                )
                if any(item.get("status") != "verified" for item in verified):
                    raise IntakeError(f"unverified numeric authority for {event_id}")
                return EvidencePacket.model_validate(raw)

            evidence, feedback = await asyncio.gather(
                asyncio.gather(*(load_evidence(item) for item in dict.fromkeys(requested_ids))),
                invoke(
                    "get_previous_feedback",
                    {"company": str(jd.get("company")), "question_type": "application"},
                ),
            )
        except Exception:
            status: Literal["completed", "failed"] = "failed"
            raise
        else:
            status = "completed"
        finally:
            spans.append(
                RunPhaseSpan(
                    phase="mcp_read",
                    started_at=read_at,
                    completed_at=datetime.now(UTC),
                    duration_ms=int((time.perf_counter() - read_started) * 1000),
                    status=status,
                )
            )

        return ApplicationInput(
            company=str(jd["company"]),
            job=str(jd["job"]),
            industry=str(application.get("industry", "battery manufacturing")),
            job_description=str(jd["job_description"]),
            company_context=str(jd["company_context"]),
            questions=questions,
            evidence=list(evidence),
            eligibility_notes=list(application.get("eligibility_notes", [])),
            existing_draft=draft_result.get("existing_draft"),
            previous_outcomes=[
                str(item.get("feedback", item)) for item in feedback.get("results", [])
            ]
            + [str(item) for item in application.get("previous_outcomes", [])],
            material_selection_mode=selection_mode,
            pinned_evidence_ids_by_question={
                str(key): str(value) for key, value in pinned_mapping.items()
            },
            preferred_evidence_ids_by_question={
                str(key): str(value) for key, value in preferred_mapping.items()
            },
        )
