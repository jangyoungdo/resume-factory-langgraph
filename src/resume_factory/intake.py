from __future__ import annotations

import json
from typing import Any

from .mcp.client import load_tools_isolated
from .schemas import ApplicationInput, EvidencePacket


class IntakeError(RuntimeError):
    pass


async def load_application_from_mcp(application_id: str) -> ApplicationInput:
    """Load an application and re-verify every selected evidence item through MCP."""
    tools, health = await load_tools_isolated(timeout_seconds=15)
    failed = [name for name, status in health.items() if status != "healthy"]
    if failed:
        raise IntakeError(f"required MCP server unavailable: {', '.join(failed)}")

    async def invoke(suffix: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = next((item for item in tools if item.name.endswith(suffix)), None)
        if tool is None:
            raise IntakeError(f"required MCP tool missing: {suffix}")
        result = await tool.ainvoke(arguments)
        if isinstance(result, list) and result:
            block = result[0]
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                try:
                    result = json.loads(block["text"])
                except json.JSONDecodeError as error:
                    raise IntakeError(f"MCP {suffix} returned invalid JSON") from error
        if not isinstance(result, dict) or result.get("error"):
            raise IntakeError(f"MCP {suffix} failed: {result}")
        return result

    application = await invoke("get_application", {"application_id": application_id})
    jd = await invoke("get_job_description", {"application_id": application_id})
    question_result = await invoke("get_application_questions", {"application_id": application_id})
    draft_result = await invoke("get_existing_draft", {"application_id": application_id})
    questions = question_result.get("questions") or []
    mapping = application.get("evidence_ids_by_question") or {}
    requested_ids: list[str] = []
    for question in questions:
        question_id = str(question["question_id"])
        await invoke(
            "search_evidence",
            {
                "query": f"{jd.get('company')} {jd.get('job')} {question.get('text', '')}",
                "capabilities": application.get("search_capabilities", []),
                "top_k": 5,
            },
        )
        event_id = mapping.get(question_id)
        if not event_id:
            raise IntakeError(f"no approved evidence ID for {question_id}")
        requested_ids.append(str(event_id))

    evidence: list[EvidencePacket] = []
    for event_id in dict.fromkeys(requested_ids):
        raw = await invoke("get_evidence_event", {"event_id": event_id})
        boundary = await invoke("get_project_boundary", {"event_id": event_id})
        raw["boundaries"] = boundary.get("boundaries", [])
        raw["forbidden_combinations"] = boundary.get("forbidden_combinations", [])
        for authority in raw.get("numeric_authorities", []):
            verified = await invoke(
                "get_numeric_authority",
                {"authority_id": authority["authority_id"]},
            )
            if verified.get("status") != "verified":
                raise IntakeError(f"unverified numeric authority: {authority['authority_id']}")
        evidence.append(EvidencePacket.model_validate(raw))

    feedback = await invoke(
        "get_previous_feedback",
        {"company": str(jd.get("company")), "question_type": "application"},
    )
    return ApplicationInput(
        company=str(jd["company"]),
        job=str(jd["job"]),
        industry=str(application.get("industry", "battery manufacturing")),
        job_description=str(jd["job_description"]),
        company_context=str(jd["company_context"]),
        questions=questions,
        evidence=evidence,
        eligibility_notes=list(application.get("eligibility_notes", [])),
        existing_draft=draft_result.get("existing_draft"),
        previous_outcomes=[str(item.get("feedback", item)) for item in feedback.get("results", [])]
        + [str(item) for item in application.get("previous_outcomes", [])],
    )
