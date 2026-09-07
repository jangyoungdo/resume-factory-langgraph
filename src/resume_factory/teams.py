from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .agents import AgentBackend, AgentSpec
from .schemas import AgentProposal, ExecutionMode, ModelTier, TeamDecision


@dataclass(frozen=True)
class TeamDefinition:
    name: str
    specialists: tuple[AgentSpec, ...]
    lead_role: str
    lead_tier: ModelTier = ModelTier.TERRA


TEAMS: dict[str, TeamDefinition] = {
    "intelligence": TeamDefinition(
        "intelligence",
        (
            AgentSpec("business_analyst", ModelTier.LUNA, "사업 문제와 경쟁 방식"),
            AgentSpec("job_demand_analyst", ModelTier.LUNA, "JD 책임과 요구 역량"),
            AgentSpec("process_kpi_analyst", ModelTier.TERRA, "업무 프로세스와 KPI"),
            AgentSpec("industry_comparison_analyst", ModelTier.LUNA, "산업 특수성"),
        ),
        "intelligence_lead",
    ),
    "question_strategy": TeamDefinition(
        "question_strategy",
        (
            AgentSpec("literal_question_analyst", ModelTier.LUNA, "명시적 요구"),
            AgentSpec("recruiter_intent_analyst", ModelTier.LUNA, "채용 구매 의도"),
            AgentSpec("hiring_manager_analyst", ModelTier.TERRA, "실무 투입 관점"),
            AgentSpec("adversarial_interviewer", ModelTier.LUNA, "예상 반론"),
        ),
        "answer_architect",
    ),
    "evidence": TeamDefinition(
        "evidence",
        (
            AgentSpec("relevance_matcher", ModelTier.LUNA, "질문 관련성"),
            AgentSpec("differentiation_analyst", ModelTier.TERRA, "지원자 차별성"),
            AgentSpec("boundary_agent", ModelTier.LUNA, "기여 범위와 과장 방지"),
            AgentSpec("transfer_designer", ModelTier.TERRA, "회사 업무 전이"),
        ),
        "evidence_lead",
    ),
    "branding": TeamDefinition(
        "branding",
        (
            AgentSpec("positioning_strategist", ModelTier.TERRA, "채용 포지셔닝"),
            AgentSpec("recruiter_simulator", ModelTier.LUNA, "서류 첫인상"),
            AgentSpec("hiring_manager_simulator", ModelTier.TERRA, "실무 기여"),
            AgentSpec("brand_copy_strategist", ModelTier.LUNA, "일관된 브랜드 언어"),
            AgentSpec("skeptical_recruiter", ModelTier.LUNA, "회사 교체 가능성과 과장"),
        ),
        "brand_director",
    ),
}


BLOCK_TEAMS: dict[str, TeamDefinition] = {
    "headline": TeamDefinition(
        "headline",
        (
            AgentSpec("direct_value_writer", ModelTier.LUNA, "질문 답과 채용 가치"),
            AgentSpec("evidence_hook_writer", ModelTier.LUNA, "수치와 행동 후크"),
            AgentSpec("company_fit_writer", ModelTier.LUNA, "회사 적용"),
        ),
        "headline_editor",
    ),
    "first_p": TeamDefinition(
        "first_p",
        (
            AgentSpec("direct_answer_writer", ModelTier.LUNA, "첫 문장 직접 답변"),
            AgentSpec("positioning_writer", ModelTier.LUNA, "채용 이유 선제시"),
        ),
        "first_p_editor",
    ),
    "reason": TeamDefinition(
        "reason",
        (
            AgentSpec("reasoning_writer", ModelTier.LUNA, "지원자의 판단 기준"),
            AgentSpec("job_logic_writer", ModelTier.LUNA, "직무 요구 논리"),
        ),
        "reason_editor",
    ),
    "soara": TeamDefinition(
        "soara",
        (
            AgentSpec("fact_architect", ModelTier.TERRA, "사건과 수치 고정"),
            AgentSpec("situation_obstacle_writer", ModelTier.LUNA, "상황과 장애물"),
            AgentSpec("action_judgment_writer", ModelTier.LUNA, "판단과 행동"),
            AgentSpec("result_validation_writer", ModelTier.LUNA, "결과와 검증"),
            AgentSpec("boundary_writer", ModelTier.LUNA, "기여 범위"),
        ),
        "evidence_editor",
    ),
    "closing": TeamDefinition(
        "closing",
        (
            AgentSpec("transfer_writer", ModelTier.LUNA, "경험의 직무 전이"),
            AgentSpec("first_action_writer", ModelTier.LUNA, "입사 후 첫 행동"),
            AgentSpec("kpi_writer", ModelTier.LUNA, "산출물과 KPI"),
        ),
        "closing_editor",
    ),
}

QA_TEAMS: dict[str, TeamDefinition] = {
    "fact_qa": TeamDefinition(
        "fact_qa",
        (
            AgentSpec("claim_extractor", ModelTier.LUNA, "원자 사실 주장 추출"),
            AgentSpec("fact_prosecutor", ModelTier.TERRA, "근거 없는 주장 공격"),
        ),
        "fact_gatekeeper",
    ),
    "sentence_value_qa": TeamDefinition(
        "sentence_value_qa",
        (
            AgentSpec("sentence_role_classifier", ModelTier.LUNA, "문장 역할 분류"),
            AgentSpec("deletion_test_agent", ModelTier.LUNA, "삭제 시 정보 손실"),
            AgentSpec("perspective_auditor", ModelTier.LUNA, "관점과 차별점"),
        ),
        "sentence_value_lead",
    ),
    "marketing_qa": TeamDefinition(
        "marketing_qa",
        (
            AgentSpec("recruiter_reader", ModelTier.LUNA, "채용 이유의 즉시성"),
            AgentSpec("company_swap_tester", ModelTier.LUNA, "회사 교체 가능성"),
            AgentSpec("differentiation_auditor", ModelTier.TERRA, "경쟁 지원자 대비 차이"),
        ),
        "marketing_gatekeeper",
    ),
    "interview_defense_qa": TeamDefinition(
        "interview_defense_qa",
        (
            AgentSpec("followup_question_generator", ModelTier.LUNA, "예상 꼬리질문"),
            AgentSpec("evidence_defense_agent", ModelTier.LUNA, "답변 근거"),
            AgentSpec("overclaim_interrogator", ModelTier.TERRA, "과장 가능성"),
        ),
        "interview_defense_lead",
    ),
}


class TeamRunState(TypedDict, total=False):
    definition: TeamDefinition
    backend: AgentBackend
    brief: dict[str, Any]
    mode: ExecutionMode
    specs: list[AgentSpec]
    spec: AgentSpec
    proposals: Annotated[list[AgentProposal], operator.add]
    decision: TeamDecision


def _adjust_for_mode(specs: tuple[AgentSpec, ...], mode: ExecutionMode) -> list[AgentSpec]:
    if mode is ExecutionMode.ECONOMY:
        return list(specs[:2])
    if mode is ExecutionMode.PREMIUM:
        return list(specs)
    return list(specs[: max(2, len(specs) - 1)])


async def run_team(
    definition: TeamDefinition,
    backend: AgentBackend,
    brief: dict[str, Any],
    mode: ExecutionMode,
) -> tuple[TeamDecision, list[AgentProposal]]:
    result = await TEAM_SUBGRAPH.ainvoke(
        {
            "definition": definition,
            "backend": backend,
            "brief": brief,
            "mode": mode,
            "proposals": [],
        }
    )
    return result["decision"], result["proposals"]


def _prepare_team(state: TeamRunState) -> dict[str, Any]:
    return {"specs": _adjust_for_mode(state["definition"].specialists, state["mode"])}


def _dispatch_specialists(state: TeamRunState) -> list[Send]:
    return [
        Send(
            "specialist",
            {
                "definition": state["definition"],
                "backend": state["backend"],
                "brief": state["brief"],
                "mode": state["mode"],
                "spec": spec,
            },
        )
        for spec in state["specs"]
    ]


async def _run_specialist(state: TeamRunState) -> dict[str, Any]:
    spec = state["spec"]
    proposal = await state["backend"].propose(
        role=spec.role,
        team=state["definition"].name,
        brief={**state["brief"], "focus": spec.focus},
        tier=spec.tier,
    )
    return {"proposals": [proposal]}


async def _finalize_team(state: TeamRunState) -> dict[str, Any]:
    definition = state["definition"]
    backend = state["backend"]
    brief = state["brief"]
    mode = state["mode"]
    proposals = state["proposals"]
    ranked = sorted(proposals, key=lambda item: item.score.weighted, reverse=True)
    winner = ranked[0]
    margin = winner.score.weighted - ranked[1].score.weighted if len(ranked) > 1 else 5.0
    anonymous_candidates = [
        {
            "proposal_id": item.proposal_id,
            "recommendation": item.recommendation,
            "evidence_ids": item.evidence_ids,
            "risks": item.risks,
            "weighted_score": item.score.weighted,
        }
        for item in ranked
    ]
    critic = await backend.propose(
        role=f"{definition.name}_anonymous_critic",
        team=definition.name,
        brief={**brief, "anonymous_candidates": anonymous_candidates},
        tier=ModelTier.LUNA,
    )
    lead = await backend.propose(
        role=definition.lead_role,
        team=definition.name,
        brief={
            **brief,
            "anonymous_candidates": anonymous_candidates,
            "critic": critic.model_dump(),
        },
        tier=definition.lead_tier,
    )
    needs_sol = (
        mode is not ExecutionMode.ECONOMY
        and (margin < 0.3 or winner.confidence < 0.75)
        and any(item.needs_escalation for item in ranked)
    )
    escalated_to = ModelTier.SOL if needs_sol else None
    if needs_sol:
        await backend.propose(
            role=f"{definition.name}_adjudicator",
            team=definition.name,
            brief={
                **brief,
                "anonymous_candidates": anonymous_candidates[:2],
                "critic": critic.model_dump(),
                "lead": lead.model_dump(),
            },
            tier=ModelTier.SOL,
        )
    return {
        "decision": TeamDecision(
            team=definition.name,
            selected_proposal_id=winner.proposal_id,
            decision=lead.recommendation,
            rationale=(
                f"익명 점수 {winner.score.weighted:.2f}, 차점자 대비 {margin:.2f}; "
                "근거 충실도와 회사 적용성을 우선"
            ),
            confidence=winner.confidence,
            evidence_ids=winner.evidence_ids,
            escalated_to=escalated_to,
        )
    }


def _build_team_subgraph() -> Any:
    builder = StateGraph(TeamRunState)
    builder.add_node("prepare", _prepare_team)
    builder.add_node("specialist", _run_specialist)
    builder.add_node("critic_and_lead", _finalize_team)
    builder.add_edge(START, "prepare")
    builder.add_conditional_edges("prepare", _dispatch_specialists, ["specialist"])
    builder.add_edge("specialist", "critic_and_lead")
    builder.add_edge("critic_and_lead", END)
    return builder.compile()


TEAM_SUBGRAPH = _build_team_subgraph()
