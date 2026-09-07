from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from .agents import AgentBackend
from .character_budget import bounds_for, rewrite_to_character_target
from .schemas import (
    ApplicationInput,
    BackendProvider,
    CallKind,
    CostStatus,
    DemandBrief,
    DraftAnswer,
    DraftProposal,
    EvidencePacket,
    ExecutionMode,
    ModelTier,
    PositioningBrief,
    PrepSoaraStructure,
    QuestionContract,
    RunResult,
    RunTelemetry,
    TeamDecision,
    TransferContract,
)
from .teams import TEAMS, WRITING_COUNCIL, run_team
from .validators import validate_answers

GRAPH_VERSION = "v0.5"


def _append(left: list[Any], right: list[Any]) -> list[Any]:
    return left + right


class ResumeGraphState(TypedDict, total=False):
    application: ApplicationInput
    mode: ExecutionMode
    demand_brief: DemandBrief
    question_contracts: list[QuestionContract]
    transfer_contracts: list[TransferContract]
    positioning_brief: PositioningBrief
    answers: list[DraftAnswer]
    character_rewrite_attempts: list[str]
    team_decisions: Annotated[list[TeamDecision], _append]
    validation: Any


def build_graph(backend: AgentBackend, checkpointer: Any | None = None) -> Any:
    async def intelligence(state: ResumeGraphState) -> dict[str, Any]:
        app = state["application"]
        decision, _ = await run_team(
            TEAMS["intelligence"],
            backend,
            {
                "company": app.company,
                "job": app.job,
                "job_description": app.job_description,
                "company_context": app.company_context,
                "evidence_ids": [item.event_id for item in app.evidence],
                "output_contract": (
                    "회사 문제, FA 책임, KPI, 실패 위험, 행동 역량을 근거 범위 안에서 제안"
                ),
            },
            state["mode"],
        )
        return {
            "demand_brief": DemandBrief(
                company_problems=[_first_clause(app.company_context)],
                responsibilities=[_first_clause(app.job_description)],
                kpis=["설비·물류 흐름의 사전 검증", "초기 안정화 조치의 추적 가능성"],
                failure_risks=["부분 최적화", "근거 없는 원인 단정", "시험·변경 기록 단절"],
                behavioral_competencies=["전체 흐름 판단", "안전 경계 분리", "사용자 기준 협업"],
                source_refs=["application.company_context", "application.job_description"],
            ),
            "team_decisions": [decision],
        }

    async def question_strategy(state: ResumeGraphState) -> dict[str, Any]:
        app = state["application"]
        decision, _ = await run_team(
            TEAMS["question_strategy"],
            backend,
            {
                "company": app.company,
                "job": app.job,
                "questions": [item.model_dump() for item in app.questions],
                "previous_outcomes": app.previous_outcomes,
                "evidence_ids": [item.event_id for item in app.evidence],
                "output_contract": "세 문항의 직접 답, 채용 의도, 요구 증거, 예상 반론을 함께 설계",
            },
            state["mode"],
        )
        contracts = []
        for question in app.questions:
            bounds = bounds_for(question.character_limit)
            contracts.append(
                QuestionContract(
                    question_id=question.question_id,
                    direct_answer_required=f"{question.text}에 첫 문장부터 직접 답한다.",
                    buyer_intent="경험의 판단 방식을 FA 업무의 구체적 행동으로 전환하는지 확인",
                    required_evidence_type="문제·판단·행동·결과·경계가 연결된 단일 사건",
                    required_company_connection=(
                        f"{app.company} {app.job}에서의 첫 행동과 검증 기준"
                    ),
                    likely_objections=["프로젝트와 실제 배터리 양산 현장의 차이는 무엇인가?"],
                    forbidden_generic_claims=["열정으로 기여하겠습니다", "최선을 다하겠습니다"],
                    character_budget=bounds.limit,
                    character_limit=bounds.limit,
                    character_hard_min=bounds.hard_min,
                    character_target_min=bounds.target_min,
                    character_target_max=bounds.target_max,
                )
            )
        return {"question_contracts": contracts, "team_decisions": [decision]}

    async def evidence_branding(state: ResumeGraphState) -> dict[str, Any]:
        app = state["application"]
        selected = [_select_evidence(item.question_id, app.evidence) for item in app.questions]
        decision, _ = await run_team(
            TEAMS["evidence_branding"],
            backend,
            {
                "company": app.company,
                "job": app.job,
                "questions": [item.model_dump() for item in app.questions],
                "evidence": [item.model_dump() for item in selected],
                "evidence_ids": [item.event_id for item in selected],
                "output_contract": "문항별 단일 경험, 금지 과장, 채용 이유, 회사 전이를 확정",
            },
            state["mode"],
        )
        transfers = [
            TransferContract(
                question_id=question.question_id,
                evidence_event_id=evidence.event_id,
                past_problem=evidence.problem,
                reusable_judgment=evidence.judgment,
                reusable_action=evidence.actions[0],
                company_task=state["demand_brief"].responsibilities[0],
                first_action=_first_action(question.question_id, app.company),
                output_or_kpi=_output_kpi(question.question_id),
                boundary=evidence.boundaries[0]
                if evidence.boundaries
                else "검증된 프로젝트 범위로 한정한다.",
            )
            for question, evidence in zip(app.questions, selected, strict=True)
        ]
        capabilities = list(dict.fromkeys(cap for item in selected for cap in item.capabilities))
        positioning = PositioningBrief(
            hire_me_reason=(
                "부분 기능보다 전체 흐름과 검증 경계를 먼저 설계해 "
                "투자 전 실패비용을 줄이는 판단 방식"
            ),
            primary_differentiator=(
                "결과를 과장하지 않고 시험 조건과 사용자 흐름까지 추적하는 실행력"
            ),
            supporting_differentiators=capabilities[:3] or ["근거 기반 판단"],
            proof_events=[item.event_id for item in selected],
            company_application=transfers[0].first_action,
            avoid_language=list(
                dict.fromkeys(boundary for item in selected for boundary in item.boundaries)
            ),
            anticipated_objections=[
                "가상 공정·프로젝트 경험을 양산 실적으로 오해할 수 있음",
                "측정하지 않은 안전 성능을 묻는 꼬리질문",
            ],
        )
        return {
            "transfer_contracts": transfers,
            "positioning_brief": positioning,
            "team_decisions": [decision],
        }

    async def writing_councils(state: ResumeGraphState) -> dict[str, Any]:
        app = state["application"]
        evidence_by_id = {item.event_id: item for item in app.evidence}

        async def write_one(index: int) -> tuple[DraftAnswer, TeamDecision]:
            question = app.questions[index]
            contract = state["question_contracts"][index]
            transfer = state["transfer_contracts"][index]
            evidence = evidence_by_id[transfer.evidence_event_id]
            brief = _writing_brief(
                app, question.model_dump(), contract, transfer, evidence, state["positioning_brief"]
            )
            decision, _ = await run_team(WRITING_COUNCIL, backend, brief, state["mode"])
            if decision.selected_draft is None:
                raise ValueError(
                    f"writing council returned no DraftProposal for {question.question_id}"
                )
            return _answer_from_proposal(
                decision.selected_draft, question.character_limit
            ), decision

        outputs = await asyncio.gather(*(write_one(index) for index in range(len(app.questions))))
        return {
            "answers": [item[0] for item in outputs],
            "team_decisions": [item[1] for item in outputs],
        }

    async def integration(state: ResumeGraphState) -> dict[str, Any]:
        app = state["application"]
        proposal = await backend.propose(
            role="integration_editor",
            team="integration",
            brief={
                "company": app.company,
                "job": app.job,
                "drafts": [
                    _proposal_from_answer(answer).model_dump() for answer in state["answers"]
                ],
                "evidence_ids": [item.event_id for item in app.evidence],
                "output_contract": (
                    "drafts 배열에 세 문항을 모두 반환. 사실·sentence_id·evidence_id를 "
                    "보존하며 반복과 목소리만 통합 편집"
                ),
            },
            tier=ModelTier.TERRA,
            call_kind=CallKind.LEAD,
        )
        if len(proposal.drafts) != len(state["answers"]):
            return {}
        limits = {item.question_id: item.character_limit for item in app.questions}
        return {
            "answers": [
                _answer_from_proposal(draft, limits[draft.question_id]) for draft in proposal.drafts
            ]
        }

    async def character_budget_gate(state: ResumeGraphState) -> dict[str, Any]:
        app = state["application"]
        evidence_by_id = {item.event_id: item for item in app.evidence}
        transfer_by_question = {item.question_id: item for item in state["transfer_contracts"]}
        contracts = {item.question_id: item for item in state["question_contracts"]}
        rewritten: list[DraftAnswer] = []
        attempted: list[str] = []
        for answer in state["answers"]:
            contract = contracts[answer.question_id]
            target_min = contract.character_target_min or 0
            target_max = contract.character_target_max or contract.character_budget
            if target_min <= answer.character_count <= target_max:
                rewritten.append(answer)
                continue
            attempted.append(answer.question_id)
            transfer = transfer_by_question[answer.question_id]
            evidence = evidence_by_id[transfer.evidence_event_id]
            if backend.provider is BackendProvider.LOCAL:
                updated, _ = rewrite_to_character_target(answer, app, evidence, transfer)
                backend.record_local_operation(
                    role="character_budget_rewriter",
                    team="integration",
                    question_id=answer.question_id,
                    call_kind=CallKind.CHARACTER_REWRITE,
                )
                rewritten.append(updated)
                continue
            proposal = await backend.propose(
                role="character_budget_rewriter",
                team="integration",
                brief={
                    **_writing_brief(
                        app,
                        _question_dump(app, answer.question_id),
                        contract,
                        transfer,
                        evidence,
                        state["positioning_brief"],
                    ),
                    "current_draft": _proposal_from_answer(answer).model_dump(),
                    "output_contract": (
                        f"draft 하나를 {target_min}~{target_max}자로 재작성. "
                        "새 사실·수치 금지, sentence plan 계보 유지"
                    ),
                },
                tier=ModelTier.TERRA,
                question_id=answer.question_id,
                call_kind=CallKind.CHARACTER_REWRITE,
            )
            rewritten.append(
                _answer_from_proposal(proposal.draft, answer.character_limit)
                if proposal.draft
                else answer
            )
        return {"answers": rewritten, "character_rewrite_attempts": attempted}

    def validate(state: ResumeGraphState) -> dict[str, Any]:
        return {"validation": validate_answers(state["application"], state["answers"])}

    builder = StateGraph(ResumeGraphState)
    for name, node in (
        ("company_job_intelligence", intelligence),
        ("question_strategy", question_strategy),
        ("evidence_branding", evidence_branding),
        ("writing_councils", writing_councils),
        ("integration", integration),
        ("character_budget_gate", character_budget_gate),
        ("deterministic_qa", validate),
    ):
        builder.add_node(name, node)
    nodes = [
        "company_job_intelligence",
        "question_strategy",
        "evidence_branding",
        "writing_councils",
        "integration",
        "character_budget_gate",
        "deterministic_qa",
    ]
    builder.add_edge(START, nodes[0])
    for left, right in zip(nodes, nodes[1:], strict=False):
        builder.add_edge(left, right)
    builder.add_edge(nodes[-1], END)
    return builder.compile(checkpointer=checkpointer)


async def run_resume_graph(
    application: ApplicationInput,
    backend: AgentBackend,
    mode: ExecutionMode,
    run_id: str | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> RunResult:
    resolved_run_id = run_id or uuid.uuid4().hex[:12]
    backend.begin_run(resolved_run_id)
    initial = (
        None
        if resume
        else {
            "application": application,
            "mode": mode,
            "team_decisions": [],
        }
    )
    config = {"configurable": {"thread_id": resolved_run_id}}
    if checkpoint_path is None:
        state = await build_graph(backend).ainvoke(initial, config)
    else:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            state = await build_graph(backend, saver).ainvoke(initial, config)
    calls = sorted(backend.calls, key=lambda call: call.sequence)
    model_calls = [call for call in calls if call.provider is not BackendProvider.LOCAL]
    optional_calls = sum(
        call.call_kind in {CallKind.ADJUDICATOR, CallKind.CHARACTER_REWRITE} for call in model_calls
    )
    question_count = len(application.questions)
    validation = state["validation"]
    telemetry = RunTelemetry(
        run_id=resolved_run_id,
        mode=mode,
        model_calls=calls,
        total_cost_usd=round(sum(call.estimated_cost_usd or 0 for call in calls), 6),
        cost_complete=all(call.cost_status is not CostStatus.UNKNOWN for call in calls),
        price_catalog_version=backend.price_catalog_version,
        prompt_versions={"core": "v0.5.0"},
        metadata={"character_rewrite_attempts": state.get("character_rewrite_attempts", [])},
        provider=backend.provider,
        billing_mode=backend.billing_mode,
        cost_status=backend.cost_status,
        graph_version=GRAPH_VERSION,
        graph_nodes_completed=[
            "company_job_intelligence",
            "question_strategy",
            "evidence_branding",
            "writing_councils",
            "integration",
            "character_budget_gate",
            "deterministic_qa",
        ],
        base_call_budget=10 + 4 * question_count,
        optional_calls_used=optional_calls,
        hard_call_cap=10 + 6 * question_count,
    )
    return RunResult(
        run_id=resolved_run_id,
        status="validated" if validation.passed else "needs_review",
        input_summary={"company": application.company, "job": application.job},
        demand_brief=state["demand_brief"],
        question_contracts=state["question_contracts"],
        transfer_contracts=state["transfer_contracts"],
        positioning_brief=state["positioning_brief"],
        answers=state["answers"],
        validation=validation,
        team_decisions=state["team_decisions"],
        eligibility_warnings=application.eligibility_notes,
        telemetry=telemetry,
    )


def _writing_brief(
    app: ApplicationInput,
    question: dict[str, Any],
    contract: QuestionContract,
    transfer: TransferContract,
    evidence: EvidencePacket,
    positioning: PositioningBrief,
) -> dict[str, Any]:
    return {
        "company": app.company,
        "job": app.job,
        "question_id": question["question_id"],
        "question": question["text"],
        "writing_contract": contract.model_dump(),
        "positioning": positioning.model_dump(),
        "transfer": transfer.model_dump(),
        "evidence": evidence.model_dump(),
        "evidence_ids": [evidence.event_id],
        "rules": [
            "모든 문장은 독립적인 판매 가치와 역할을 가짐",
            "수치는 verified numeric_authorities만 사용",
            "실제 양산 실적으로 확대하지 않음",
            "소제목·줄바꿈·공백 포함 Python len 기준 목표 구간 준수",
            (
                "정확히 10개 sentence plan을 사용: answer 1, company_need 1, "
                "perspective 1, differentiation 1, problem 1, judgment 1, action 1, "
                "result 1, validation 1, transfer 1"
            ),
            "근거 문장은 interview_defensible=true, 회사 전이 문장은 company_connection 명시",
        ],
        "output_contract": (
            "draft 필수. sentence_plans의 text를 순서대로 합치면 본문이 되며 "
            "각 문장은 evidence 또는 company_connection을 가짐"
        ),
    }


def _answer_from_proposal(proposal: DraftProposal, character_limit: int | None) -> DraftAnswer:
    plans = [item.model_copy(deep=True) for item in proposal.sentence_plans]
    for index, plan in enumerate(plans, start=1):
        plan.sentence_id = f"{proposal.question_id}-S{index:02d}"
        if not plan.claim_ids:
            plan.claim_ids = [f"{proposal.question_id}-C{index:02d}"]
    return DraftAnswer(
        question_id=proposal.question_id,
        headline=proposal.headline,
        body=" ".join(item.text.strip() for item in plans),
        sentence_plans=plans,
        evidence_ids=list(dict.fromkeys(proposal.evidence_ids)),
        character_limit=character_limit,
    )


def _proposal_from_answer(answer: DraftAnswer) -> DraftProposal:
    return DraftProposal(
        question_id=answer.question_id,
        headline=answer.headline,
        direct_answer=answer.sentence_plans[0].text if answer.sentence_plans else "",
        prep_soara_structure=PrepSoaraStructure(),
        sentence_plans=answer.sentence_plans,
        evidence_ids=answer.evidence_ids,
        company_transfer=next(
            (item.text for item in answer.sentence_plans if item.role.value == "transfer"), ""
        ),
        interview_defense=[],
        confidence=0.8,
    )


def _question_dump(app: ApplicationInput, question_id: str) -> dict[str, Any]:
    return next(item.model_dump() for item in app.questions if item.question_id == question_id)


def _first_clause(text: str) -> str:
    stripped = text.strip()
    for delimiter in ("\n", ".", "다."):
        if delimiter in stripped:
            return stripped.split(delimiter, 1)[0].strip() or stripped
    return stripped


def _select_evidence(question_id: str, evidence: list[EvidencePacket]) -> EvidencePacket:
    return next((item for item in evidence if question_id in item.best_for_questions), evidence[0])


def _first_action(question_id: str, company: str) -> str:
    return {
        "Q1": (
            f"{company}의 물류설비 사양과 공정 간 대기·투입 조건을 확인해 "
            "대안을 같은 기준으로 비교하겠습니다."
        ),
        "Q2": (
            f"{company} 설비의 정지·복구 조건을 기능별로 나누고 "
            "시험 결과와 미측정 범위를 함께 기록하겠습니다."
        ),
        "Q3": (
            f"{company} 현업 사용 순서와 개발 변경 범위를 먼저 합의하고 "
            "동일 입력의 판정 유지 여부를 인수 기준으로 삼겠습니다."
        ),
    }.get(question_id, f"{company}의 현장 기준과 이력을 먼저 확인하겠습니다.")


def _output_kpi(question_id: str) -> str:
    return {
        "Q1": "대안별 병목·처리량·초기 안정화 조건",
        "Q2": "정지·복구 시험의 재현성과 미확인 위험",
        "Q3": "요구사항·변경·시험 결과의 추적성",
    }.get(question_id, "검증 가능한 직무 산출물")
