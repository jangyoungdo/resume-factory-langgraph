from __future__ import annotations

import asyncio
import math
import uuid
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import AgentBackend
from .character_budget import rewrite_to_character_target
from .schemas import (
    ApplicationInput,
    CallKind,
    DemandBrief,
    DraftAnswer,
    EvidencePacket,
    ExecutionMode,
    PositioningBrief,
    QuestionContract,
    RunResult,
    RunTelemetry,
    SentencePlan,
    SentenceRole,
    TeamDecision,
    TransferContract,
)
from .teams import BLOCK_TEAMS, QA_TEAMS, TEAMS, run_team
from .validators import validate_answers


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


def build_graph(backend: AgentBackend) -> Any:
    async def intelligence(state: ResumeGraphState) -> dict[str, Any]:
        app = state["application"]
        evidence_ids = [item.event_id for item in app.evidence]
        decision, _ = await run_team(
            TEAMS["intelligence"],
            backend,
            {
                "company": app.company,
                "job": app.job,
                "job_description": app.job_description,
                "company_context": app.company_context,
                "evidence_ids": evidence_ids,
            },
            state["mode"],
        )
        brief = DemandBrief(
            company_problems=[_first_clause(app.company_context)],
            responsibilities=[_first_clause(app.job_description)],
            kpis=["예방보전 조치의 추적 가능성", "설비 이상 대응 우선순위의 일관성"],
            failure_risks=["근거 없는 진단", "점검 우선순위 불명확", "기록 단절"],
            behavioral_competencies=["데이터 기반 판단", "현장 검증", "협업과 기록"],
            source_refs=["application.company_context", "application.job_description"],
        )
        return {"demand_brief": brief, "team_decisions": [decision]}

    async def question_strategy(state: ResumeGraphState) -> dict[str, Any]:
        app = state["application"]
        decision, _ = await run_team(
            TEAMS["question_strategy"],
            backend,
            {
                "company": app.company,
                "job": app.job,
                "questions": [q.model_dump() for q in app.questions],
                "evidence_ids": [item.event_id for item in app.evidence],
            },
            state["mode"],
        )
        contracts = [
            QuestionContract(
                question_id=question.question_id,
                direct_answer_required=f"{question.text}에 첫 문장부터 직접 답한다.",
                buyer_intent="검증된 경험을 해당 직무의 구체적 행동으로 전환할 수 있는지 확인",
                required_evidence_type="문제·판단·행동·검증 결과가 연결된 실제 경험",
                required_company_connection=f"{app.company} {app.job}의 첫 실행 행동과 KPI",
                likely_objections=["프로젝트 경험과 실제 현장의 차이는 무엇인가?"],
                forbidden_generic_claims=["열정으로 기여하겠습니다", "최선을 다하겠습니다"],
                character_budget=question.character_limit,
                character_limit=question.character_limit,
                character_hard_min=math.ceil(question.character_limit * 0.95),
                character_target_min=math.ceil(question.character_limit * 0.97),
                character_target_max=math.floor(question.character_limit * 0.98),
            )
            for question in app.questions
        ]
        return {"question_contracts": contracts, "team_decisions": [decision]}

    async def evidence(state: ResumeGraphState) -> dict[str, Any]:
        app = state["application"]
        evidence_ids = [item.event_id for item in app.evidence]
        decision, _ = await run_team(
            TEAMS["evidence"],
            backend,
            {
                "company": app.company,
                "job": app.job,
                "evidence_ids": evidence_ids,
                "contracts": [item.model_dump() for item in state["question_contracts"]],
            },
            state["mode"],
        )
        selected_by_question = [
            _select_evidence(question.question_id, app.evidence) for question in app.questions
        ]
        contracts = [
            TransferContract(
                question_id=question.question_id,
                evidence_event_id=selected.event_id,
                past_problem=selected.problem,
                reusable_judgment=selected.judgment,
                reusable_action=selected.actions[0],
                company_task=state["demand_brief"].responsibilities[0],
                first_action="설비 이력과 측정 데이터를 연결해 점검 우선순위를 제안한다.",
                output_or_kpi=state["demand_brief"].kpis[0],
                boundary="검증된 분석 경험을 적용하되 실제 설비 기준과 현장 절차를 먼저 학습한다.",
            )
            for question, selected in zip(app.questions, selected_by_question, strict=True)
        ]
        return {"transfer_contracts": contracts, "team_decisions": [decision]}

    async def branding(state: ResumeGraphState) -> dict[str, Any]:
        app = state["application"]
        selected_ids = list(
            dict.fromkeys(contract.evidence_event_id for contract in state["transfer_contracts"])
        )
        selected_by_id = {item.event_id: item for item in app.evidence}
        selected = selected_by_id[selected_ids[0]]
        decision, _ = await run_team(
            TEAMS["branding"],
            backend,
            {
                "company": app.company,
                "job": app.job,
                "evidence_ids": selected_ids,
                "transfer_contracts": [item.model_dump() for item in state["transfer_contracts"]],
            },
            state["mode"],
        )
        brief = PositioningBrief(
            hire_me_reason=(
                f"{selected.judgment}을 바탕으로 데이터를 현장 점검 우선순위로 전환할 수 있다."
            ),
            primary_differentiator="분석 결과를 검증 가능한 현장 행동으로 연결하는 역량",
            supporting_differentiators=selected.capabilities[:2],
            proof_events=selected_ids,
            company_application=state["transfer_contracts"][0].first_action,
            avoid_language=["근거 없는 전문가 표현", "취득하지 않은 자격의 보유 주장"],
            anticipated_objections=["PoC 경험과 실제 생산설비의 차이"],
        )
        return {"positioning_brief": brief, "team_decisions": [decision]}

    async def draft(state: ResumeGraphState) -> dict[str, Any]:
        app = state["application"]
        evidence_by_id = {item.event_id: item for item in app.evidence}

        async def draft_question(question_index: int) -> tuple[DraftAnswer, list[TeamDecision]]:
            question = app.questions[question_index]
            transfer = state["transfer_contracts"][question_index]
            selected = evidence_by_id[transfer.evidence_event_id]
            brief = {
                "company": app.company,
                "job": app.job,
                "question_id": question.question_id,
                "question": question.text,
                "positioning": state["positioning_brief"].model_dump(),
                "transfer": transfer.model_dump(),
                "evidence_ids": [selected.event_id],
            }
            results = await asyncio.gather(
                *(run_team(team, backend, brief, state["mode"]) for team in BLOCK_TEAMS.values())
            )
            decisions = [item[0] for item in results]
            answer = _compose_grounded_answer(
                question.question_id,
                app,
                selected,
                transfer,
                question.character_limit,
            )
            return answer, decisions

        drafted = await asyncio.gather(*(draft_question(i) for i in range(len(app.questions))))
        return {
            "answers": [item[0] for item in drafted],
            "team_decisions": [decision for item in drafted for decision in item[1]],
        }

    async def integrate(state: ResumeGraphState) -> dict[str, Any]:
        app = state["application"]
        decision, _ = await run_team(
            TeamDefinitionIntegration,
            backend,
            {
                "company": app.company,
                "job": app.job,
                "evidence_ids": [item.event_id for item in app.evidence],
                "answers": [answer.model_dump() for answer in state["answers"]],
            },
            state["mode"],
        )
        return {"team_decisions": [decision]}

    async def character_budget_gate(state: ResumeGraphState) -> dict[str, Any]:
        app = state["application"]
        evidence_by_id = {item.event_id: item for item in app.evidence}
        transfer_by_question = {
            item.question_id: item for item in state["transfer_contracts"]
        }
        rewritten: list[DraftAnswer] = []
        attempted: list[str] = []
        for answer in state["answers"]:
            transfer = transfer_by_question[answer.question_id]
            evidence = evidence_by_id[transfer.evidence_event_id]
            updated, did_attempt = rewrite_to_character_target(
                answer, app, evidence, transfer
            )
            rewritten.append(updated)
            if did_attempt:
                attempted.append(answer.question_id)
                backend.record_local_operation(
                    role="character_budget_rewriter",
                    team="integration",
                    question_id=answer.question_id,
                    call_kind=CallKind.CHARACTER_REWRITE,
                )
        return {
            "answers": rewritten,
            "character_rewrite_attempts": attempted,
        }

    async def validate(state: ResumeGraphState) -> dict[str, Any]:
        app = state["application"]
        qa_brief = {
            "company": app.company,
            "job": app.job,
            "evidence_ids": [item.event_id for item in app.evidence],
            "answers": [answer.model_dump() for answer in state["answers"]],
        }
        qa_results = await asyncio.gather(
            *(run_team(team, backend, qa_brief, state["mode"]) for team in QA_TEAMS.values())
        )
        return {
            "validation": validate_answers(app, state["answers"]),
            "team_decisions": [item[0] for item in qa_results],
        }

    builder = StateGraph(ResumeGraphState)
    builder.add_node("company_job_intelligence", intelligence)
    builder.add_node("question_strategy", question_strategy)
    builder.add_node("evidence_matching", evidence)
    builder.add_node("self_branding", branding)
    builder.add_node("parallel_block_factory", draft)
    builder.add_node("integration", integrate)
    builder.add_node("character_budget_gate", character_budget_gate)
    builder.add_node("qa_council", validate)
    builder.add_edge(START, "company_job_intelligence")
    builder.add_edge("company_job_intelligence", "question_strategy")
    builder.add_edge("question_strategy", "evidence_matching")
    builder.add_edge("evidence_matching", "self_branding")
    builder.add_edge("self_branding", "parallel_block_factory")
    builder.add_edge("parallel_block_factory", "integration")
    builder.add_edge("integration", "character_budget_gate")
    builder.add_edge("character_budget_gate", "qa_council")
    builder.add_edge("qa_council", END)
    return builder.compile()


from .agents import AgentSpec  # noqa: E402
from .schemas import ModelTier  # noqa: E402
from .teams import TeamDefinition  # noqa: E402

TeamDefinitionIntegration = TeamDefinition(
    "integration",
    (
        AgentSpec("narrative_editor", ModelTier.TERRA, "서사 연결"),
        AgentSpec("repetition_hunter", ModelTier.LUNA, "의미 반복 제거"),
        AgentSpec("character_budget_optimizer", ModelTier.LUNA, "글자 예산"),
        AgentSpec("voice_consistency_agent", ModelTier.LUNA, "목소리 일관성"),
    ),
    "integration_lead",
)


async def run_resume_graph(
    application: ApplicationInput,
    backend: AgentBackend,
    mode: ExecutionMode,
    run_id: str | None = None,
) -> RunResult:
    resolved_run_id = run_id or uuid.uuid4().hex[:12]
    backend.begin_run(resolved_run_id)
    graph = build_graph(backend)
    state = await graph.ainvoke(
        {
            "application": application,
            "mode": mode,
            "team_decisions": [],
        }
    )
    calls = sorted(backend.calls, key=lambda call: call.sequence)
    known_cost = sum(call.estimated_cost_usd or 0 for call in calls)
    telemetry = RunTelemetry(
        run_id=resolved_run_id,
        mode=mode,
        model_calls=calls,
        total_cost_usd=round(known_cost, 6),
        cost_complete=all(call.estimated_cost_usd is not None for call in calls),
        price_catalog_version=backend.price_catalog_version,
        prompt_versions={"core": "v0.1.0"},
        metadata={"character_rewrite_attempts": state.get("character_rewrite_attempts", [])},
    )
    validation = state["validation"]
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


def _compose_grounded_answer(
    question_id: str,
    app: ApplicationInput,
    evidence: EvidencePacket,
    transfer: TransferContract,
    character_limit: int,
) -> DraftAnswer:
    sentences = [
        (
            SentenceRole.ANSWER,
            f"이 경험에서 증명한 판단 방식을 {app.company} {app.job}에 적용하겠습니다.",
            "질문에 대한 직접 답변",
            None,
        ),
        (
            SentenceRole.COMPANY_NEED,
            "직무에는 예방보전 판단의 일관성이 필요합니다.",
            "회사의 직무 수요",
            None,
        ),
        (
            SentenceRole.PERSPECTIVE,
            "분석값보다 확인 가능한 근거를 중시합니다.",
            "지원자의 판단 기준",
            evidence.event_id,
        ),
        (
            SentenceRole.PROBLEM,
            evidence.problem,
            "해결 대상 문제",
            evidence.event_id,
        ),
        (
            SentenceRole.JUDGMENT,
            evidence.judgment,
            "문제 해결 판단",
            evidence.event_id,
        ),
        (
            SentenceRole.ACTION,
            evidence.actions[0],
            "실제 수행 행동",
            evidence.event_id,
        ),
        (
            SentenceRole.RESULT,
            evidence.results[0],
            "검증된 결과",
            evidence.event_id,
        ),
        (
            SentenceRole.DIFFERENTIATION,
            "결과를 점검 순서와 기록으로 연결했습니다.",
            "분석을 행동으로 바꾸는 차별점",
            evidence.event_id,
        ),
        (
            SentenceRole.TRANSFER,
            transfer.first_action,
            "입사 후 첫 적용 행동",
            evidence.event_id,
        ),
        (
            SentenceRole.VALIDATION,
            f"성과는 {transfer.output_or_kpi}로 확인하겠습니다.",
            "성과 확인 기준",
            evidence.event_id,
        ),
    ]
    plans = [
        SentencePlan(
            sentence_id=f"{question_id}-S{index:02d}",
            text=text,
            role=role,
            selling_point=selling_point,
            evidence_event_id=evidence_id,
            claim_ids=[f"{question_id}-C{index:02d}"],
            company_connection=(
                app.job if role in {SentenceRole.ANSWER, SentenceRole.TRANSFER} else None
            ),
            interview_defensible=evidence_id is not None,
        )
        for index, (role, text, selling_point, evidence_id) in enumerate(sentences, start=1)
    ]
    return DraftAnswer(
        question_id=question_id,
        headline=f"[{evidence.title}]",
        body=" ".join(item.text for item in plans),
        sentence_plans=plans,
        evidence_ids=[evidence.event_id],
        character_limit=character_limit,
    )


def _first_clause(text: str) -> str:
    stripped = text.strip()
    for delimiter in ("\n", ".", "다."):
        if delimiter in stripped:
            return stripped.split(delimiter, 1)[0].strip() or stripped
    return stripped


def _select_evidence(question_id: str, evidence: list[EvidencePacket]) -> EvidencePacket:
    """Prefer an explicit question-to-evidence contract, with stable fallback ordering."""
    return next(
        (item for item in evidence if question_id in item.best_for_questions),
        evidence[0],
    )
