from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict, cast

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from .agents import AgentBackend
from .character_budget import rewrite_to_character_target
from .editorial_quality import (
    apply_edit_operations,
    assessment_from_proposal,
    operations_from_proposal,
)
from .material_planning import allocate_materials, preflight_material_capacity
from .question_planning import build_question_contracts, ground_contract
from .schemas import (
    ApplicationInput,
    BackendProvider,
    CallKind,
    CostStatus,
    DemandBrief,
    DraftAnswer,
    DraftProposal,
    EditorialAssessment,
    EvidencePacket,
    ExecutionMode,
    MaterialPortfolioPlan,
    ModelTier,
    PositioningBrief,
    PrepSoaraStructure,
    QuestionNarrativeContract,
    RunPhaseSpan,
    RunResult,
    RunTelemetry,
    TeamDecision,
    TransferContract,
    ValidationIssue,
    ValidationReport,
)
from .teams import TEAMS, WRITING_COUNCIL, run_team
from .validators import validate_answers

GRAPH_VERSION = "v0.7"
CHARACTER_REWRITE_MAX_ROUNDS = 3
SEMANTIC_REPAIR_MAX_ROUNDS = 2
HARD_CALL_CAP = 30


def _append(left: list[Any], right: list[Any]) -> list[Any]:
    return left + right


class ResumeGraphState(TypedDict, total=False):
    application: ApplicationInput
    mode: ExecutionMode
    demand_brief: DemandBrief
    question_contracts: list[QuestionNarrativeContract]
    material_plan: MaterialPortfolioPlan
    transfer_contracts: list[TransferContract]
    positioning_brief: PositioningBrief
    answers: list[DraftAnswer]
    character_rewrite_attempts: list[str]
    character_rewrite_round_count: int
    editorial_assessments: list[EditorialAssessment]
    editorial_repair_history: list[dict[str, Any]]
    strategy_fallback_used: bool
    team_decisions: Annotated[list[TeamDecision], _append]
    validation: Any


def build_graph(
    backend: AgentBackend,
    checkpointer: Any | None = None,
    phase_spans: list[RunPhaseSpan] | None = None,
) -> Any:
    spans = phase_spans if phase_spans is not None else []

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
                    "해당 회사·직무의 문제, 책임, KPI, 실패 위험, 행동 역량을 "
                    "입력 자료의 근거 범위 안에서 제안"
                ),
            },
            state["mode"],
        )
        raw_demand = (
            decision.structured_payload.demand_brief if decision.structured_payload else None
        )
        if raw_demand:
            try:
                demand = DemandBrief.model_validate(raw_demand)
            except ValueError:
                demand = None
        else:
            demand = None
        return {
            "demand_brief": demand
            or DemandBrief(
                company_problems=[_first_clause(app.company_context)],
                responsibilities=[_first_clause(app.job_description)],
                kpis=[
                    f"{app.job} 업무 흐름의 검증 가능성",
                    "변경·운영 이력의 추적 가능성",
                ],
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
                "output_contract": (
                    "모든 문항의 직접 답, 채용 의도, 요구 증거, 예상 반론을 함께 설계"
                ),
            },
            state["mode"],
        )
        contracts, fallback_used = build_question_contracts(app, decision)
        return {
            "question_contracts": contracts,
            "strategy_fallback_used": fallback_used,
            "team_decisions": [decision],
        }

    async def evidence_branding(state: ResumeGraphState) -> dict[str, Any]:
        app = state["application"]
        material_plan = allocate_materials(app, state["question_contracts"])
        if material_plan.status != "ready":
            raise ValueError(material_plan.code or "INSUFFICIENT_DISTINCT_EVIDENCE")
        evidence_by_id = {item.event_id: item for item in app.evidence}
        selected = [evidence_by_id[item.primary_event_id] for item in material_plan.assignments]
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
        grounded_contracts = [
            ground_contract(contract, evidence)
            for contract, evidence in zip(state["question_contracts"], selected, strict=True)
        ]
        transfers = [
            TransferContract(
                question_id=question.question_id,
                evidence_event_id=evidence.event_id,
                past_problem=evidence.problem,
                reusable_judgment=evidence.judgment,
                reusable_action=evidence.actions[0],
                company_task=_first_clause(app.job_description),
                first_action=_first_action(question.question_id, app),
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
                "운영 실패와 재작업을 줄이는 판단 방식"
            ),
            primary_differentiator=(
                "시험 조건과 사용자 흐름을 끝까지 추적해 팀이 같은 기준으로 검증하게 하는 실행력"
            ),
            supporting_differentiators=capabilities[:3] or ["근거 기반 판단"],
            proof_events=[item.event_id for item in selected],
            company_application=transfers[0].first_action,
            avoid_language=list(
                dict.fromkeys(boundary for item in selected for boundary in item.boundaries)
            ),
            anticipated_objections=[
                "프로젝트 경험을 실제 현업 성과로 오해할 수 있음",
                "검증하지 않은 수치나 적용 범위를 묻는 꼬리질문",
            ],
        )
        return {
            "question_contracts": grounded_contracts,
            "material_plan": material_plan,
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
                "character_bounds": {
                    contract.question_id: {
                        "hard_min": contract.character_hard_min,
                        "target_min": contract.character_target_min,
                        "target_max": contract.character_target_max,
                        "hard_max": contract.character_limit,
                    }
                    for contract in state["question_contracts"]
                },
                "evidence_ids": [item.event_id for item in app.evidence],
                "output_contract": (
                    "drafts 배열에 모든 문항을 반환. 각 본문에는 해당 회사명을 정확히 한 번 이상 "
                    "포함한다. 사실·sentence_id·evidence_id를 "
                    "보존하며 반복과 목소리만 통합 편집. 최초 맥락 뒤의 반복 방어 문장, "
                    "독자 가치 없는 미측정 설명, 추상적인 좋은 결과를 제거한다. 판단 문장은 "
                    "구체 기술 행동과 검증 결과로 이어지게 하되 새 사실은 추가하지 않는다. "
                    "각 문항은 character_bounds의 hard_min~hard_max를 반드시 지킨다"
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
        rewritten = list(state["answers"])
        attempted: list[str] = []
        rewrite_round_count = 0

        for round_number in range(1, CHARACTER_REWRITE_MAX_ROUNDS + 1):
            failed: list[
                tuple[
                    int,
                    DraftAnswer,
                    QuestionNarrativeContract,
                    TransferContract,
                    EvidencePacket,
                ]
            ] = []
            for index, answer in enumerate(rewritten):
                contract = contracts[answer.question_id]
                hard_min = contract.character_hard_min or 0
                hard_max = contract.character_limit or contract.character_budget
                if not _outside_hard_gate(answer.character_count, hard_min, hard_max):
                    continue
                if answer.question_id not in attempted:
                    attempted.append(answer.question_id)
                transfer = transfer_by_question[answer.question_id]
                evidence = evidence_by_id[transfer.evidence_event_id]
                failed.append((index, answer, contract, transfer, evidence))

            if not failed:
                break

            rewrite_round_count = round_number
            if backend.provider is BackendProvider.LOCAL:
                for index, answer, _, transfer, evidence in failed:
                    updated, _ = rewrite_to_character_target(answer, app, evidence, transfer)
                    rewritten[index] = updated
                backend.record_local_operation(
                    role="character_budget_batch_rewriter",
                    team="integration",
                    question_id=None,
                    call_kind=CallKind.CHARACTER_REWRITE,
                )
                continue

            proposal = await backend.propose(
                role="character_budget_batch_rewriter",
                team="integration",
                brief={
                    "company": app.company,
                    "job": app.job,
                    "rewrite_round": round_number,
                    "max_rewrite_rounds": CHARACTER_REWRITE_MAX_ROUNDS,
                    "repairs": [
                        {
                            **_writing_brief(
                                app,
                                _question_dump(app, answer.question_id),
                                contract,
                                transfer,
                                evidence,
                                state["positioning_brief"],
                            ),
                            "current_draft": _proposal_from_answer(answer).model_dump(),
                            "current_character_count": answer.character_count,
                            "hard_min": contract.character_hard_min,
                            "hard_max": contract.character_limit,
                        }
                        for _, answer, contract, transfer, evidence in failed
                    ],
                    "evidence_ids": list(
                        dict.fromkeys(
                            event_id
                            for _, answer, _, _, _ in failed
                            for event_id in answer.evidence_ids
                        )
                    ),
                    "output_contract": (
                        "repairs의 각 문항만 drafts 배열에 같은 순서로 반환. "
                        "current_character_count와 hard_min~hard_max를 기준으로 정확히 조절하고, "
                        "해당 회사명을 본문에 정확히 한 번 이상 보존하며, 새 사실·수치 금지, "
                        "sentence plan 계보 유지"
                    ),
                },
                tier=ModelTier.TERRA,
                question_id=None,
                call_kind=CallKind.CHARACTER_REWRITE,
            )
            if len(proposal.drafts) != len(failed):
                continue
            for draft, (index, answer, _, _, _) in zip(proposal.drafts, failed, strict=True):
                rewritten[index] = _answer_from_proposal(draft, answer.character_limit)

        return {
            "answers": rewritten,
            "character_rewrite_attempts": attempted,
            "character_rewrite_round_count": rewrite_round_count,
        }

    async def editorial_quality_gate(state: ResumeGraphState) -> dict[str, Any]:
        app = state["application"]
        contracts = {item.question_id: item for item in state["question_contracts"]}
        evidence_by_id = {item.event_id: item for item in app.evidence}
        answers = list(state["answers"])
        history: list[dict[str, Any]] = []

        def budget_assessment(answer: DraftAnswer) -> EditorialAssessment:
            return EditorialAssessment(
                question_id=answer.question_id,
                inferred_takeaway="호출 예산 소진으로 평가하지 못함",
                question_directness=1,
                thesis_clarity=1,
                logical_continuity=1,
                evidence_to_claim=1,
                effort_or_action_specificity=1,
                company_transfer=1,
                missing_information=["CALL_BUDGET_EXHAUSTED"],
                verdict="repair",
            )

        async def assess(answer: DraftAnswer) -> EditorialAssessment:
            contract = contracts[answer.question_id]
            if _online_call_count(backend) >= HARD_CALL_CAP:
                return budget_assessment(answer)
            proposal = await backend.propose(
                role="human_reader_critic",
                team="editorial_quality",
                brief={
                    "company": app.company,
                    "job": app.job,
                    "question_id": answer.question_id,
                    "question": _question_dump(app, answer.question_id)["text"],
                    "narrative_contract": contract.model_dump(),
                    "submission_text": answer.submission_text,
                    "sentence_plans": [item.model_dump() for item in answer.sentence_plans],
                    "allowed_evidence": [
                        evidence_by_id[item].model_dump()
                        for item in answer.evidence_ids
                        if item in evidence_by_id
                    ],
                    "evidence_ids": answer.evidence_ids,
                    "output_contract": "독립 독자 관점의 EditorialAssessment만 구조화해 반환",
                },
                tier=ModelTier.LUNA,
                question_id=answer.question_id,
                call_kind=CallKind.SEMANTIC_CRITIC,
            )
            return assessment_from_proposal(proposal, answer, contract)

        initial_slots = max(0, HARD_CALL_CAP - _online_call_count(backend))
        assessed = list(
            await asyncio.gather(*(assess(answer) for answer in answers[:initial_slots]))
        )
        assessments = assessed + [budget_assessment(item) for item in answers[initial_slots:]]
        for round_number in range(1, SEMANTIC_REPAIR_MAX_ROUNDS + 1):
            targets = [
                (index, answer, assessments[index])
                for index, answer in enumerate(answers)
                if not assessments[index].passed and assessments[index].verdict != "replan"
            ]
            if not targets or _online_call_count(backend) >= HARD_CALL_CAP:
                break
            available_pairs = max(0, (HARD_CALL_CAP - _online_call_count(backend)) // 2)
            targets = sorted(
                targets,
                key=lambda item: (
                    min(
                        item[2].evidence_to_claim,
                        item[2].question_directness,
                        item[2].thesis_clarity,
                        item[2].logical_continuity,
                    ),
                    item[1].question_id,
                ),
            )[:available_pairs]
            if not targets:
                break

            async def repair_one(
                index: int,
                answer: DraftAnswer,
                assessment: EditorialAssessment,
                repair_round: int = round_number,
            ) -> tuple[int, DraftAnswer, dict[str, Any]]:
                if _online_call_count(backend) >= HARD_CALL_CAP:
                    return index, answer, {"accepted": False, "reason": "call_budget"}
                proposal = await backend.propose(
                    role="sentence_scoped_editor",
                    team="editorial_quality",
                    brief={
                        "company": app.company,
                        "job": app.job,
                        "question_id": answer.question_id,
                        "narrative_contract": contracts[answer.question_id].model_dump(),
                        "current_draft": _proposal_from_answer(answer).model_dump(),
                        "editorial_assessment": assessment.model_dump(),
                        "allowed_evidence": [
                            evidence_by_id[item].model_dump()
                            for item in answer.evidence_ids
                            if item in evidence_by_id
                        ],
                        "evidence_ids": answer.evidence_ids,
                        "character_count": answer.character_count,
                        "character_hard_min": contracts[answer.question_id].character_hard_min,
                        "character_limit": contracts[answer.question_id].character_limit,
                        "output_contract": "지적된 sentence_id에 대한 edit_operations만 반환",
                    },
                    tier=ModelTier.TERRA,
                    question_id=answer.question_id,
                    call_kind=CallKind.SEMANTIC_REPAIR,
                )
                operations = operations_from_proposal(proposal)
                candidate = apply_edit_operations(answer, operations) if operations else answer
                report = validate_answers(app, [candidate])
                fact_codes = {
                    "UNKNOWN_EVIDENCE",
                    "UNVERIFIED_NUMBER",
                    "FORBIDDEN_COMBINATION",
                    "NOT_INTERVIEW_DEFENSIBLE",
                }
                introduced_fact_error = any(
                    issue.severity == "hard_fail" and issue.code in fact_codes
                    for issue in report.issues
                )
                introduced_character_error = _outside_hard_gate(
                    candidate.character_count,
                    contracts[answer.question_id].character_hard_min or 0,
                    contracts[answer.question_id].character_limit
                    or contracts[answer.question_id].character_budget,
                )
                rejected = introduced_fact_error or introduced_character_error
                return (
                    index,
                    (answer if rejected else candidate),
                    {
                        "question_id": answer.question_id,
                        "round": repair_round,
                        "accepted": bool(operations) and not rejected,
                        "operations": [item.model_dump() for item in operations],
                        "reason": (
                            "fact_boundary"
                            if introduced_fact_error
                            else "character_boundary"
                            if introduced_character_error
                            else None
                        ),
                    },
                )

            repaired = await asyncio.gather(*(repair_one(*target) for target in targets))
            changed_indexes = []
            for index, answer, item_history in repaired:
                if item_history["accepted"]:
                    answers[index] = answer
                    changed_indexes.append(index)
                history.append(item_history)
            if not changed_indexes:
                break
            refreshed = await asyncio.gather(*(assess(answers[index]) for index in changed_indexes))
            for index, assessment in zip(changed_indexes, refreshed, strict=True):
                assessments[index] = assessment

        return {
            "answers": answers,
            "editorial_assessments": assessments,
            "editorial_repair_history": history,
        }

    def validate(state: ResumeGraphState) -> dict[str, Any]:
        return {"validation": validate_answers(state["application"], state["answers"])}

    def timed(name: str, node: Any) -> Any:
        async def wrapped(state: ResumeGraphState) -> dict[str, Any]:
            started_at = datetime.now(UTC)
            started = time.perf_counter()
            status: Literal["completed", "failed"] = "completed"
            try:
                result = node(state)
                resolved = await result if inspect.isawaitable(result) else result
                return cast(dict[str, Any], resolved)
            except Exception:
                status = "failed"
                raise
            finally:
                completed_at = datetime.now(UTC)
                spans.append(
                    RunPhaseSpan(
                        phase=name,
                        started_at=started_at,
                        completed_at=completed_at,
                        duration_ms=int((time.perf_counter() - started) * 1000),
                        status=status,
                    )
                )

        return wrapped

    builder = StateGraph(ResumeGraphState)
    for name, node in (
        ("company_job_intelligence", intelligence),
        ("question_strategy", question_strategy),
        ("evidence_branding", evidence_branding),
        ("writing_councils", writing_councils),
        ("integration", integration),
        ("character_budget_gate", character_budget_gate),
        ("editorial_quality_gate", editorial_quality_gate),
        ("deterministic_qa", validate),
    ):
        builder.add_node(name, timed(name, node))
    common = ["company_job_intelligence", "question_strategy"]
    for common_node in common:
        builder.add_edge(START, common_node)
    builder.add_edge(common, "evidence_branding")
    builder.add_edge("evidence_branding", "writing_councils")
    builder.add_edge("writing_councils", "integration")
    builder.add_edge("integration", "character_budget_gate")
    builder.add_edge("character_budget_gate", "editorial_quality_gate")
    builder.add_edge("editorial_quality_gate", "deterministic_qa")
    builder.add_edge("deterministic_qa", END)
    return builder.compile(checkpointer=checkpointer)


async def run_resume_graph(
    application: ApplicationInput,
    backend: AgentBackend,
    mode: ExecutionMode,
    run_id: str | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> RunResult:
    run_started_at = datetime.now(UTC)
    run_started = time.perf_counter()
    phase_spans: list[RunPhaseSpan] = []
    resolved_run_id = run_id or uuid.uuid4().hex[:12]
    backend.begin_run(resolved_run_id)
    blocked_plan = None if resume else preflight_material_capacity(application)
    if blocked_plan is not None:
        contracts, fallback_used = build_question_contracts(application)
        completed_at = datetime.now(UTC)
        return RunResult(
            run_id=resolved_run_id,
            status="blocked_insufficient_evidence",
            input_summary={"company": application.company, "job": application.job},
            demand_brief=DemandBrief(
                company_problems=[_first_clause(application.company_context)],
                responsibilities=[_first_clause(application.job_description)],
                kpis=[],
                failure_risks=[],
                behavioral_competencies=[],
                source_refs=["application.company_context", "application.job_description"],
            ),
            question_contracts=contracts,
            transfer_contracts=[],
            positioning_brief=PositioningBrief(
                hire_me_reason="",
                primary_differentiator="",
                supporting_differentiators=[],
                proof_events=[],
                company_application="",
                avoid_language=[],
                anticipated_objections=[],
            ),
            answers=[],
            validation=ValidationReport(
                passed=False,
                issues=[
                    ValidationIssue(
                        code="INSUFFICIENT_DISTINCT_EVIDENCE",
                        severity="hard_fail",
                        message="문항 수만큼 서로 다른 검증 소재를 확보하지 못했습니다.",
                    )
                ],
                metrics={"model_call_count": 0},
            ),
            team_decisions=[],
            eligibility_warnings=application.eligibility_notes,
            telemetry=RunTelemetry(
                run_id=resolved_run_id,
                mode=mode,
                model_calls=[],
                total_cost_usd=0,
                cost_complete=True,
                price_catalog_version=backend.price_catalog_version,
                prompt_versions={"core": "v0.7-material-editorial"},
                metadata={"strategy_fallback_used": fallback_used},
                provider=backend.provider,
                billing_mode=backend.billing_mode,
                cost_status=backend.cost_status,
                graph_version=GRAPH_VERSION,
                graph_nodes_completed=["material_preflight"],
                base_call_budget=10 + 4 * len(application.questions),
                optional_calls_used=0,
                hard_call_cap=HARD_CALL_CAP,
                command_started_at=run_started_at,
                completed_at=completed_at,
                wall_time_ms=int((time.perf_counter() - run_started) * 1000),
            ),
            material_plan=blocked_plan,
        )
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
        state = await build_graph(backend, phase_spans=phase_spans).ainvoke(initial, config)
    else:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            state = await build_graph(backend, saver, phase_spans).ainvoke(initial, config)
    calls = sorted(backend.calls, key=lambda call: call.sequence)
    model_calls = [call for call in calls if call.provider is not BackendProvider.LOCAL]
    optional_calls = sum(
        call.call_kind
        in {
            CallKind.CRITIC,
            CallKind.ADJUDICATOR,
            CallKind.CHARACTER_REWRITE,
            CallKind.SEMANTIC_REPAIR,
        }
        for call in model_calls
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
        prompt_versions={"core": "v0.7-material-editorial"},
        metadata={
            "character_rewrite_attempts": state.get("character_rewrite_attempts", []),
            "character_rewrite_round_count": state.get("character_rewrite_round_count", 0),
            "strategy_fallback_used": state.get("strategy_fallback_used", False),
            "editorial_repair_history": state.get("editorial_repair_history", []),
        },
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
            "editorial_quality_gate",
            "deterministic_qa",
        ],
        base_call_budget=10 + 4 * question_count,
        optional_calls_used=optional_calls,
        hard_call_cap=HARD_CALL_CAP,
        command_started_at=run_started_at,
        completed_at=datetime.now(UTC),
        wall_time_ms=int((time.perf_counter() - run_started) * 1000),
        phase_spans=phase_spans,
        network_status=(
            "unknown"
            if not model_calls
            else (
                "degraded"
                if any(not call.success or (call.retry_count or 0) for call in model_calls)
                else "healthy"
            )
        ),
    )
    return RunResult(
        run_id=resolved_run_id,
        status=(
            "validated"
            if validation.passed
            and all(item.passed for item in state.get("editorial_assessments", []))
            else "needs_review"
        ),
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
        material_plan=state.get("material_plan"),
        editorial_assessments=state.get("editorial_assessments", []),
    )


def _writing_brief(
    app: ApplicationInput,
    question: dict[str, Any],
    contract: QuestionNarrativeContract,
    transfer: TransferContract,
    evidence: EvidencePacket,
    positioning: PositioningBrief,
) -> dict[str, Any]:
    supporting_ids = [str(item) for item in question.get("supporting_evidence_ids", [])]
    supporting_evidence = [
        item.model_dump() for item in app.evidence if item.event_id in supporting_ids
    ]
    learning_transfer = bool(supporting_evidence) and _is_learning_question(str(question["text"]))
    return {
        "company": app.company,
        "job": app.job,
        "question_id": question["question_id"],
        "question": question["text"],
        "writing_contract": contract.model_dump(),
        "positioning": positioning.model_dump(),
        "transfer": transfer.model_dump(),
        "evidence": evidence.model_dump(),
        "supporting_evidence": supporting_evidence,
        "evidence_ids": [evidence.event_id, *supporting_ids],
        "learning_transfer_required": learning_transfer,
        "rules": [
            "모든 문장은 독립적인 판매 가치와 역할을 가짐",
            "수치는 verified numeric_authorities만 사용",
            "실제 양산 실적으로 확대하지 않음",
            (
                "사실 범위는 최초 맥락에서 교육용·가상·프로젝트 등 긍정형 표현으로 한 번만 "
                "밝히고 뒤에서 반복하지 않음"
            ),
            (
                "경계·미측정 항목은 interview_defense에 보존하되, 독자에게 도움이 없는 "
                "'일 뿐', '보장한 값이 아니다', '실제 경험이 아니다' 같은 보험 문장은 본문 금지"
            ),
            (
                "판단을 설명한 뒤 반드시 구체적인 기술 구조·입출력·제어 조건·검증 행동 중 "
                "하나와 관찰 가능한 결과를 연결하며 '좋은 결과'처럼 뭉뚱그리지 않음"
            ),
            (
                "질문의 섹션명이나 영문 라벨에 맞추지 말고, 질문이 요구하는 노력·성공/실패·"
                "과정·배움을 직접 답함"
            ),
            (
                "learning_transfer_required=true이면 supporting_evidence를 1~2문장만 사용해 "
                "배운 방식을 후행 프로젝트에 적용한 행동과 구체 결과를 증명하되, 다른 문항의 "
                "기술 설명을 반복하지 않음"
            ),
            "소제목·줄바꿈·공백 포함 Python len 기준 목표 구간 준수",
            (
                f"질문 유형은 {contract.archetype.value}이며 다음 서사 순서를 따른다: "
                + " → ".join(contract.narrative_sequence)
            ),
            "역할을 채우기 위한 별도 문장을 만들지 말고 한 문장마다 새로운 정보나 판단을 제공",
            f"독자가 기억할 단일 핵심 메시지: {contract.core_message}",
            "근거 문장은 interview_defensible=true, 회사 전이 문장은 company_connection 명시",
            f"본문의 구체적인 직무 전이 문장에 회사명 '{app.company}'을 정확히 한 번 이상 포함",
        ],
        "output_contract": (
            "draft 필수. sentence_plans의 text를 순서대로 합치면 본문이 되며 "
            "각 문장은 evidence 또는 company_connection을 가짐"
        ),
    }


def _is_learning_question(text: str) -> bool:
    normalized = text.lower()
    return ("배웠" in text or "배운" in text or "learn" in normalized) and any(
        token in text for token in ("경험", "성공", "실패", "노력")
    )


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


def _outside_hard_gate(character_count: int, hard_min: int, hard_max: int) -> bool:
    return character_count < hard_min or character_count > hard_max


def _online_call_count(backend: AgentBackend) -> int:
    return sum(call.provider is not BackendProvider.LOCAL for call in backend.calls)


def _first_clause(text: str) -> str:
    stripped = text.strip()
    for delimiter in ("\n", ".", "다."):
        if delimiter in stripped:
            return stripped.split(delimiter, 1)[0].strip() or stripped
    return stripped


def _select_evidence(question_id: str, evidence: list[EvidencePacket]) -> EvidencePacket:
    return next((item for item in evidence if question_id in item.best_for_questions), evidence[0])


def _first_action(question_id: str, app: ApplicationInput) -> str:
    company = app.company
    job = app.job
    return {
        "Q1": (
            f"{company} {job}의 핵심 입력 데이터와 업무 흐름을 현업과 확인하고 "
            "대안을 같은 검증 기준으로 비교하겠습니다."
        ),
        "Q2": (
            f"{company} {job}에서 입력·처리·출력의 경계를 나누고 "
            "변경 전후 시험과 조치 이력을 연결하겠습니다."
        ),
        "Q3": (
            f"{company} {job}의 현업 사용 순서와 개발 변경 범위를 먼저 합의하고 "
            "동일 입력의 판정 유지 여부를 인수 기준으로 삼겠습니다."
        ),
    }.get(
        question_id,
        f"{company} {job}의 현업 기준과 운영 이력을 먼저 확인하겠습니다.",
    )


def _output_kpi(question_id: str) -> str:
    return {
        "Q1": "대안별 영향과 검증 기준의 비교 가능성",
        "Q2": "변경 전후 시험의 재현성과 장애 재발 방지",
        "Q3": "요구사항·변경·시험 결과의 추적성",
    }.get(question_id, "검증 가능한 직무 산출물")
