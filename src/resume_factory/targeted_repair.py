from __future__ import annotations

import math
from typing import Any

from .agents import AgentBackend
from .graph import _answer_from_proposal, _proposal_from_answer, _writing_brief
from .schemas import (
    ApplicationInput,
    CallKind,
    DraftAnswer,
    EvidencePacket,
    ModelTier,
    PositioningBrief,
    QuestionNarrativeContract,
    SentenceRole,
    TransferContract,
)
from .validators import validate_answers

CHARACTER_CODES = {"CHARACTER_UNDERFILL", "CHARACTER_LIMIT"}


def diagnose_answer_gaps(
    application: ApplicationInput,
    answer: DraftAnswer,
    contract: QuestionNarrativeContract,
    evidence: EvidencePacket,
) -> dict[str, Any]:
    roles = [item.role for item in answer.sentence_plans]
    hard_min = contract.character_hard_min or math.ceil(contract.character_budget * 0.95)
    target_min = contract.character_target_min or math.ceil(contract.character_budget * 0.97)
    missing_roles = [
        role.value
        for role in (
            SentenceRole.ANSWER,
            SentenceRole.PROBLEM,
            SentenceRole.JUDGMENT,
            SentenceRole.ACTION,
            SentenceRole.RESULT,
            SentenceRole.TRANSFER,
        )
        if role not in roles
    ]
    judgment = next((i for i, role in enumerate(roles) if role is SentenceRole.JUDGMENT), -1)
    action = next((i for i, role in enumerate(roles) if role is SentenceRole.ACTION), -1)
    result = next(
        (
            i
            for i, role in enumerate(roles)
            if role in {SentenceRole.RESULT, SentenceRole.VALIDATION}
        ),
        -1,
    )
    action_chain_gap = not (0 <= judgment < action < result)
    body = answer.body
    company_vision_gap = not (
        application.company in body
        and any(token in body for token in _context_tokens(application.company_context))
    )
    return {
        "current_character_count": answer.character_count,
        "hard_min": hard_min,
        "target_min": target_min,
        "characters_to_hard_min": max(0, hard_min - answer.character_count),
        "characters_to_target_min": max(0, target_min - answer.character_count),
        "missing_roles": missing_roles,
        "action_chain_gap": action_chain_gap,
        "company_vision_gap": company_vision_gap,
        "repair_priorities": [
            "판단 이유: 초기 가설을 유지하지 않은 비교 근거",
            "구체 행동: 사용한 입력·변환·비교·검증 순서",
            "결과 의미: 수치가 다음 판단을 어떻게 바꿨는지",
            f"회사 적용: {application.company_context}",
        ],
        "allowed_actions": evidence.actions,
        "allowed_results": evidence.results,
    }


async def repair_question(
    application: ApplicationInput,
    answer: DraftAnswer,
    contract: QuestionNarrativeContract,
    transfer: TransferContract,
    positioning: PositioningBrief,
    evidence: EvidencePacket,
    backend: AgentBackend,
    *,
    max_calls: int,
) -> tuple[DraftAnswer, list[dict[str, Any]]]:
    current = answer
    history: list[dict[str, Any]] = []
    question = next(
        item.model_dump()
        for item in application.questions
        if item.question_id == answer.question_id
    )

    for attempt in range(1, max_calls + 1):
        current_report = validate_answers(application, [current])
        current_diagnosis = diagnose_answer_gaps(application, current, contract, evidence)
        target_min = contract.character_target_min or contract.character_budget
        target_max = contract.character_target_max or contract.character_budget
        if (
            not any(issue.severity == "hard_fail" for issue in current_report.issues)
            and target_min <= current.character_count <= target_max
            and not current_diagnosis["company_vision_gap"]
        ):
            break
        diagnosis = current_diagnosis
        brief = _writing_brief(
            application,
            question,
            contract,
            transfer,
            evidence,
            positioning,
        )
        brief.update(
            {
                "repair_mode": "targeted_gap_fill",
                "repair_attempt": attempt,
                "current_draft": _proposal_from_answer(current).model_dump(),
                "diagnosis": diagnosis,
                "current_validation_issues": [
                    issue.model_dump() for issue in current_report.issues
                ],
                "rules": [
                    *brief["rules"],
                    "진단된 부족 구간만 보강하고 이미 유효한 문장은 가능한 한 보존",
                    "논리 연결, 구체 행동, 결과 의미, 회사 적용 중 실제로 부족한 요소만 추가",
                    "분량을 채우기 위한 동의어 반복·수식어·근거 없는 회사 내부 추정 금지",
                ],
                "output_contract": (
                    "draft 하나만 반환. 현재 문장의 계보와 evidence_id를 보존하면서 진단된 "
                    "간극만 보강한다. 제출문은 character_target_min~character_target_max자를 "
                    "목표로 하며 회사명과 제공된 company_context의 고유 사업 맥락을 본문에 "
                    "포함한다. 새 사실과 numeric_authorities 밖의 수치는 금지한다."
                ),
            }
        )
        proposal = await backend.propose(
            role="targeted_gap_rewriter",
            team="targeted_repair",
            brief=brief,
            tier=ModelTier.TERRA,
            question_id=answer.question_id,
            call_kind=CallKind.CHARACTER_REWRITE,
        )
        draft = proposal.draft or (proposal.drafts[0] if proposal.drafts else None)
        if draft is None or draft.question_id != answer.question_id:
            history.append({"attempt": attempt, "accepted": False, "reason": "missing_draft"})
            continue
        candidate = _answer_from_proposal(draft, answer.character_limit)
        candidate_report = validate_answers(application, [candidate])
        accepted = _repair_score(application, candidate, contract) < _repair_score(
            application, current, contract
        )
        history.append(
            {
                "attempt": attempt,
                "accepted": accepted,
                "before": current.character_count,
                "after": candidate.character_count,
                "hard_failures": [
                    issue.code for issue in candidate_report.issues if issue.severity == "hard_fail"
                ],
                "diagnosis": diagnosis,
            }
        )
        if accepted:
            current = candidate

    return current, history


def _repair_score(
    application: ApplicationInput,
    answer: DraftAnswer,
    contract: QuestionNarrativeContract,
) -> tuple[int, int, int, int]:
    report = validate_answers(application, [answer])
    non_character_hard = sum(
        issue.severity == "hard_fail" and issue.code not in CHARACTER_CODES
        for issue in report.issues
    )
    character_hard = sum(
        issue.severity == "hard_fail" and issue.code in CHARACTER_CODES for issue in report.issues
    )
    company_vision_gap = int(
        diagnose_answer_gaps(
            application,
            answer,
            contract,
            next(
                evidence
                for evidence in application.evidence
                if evidence.event_id in answer.evidence_ids
            ),
        )["company_vision_gap"]
    )
    target_min = contract.character_target_min or contract.character_budget
    target_max = contract.character_target_max or contract.character_budget
    if answer.character_count < target_min:
        distance = target_min - answer.character_count
    elif answer.character_count > target_max:
        distance = answer.character_count - target_max
    else:
        distance = 0
    return non_character_hard, character_hard, company_vision_gap, distance


def _context_tokens(text: str) -> list[str]:
    separators = "·,()–—/"
    normalized = text
    for separator in separators:
        normalized = normalized.replace(separator, " ")
    generic = {
        "데이터를",
        "연결해",
        "높이는",
        "생산",
        "업무",
        "현업",
        "검증",
        "시스템",
        "제조에서",
    }
    tokens = [
        token.strip(".을를이가은는에서")
        for token in normalized.split()
        if len(token) >= 3 and token not in generic
    ]
    distinctive = [
        token
        for token in tokens
        if any(character.isascii() for character in token) or len(token) >= 4
    ]
    return distinctive[:12] or tokens[:12]
