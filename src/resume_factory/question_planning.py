from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter, ValidationError

from .character_budget import bounds_for
from .schemas import (
    ApplicationInput,
    EvidencePacket,
    QuestionArchetype,
    QuestionNarrativeContract,
    TeamDecision,
)

SEQUENCES: dict[QuestionArchetype, list[str]] = {
    QuestionArchetype.COMPETENCY_EFFORT: [
        "경쟁력 선언",
        "부족함 또는 계기",
        "역량을 기른 구체 노력",
        "검증 결과",
        "현재의 판단 원칙",
        "직무 적용",
    ],
    QuestionArchetype.TEAMWORK_ROLE: [
        "맡은 역할과 이유",
        "공동 문제",
        "의견과 책임 조정",
        "구체 행동",
        "공동 결과",
        "협업 원칙",
    ],
    QuestionArchetype.CONTRIBUTION_TRANSFER: [
        "기여 명제",
        "회사와 직무 과제",
        "다른 경험의 증거",
        "전이 가능한 방법",
        "입사 후 첫 행동과 검증 기준",
    ],
    QuestionArchetype.LEARNING_TRANSFER: [
        "가장 큰 노력 또는 실패",
        "대응 과정",
        "깨달음",
        "후행 경험의 재적용",
        "달라진 결과와 태도",
    ],
    QuestionArchetype.MOTIVATION_FIT: [
        "회사 선택 이유",
        "직무 접점",
        "경험 근거",
        "회사에서 만들 가치",
    ],
    QuestionArchetype.GROWTH_VALUES: [
        "현재 가치관",
        "형성 계기",
        "행동 변화",
        "검증 경험",
        "직무에서의 태도",
    ],
    QuestionArchetype.FREEFORM: ["직접 답변", "근거", "판단과 행동", "결과", "직무 연결"],
}


def build_question_contracts(
    application: ApplicationInput,
    decision: TeamDecision | None = None,
) -> tuple[list[QuestionNarrativeContract], bool]:
    raw = (
        decision.structured_payload.question_narratives
        if decision and decision.structured_payload
        else []
    )
    if raw:
        try:
            parsed = TypeAdapter(list[QuestionNarrativeContract]).validate_python(raw)
            if {item.question_id for item in parsed} == {
                item.question_id for item in application.questions
            }:
                return parsed, False
        except ValidationError:
            pass
    return (
        [_fallback_contract(application, item.model_dump()) for item in application.questions],
        True,
    )


def ground_contract(
    contract: QuestionNarrativeContract,
    evidence: EvidencePacket,
) -> QuestionNarrativeContract:
    core = contract.core_message.strip()
    if not core or core == contract.direct_answer.strip():
        if contract.archetype is QuestionArchetype.COMPETENCY_EFFORT:
            core = f"{evidence.judgment.rstrip('.')}는 경쟁력"
        elif contract.archetype is QuestionArchetype.TEAMWORK_ROLE:
            core = f"{evidence.title}에서 맡은 역할과 공동 목표를 완성한 방식"
        elif contract.archetype is QuestionArchetype.CONTRIBUTION_TRANSFER:
            core = f"{evidence.title}의 방법을 지원 직무의 구체 행동으로 전이"
        else:
            core = evidence.judgment
    return contract.model_copy(update={"core_message": core})


def classify_question(text: str) -> QuestionArchetype:
    lowered = text.lower()
    if ("배웠" in text or "배운" in text or "깨달" in text) and any(
        token in text for token in ("경험", "성공", "실패", "노력")
    ):
        return QuestionArchetype.LEARNING_TRANSFER
    if any(token in text for token in ("팀 활동", "공동 목표", "협업", "역할을 맡")):
        return QuestionArchetype.TEAMWORK_ROLE
    if any(token in text for token in ("입사 후", "기여할", "기여 방안")) and not any(
        token in text for token in ("지원 동기", "지원동기")
    ):
        return QuestionArchetype.CONTRIBUTION_TRANSFER
    if any(token in text for token in ("경쟁력", "역량")) and any(
        token in text for token in ("노력", "갖추", "기울인")
    ):
        return QuestionArchetype.COMPETENCY_EFFORT
    if "지원 동기" in text or "지원동기" in text or "why" in lowered:
        return QuestionArchetype.MOTIVATION_FIT
    if any(token in text for token in ("성장과정", "가치관", "영향을 끼친")):
        return QuestionArchetype.GROWTH_VALUES
    return QuestionArchetype.FREEFORM


def _fallback_contract(
    application: ApplicationInput,
    question: dict[str, Any],
) -> QuestionNarrativeContract:
    archetype = classify_question(str(question["text"]))
    bounds = bounds_for(int(question["character_limit"]))
    return QuestionNarrativeContract(
        question_id=str(question["question_id"]),
        archetype=archetype,
        direct_answer_required=f"{question['text']}에 첫 문장부터 직접 답한다.",
        direct_answer=str(question["text"]),
        core_message="",
        buyer_intent="질문이 요구하는 판단과 행동을 근거로 확인",
        required_evidence_type="문제·판단·행동·결과가 연결된 검증 사건",
        required_company_connection=f"{application.company} {application.job}의 구체 행동",
        likely_objections=["주장과 경험의 구체 행동이 실제로 연결되는가?"],
        forbidden_generic_claims=["열정으로 기여하겠습니다", "최선을 다하겠습니다"],
        required_elements=SEQUENCES[archetype],
        optional_elements=["검증된 정량 결과", "후행 적용"],
        forbidden_detours=["도구 목록", "질문과 무관한 두 번째 성공담", "반복 방어 문장"],
        preferred_evidence_traits=_preferred_traits(archetype),
        narrative_sequence=SEQUENCES[archetype],
        character_budget=bounds.limit,
        character_limit=bounds.limit,
        character_hard_min=bounds.hard_min,
        character_target_min=bounds.target_min,
        character_target_max=bounds.target_max,
    )


def _preferred_traits(archetype: QuestionArchetype) -> list[str]:
    return {
        QuestionArchetype.COMPETENCY_EFFORT: ["역량 형성", "반복 노력", "가설 검증"],
        QuestionArchetype.TEAMWORK_ROLE: ["팀 역할", "조정", "공동 결과"],
        QuestionArchetype.CONTRIBUTION_TRANSFER: ["직무 전이", "구체 행동", "검증 기준"],
        QuestionArchetype.LEARNING_TRANSFER: ["실패", "깨달음", "후행 적용"],
        QuestionArchetype.MOTIVATION_FIT: ["회사 선택", "직무 접점", "차별성"],
        QuestionArchetype.GROWTH_VALUES: ["가치관", "행동 변화", "지속성"],
        QuestionArchetype.FREEFORM: ["직접 답변", "판단", "결과"],
    }[archetype]
