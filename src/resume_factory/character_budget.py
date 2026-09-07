from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from .schemas import (
    ApplicationInput,
    DraftAnswer,
    EvidencePacket,
    SentencePlan,
    SentenceRole,
    TransferContract,
)


@dataclass(frozen=True)
class CharacterBounds:
    limit: int
    hard_min: int
    target_min: int
    target_max: int


def bounds_for(limit: int) -> CharacterBounds:
    return CharacterBounds(
        limit=limit,
        hard_min=math.ceil(limit * 0.95),
        target_min=math.ceil(limit * 0.97),
        target_max=math.floor(limit * 0.98),
    )


def rewrite_to_character_target(
    answer: DraftAnswer,
    application: ApplicationInput,
    evidence: EvidencePacket,
    transfer: TransferContract,
) -> tuple[DraftAnswer, bool]:
    """Perform one conservative, evidence-bound expansion attempt.

    The offline implementation only adds complete sentences derived from supplied
    evidence, company context, or the transfer contract. It never pads with spaces or
    truncates a factual sentence merely to hit a count.
    """

    if answer.character_limit is None:
        return answer, False
    bounds = bounds_for(answer.character_limit)
    if bounds.target_min <= answer.character_count <= bounds.target_max:
        return answer, False
    if answer.character_count > bounds.target_max:
        return _shorten_to_target(answer, bounds), True

    candidates = _candidate_plans(answer, application, evidence, transfer)
    chosen = _best_additions(answer, candidates, bounds)
    if not chosen:
        return answer, True

    plans = _merge_additions(answer.sentence_plans, chosen)
    rewritten = answer.model_copy(
        update={
            "body": " ".join(item.text for item in plans),
            "sentence_plans": plans,
        }
    )
    return rewritten, True


def _candidate_plans(
    answer: DraftAnswer,
    application: ApplicationInput,
    evidence: EvidencePacket,
    transfer: TransferContract,
) -> list[SentencePlan]:
    raw: list[tuple[SentenceRole, str, str, str | None]] = []
    raw.extend(
        (SentenceRole.ACTION, text, "추가로 확인된 수행 행동", evidence.event_id)
        for text in evidence.actions[1:]
    )
    raw.extend(
        (SentenceRole.RESULT, text, "추가로 확인된 결과", evidence.event_id)
        for text in evidence.results[1:]
    )
    if evidence.capabilities:
        capabilities = "과 ".join(evidence.capabilities[:2])
        raw.append(
            (
                SentenceRole.DIFFERENTIATION,
                f"이 과정에서 {capabilities} 역량을 함께 검증했습니다.",
                "경험으로 증명한 차별점",
                evidence.event_id,
            )
        )
    raw.extend(
        [
            (
                SentenceRole.COMPANY_NEED,
                f"{application.company}의 실제 설비 기준과 작업 이력을 먼저 대조하겠습니다.",
                "회사 기준을 우선하는 적용 방식",
                None,
            ),
            (
                SentenceRole.CAUSAL_BRIDGE,
                "이 판단은 불필요한 조치를 줄이고 다음 작업자가 같은 근거를 확인하게 합니다.",
                "판단과 현장 가치의 인과 연결",
                evidence.event_id,
            ),
            (
                SentenceRole.CAUSAL_BRIDGE,
                "전제도 기록했습니다.",
                "판단을 재검토할 수 있는 기록",
                evidence.event_id,
            ),
        ]
    )
    existing = {item.text for item in answer.sentence_plans}
    candidates: list[SentencePlan] = []
    next_index = len(answer.sentence_plans) + 1
    for role, text, selling_point, evidence_id in raw:
        normalized = _sentence(text)
        if not normalized or normalized in existing:
            continue
        candidates.append(
            SentencePlan(
                sentence_id=f"{answer.question_id}-S{next_index:02d}",
                text=normalized,
                role=role,
                selling_point=selling_point,
                evidence_event_id=evidence_id,
                claim_ids=[f"{answer.question_id}-C{next_index:02d}"],
                company_connection=(application.job if role is SentenceRole.COMPANY_NEED else None),
                interview_defensible=evidence_id is not None,
            )
        )
        next_index += 1
    return candidates


def _best_additions(
    answer: DraftAnswer,
    candidates: list[SentencePlan],
    bounds: CharacterBounds,
) -> list[SentencePlan]:
    best: tuple[tuple[int, int, int], list[SentencePlan]] | None = None
    for count in range(1, len(candidates) + 1):
        for combination in itertools.combinations(candidates, count):
            added = sum(len(item.text) + 1 for item in combination)
            total = answer.character_count + added
            if total > bounds.limit:
                continue
            target_distance = (
                0
                if bounds.target_min <= total <= bounds.target_max
                else min(abs(total - bounds.target_min), abs(total - bounds.target_max))
            )
            hard_penalty = 0 if total >= bounds.hard_min else bounds.hard_min - total
            score = (hard_penalty, target_distance, count)
            if best is None or score < best[0]:
                best = (score, list(combination))
    return best[1] if best else []


def _merge_additions(
    plans: list[SentencePlan], additions: list[SentencePlan]
) -> list[SentencePlan]:
    role_target = {
        SentenceRole.ACTION: SentenceRole.ACTION,
        SentenceRole.RESULT: SentenceRole.RESULT,
        SentenceRole.DIFFERENTIATION: SentenceRole.DIFFERENTIATION,
        SentenceRole.COMPANY_NEED: SentenceRole.COMPANY_NEED,
        SentenceRole.CAUSAL_BRIDGE: SentenceRole.JUDGMENT,
    }
    merged = [item.model_copy(deep=True) for item in plans]
    for addition in additions:
        target_role = role_target.get(addition.role, SentenceRole.JUDGMENT)
        target = next((item for item in merged if item.role is target_role), merged[-1])
        target.text = f"{target.text} {addition.text}"
        target.selling_point = f"{target.selling_point}; {addition.selling_point}"
        target.claim_ids.extend(addition.claim_ids)
        if addition.company_connection:
            target.company_connection = addition.company_connection
        target.interview_defensible = target.interview_defensible or addition.interview_defensible
    return merged


def _shorten_to_target(answer: DraftAnswer, bounds: CharacterBounds) -> DraftAnswer:
    replacements = (
        ("결과를 점검 순서와 기록으로 연결했습니다.", "결과를 점검 기록으로 연결했습니다."),
        ("분석값보다 확인 가능한 근거를 중시합니다.", "확인 가능한 근거를 중시합니다."),
        ("이 경험에서 증명한 판단 방식을", "이 판단 방식을"),
        ("직무에는 예방보전 판단의 일관성이 필요합니다.", "예방보전 판단에는 일관성이 필요합니다."),
    )
    available = [
        pair for pair in replacements if any(pair[0] in plan.text for plan in answer.sentence_plans)
    ]
    best: tuple[tuple[int, int], tuple[tuple[str, str], ...]] | None = None
    for count in range(1, len(available) + 1):
        for combination in itertools.combinations(available, count):
            saved = sum(len(old) - len(new) for old, new in combination)
            total = answer.character_count - saved
            if total > bounds.limit:
                continue
            target_distance = (
                0
                if bounds.target_min <= total <= bounds.target_max
                else min(abs(total - bounds.target_min), abs(total - bounds.target_max))
            )
            hard_penalty = 0 if total >= bounds.hard_min else bounds.hard_min - total
            score = (hard_penalty, target_distance)
            if best is None or score < best[0]:
                best = (score, combination)
    if best is None:
        return answer
    plans = [item.model_copy(deep=True) for item in answer.sentence_plans]
    for old, new in best[1]:
        for plan in plans:
            if old in plan.text:
                plan.text = plan.text.replace(old, new, 1)
                break
    return answer.model_copy(
        update={"body": " ".join(item.text for item in plans), "sentence_plans": plans}
    )


def _sentence(text: str) -> str:
    normalized = " ".join(text.strip().split())
    if normalized and normalized[-1] not in ".!?다요":
        normalized += "."
    return normalized
