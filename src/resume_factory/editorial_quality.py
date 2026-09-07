from __future__ import annotations

from collections import Counter

from pydantic import ValidationError

from .schemas import (
    AgentProposal,
    DraftAnswer,
    EditOperation,
    EditorialAssessment,
    QuestionNarrativeContract,
    SentencePlan,
    SentenceRole,
)


def assessment_from_proposal(
    proposal: AgentProposal,
    answer: DraftAnswer,
    contract: QuestionNarrativeContract,
) -> EditorialAssessment:
    raw = proposal.structured_payload.editorial_assessment if proposal.structured_payload else None
    if raw:
        try:
            return EditorialAssessment.model_validate(raw)
        except ValidationError:
            pass
    return deterministic_assessment(answer, contract)


def deterministic_assessment(
    answer: DraftAnswer,
    contract: QuestionNarrativeContract,
) -> EditorialAssessment:
    normalized = [" ".join(item.text.split()) for item in answer.sentence_plans]
    counts = Counter(normalized)
    redundant = [
        item.sentence_id
        for item, text in zip(answer.sentence_plans, normalized, strict=True)
        if counts[text] > 1
    ]
    low_value = [
        item.sentence_id
        for item in answer.sentence_plans
        if len(item.text.strip()) < 12 or not item.selling_point.strip()
    ]
    excessive = [
        item.sentence_id
        for item in answer.sentence_plans
        if sum(token in item.text for token in ("·", "/", "→")) >= 4
    ]
    has_action = any(item.role is SentenceRole.ACTION for item in answer.sentence_plans)
    has_result = any(
        item.role in {SentenceRole.RESULT, SentenceRole.VALIDATION}
        for item in answer.sentence_plans
    )
    missing = []
    if not has_action:
        missing.append("구체 행동")
    if not has_result:
        missing.append("검증 결과")
    passed = not redundant and not low_value and not excessive and not missing
    return EditorialAssessment(
        question_id=answer.question_id,
        inferred_takeaway=contract.core_message,
        question_directness=5 if answer.sentence_plans else 1,
        thesis_clarity=4 if contract.core_message else 3,
        logical_continuity=4 if has_action and has_result else 2,
        evidence_to_claim=4,
        effort_or_action_specificity=4 if has_action else 2,
        company_transfer=4 if any(item.company_connection for item in answer.sentence_plans) else 3,
        redundant_sentence_ids=redundant,
        low_value_sentence_ids=low_value,
        excessive_technical_detail_ids=excessive,
        missing_information=missing,
        verdict="pass" if passed else "repair",
    )


def operations_from_proposal(proposal: AgentProposal) -> list[EditOperation]:
    raw = proposal.structured_payload.edit_operations if proposal.structured_payload else []
    operations = []
    for item in raw:
        try:
            operations.append(EditOperation.model_validate(item))
        except ValidationError:
            continue
    return operations


def apply_edit_operations(answer: DraftAnswer, operations: list[EditOperation]) -> DraftAnswer:
    """Apply sentence-scoped edits while preserving every untouched plan verbatim."""
    plans = [item.model_copy(deep=True) for item in answer.sentence_plans]
    for operation in operations:
        indices = [
            index
            for index, item in enumerate(plans)
            if item.sentence_id in operation.target_sentence_ids
        ]
        if not indices:
            continue
        first = min(indices)
        if operation.operation == "delete_sentence":
            plans = [
                item for item in plans if item.sentence_id not in operation.target_sentence_ids
            ]
        elif operation.operation == "replace_sentence" and operation.text:
            plans[first] = _updated_plan(plans[first], operation)
        elif operation.operation == "insert_after" and operation.text:
            anchor = plans[first]
            inserted = SentencePlan(
                sentence_id=f"{anchor.sentence_id}-I",
                text=operation.text,
                role=operation.role or SentenceRole.CAUSAL_BRIDGE,
                selling_point=operation.selling_point or "비평에서 확인된 논리 간극 보강",
                evidence_event_id=operation.evidence_event_id,
                company_connection=operation.company_connection,
                interview_defensible=operation.evidence_event_id is not None,
            )
            plans.insert(first + 1, inserted)
        elif operation.operation == "merge_sentences" and operation.text:
            merged = _updated_plan(plans[first], operation)
            plans = [
                item
                for index, item in enumerate(plans)
                if index == first or item.sentence_id not in operation.target_sentence_ids
            ]
            plans[first] = merged
        elif operation.operation == "reorder_span":
            by_id = {item.sentence_id: item for item in plans}
            selected = [by_id[item] for item in operation.target_sentence_ids if item in by_id]
            remainder = [
                item for item in plans if item.sentence_id not in operation.target_sentence_ids
            ]
            plans = remainder[:first] + selected + remainder[first:]

    for index, plan in enumerate(plans, start=1):
        plan.sentence_id = f"{answer.question_id}-S{index:02d}"
    return answer.model_copy(
        update={
            "sentence_plans": plans,
            "body": " ".join(item.text.strip() for item in plans),
            "evidence_ids": list(
                dict.fromkeys(item.evidence_event_id for item in plans if item.evidence_event_id)
            ),
        }
    )


def _updated_plan(plan: SentencePlan, operation: EditOperation) -> SentencePlan:
    return plan.model_copy(
        update={
            "text": operation.text or plan.text,
            "role": operation.role or plan.role,
            "selling_point": operation.selling_point or plan.selling_point,
            "evidence_event_id": operation.evidence_event_id or plan.evidence_event_id,
            "company_connection": operation.company_connection or plan.company_connection,
            "interview_defensible": bool(operation.evidence_event_id or plan.evidence_event_id),
        }
    )
