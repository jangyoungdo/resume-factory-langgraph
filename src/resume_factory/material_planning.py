from __future__ import annotations

import itertools
import re
from collections import Counter

from .schemas import (
    ApplicationInput,
    EvidencePacket,
    MaterialAssignment,
    MaterialPortfolioPlan,
    MaterialSelectionMode,
    QuestionNarrativeContract,
)


def preflight_material_capacity(application: ApplicationInput) -> MaterialPortfolioPlan | None:
    """Return a blocking plan before model calls when unique material is insufficient."""
    question_ids = [item.question_id for item in application.questions]
    if application.material_selection_mode is MaterialSelectionMode.PINNED:
        mapping = application.pinned_evidence_ids_by_question
        if not mapping:
            # Direct/legacy ApplicationInput uses best_for_questions as its pinned mapping.
            mapping = {
                question_id: next(
                    (
                        item.event_id
                        for item in application.evidence
                        if question_id in item.best_for_questions
                    ),
                    application.evidence[0].event_id,
                )
                for question_id in question_ids
            }
        by_id = {item.event_id: item for item in application.evidence}
        keys = [
            by_id[event_id].experience_key for event_id in mapping.values() if event_id in by_id
        ]
        duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
        missing = [qid for qid in question_ids if mapping.get(qid) not in by_id]
        if duplicates or missing:
            return _blocked_plan(application, missing or question_ids, duplicates)
        return None

    unique_keys = {item.experience_key for item in application.evidence if item.experience_key}
    if len(unique_keys) < len(question_ids):
        return _blocked_plan(application, question_ids[len(unique_keys) :], [])
    return None


def allocate_materials(
    application: ApplicationInput,
    contracts: list[QuestionNarrativeContract],
) -> MaterialPortfolioPlan:
    by_id = {item.event_id: item for item in application.evidence}
    if application.material_selection_mode is MaterialSelectionMode.PINNED:
        mapping = application.pinned_evidence_ids_by_question or {
            contract.question_id: next(
                (
                    item.event_id
                    for item in application.evidence
                    if contract.question_id in item.best_for_questions
                ),
                application.evidence[0].event_id,
            )
            for contract in contracts
        }
        choices = tuple(by_id[mapping[item.question_id]] for item in contracts)
    else:
        ranked_choices = []
        for contract in contracts:
            ranked_choices.append(
                sorted(
                    application.evidence,
                    key=lambda evidence: (
                        -_allocation_score(application, contract, evidence),
                        evidence.experience_key,
                        evidence.event_id,
                    ),
                )[:8]
            )
        valid = [
            combination
            for combination in itertools.product(*ranked_choices)
            if len({item.experience_key for item in combination}) == len(combination)
        ]
        if not valid:
            return _blocked_plan(application, [item.question_id for item in contracts], [])
        choices = min(
            valid,
            key=lambda combination: (
                -round(
                    sum(
                        _allocation_score(application, contract, evidence)
                        for contract, evidence in zip(contracts, combination, strict=True)
                    ),
                    2,
                ),
                -sum(
                    application.preferred_evidence_ids_by_question.get(contract.question_id)
                    == evidence.event_id
                    for contract, evidence in zip(contracts, combination, strict=True)
                ),
                tuple(item.experience_key for item in combination),
                tuple(item.event_id for item in combination),
            ),
        )

    assignments = []
    for contract, evidence in zip(contracts, choices, strict=True):
        score = _allocation_score(application, contract, evidence)
        rejected = sorted(
            (
                item.event_id
                for item in application.evidence
                if item.experience_key != evidence.experience_key
            ),
            key=lambda event_id: (
                -_allocation_score(application, contract, by_id[event_id]),
                event_id,
            ),
        )[:3]
        supporting = next(
            (item for item in application.questions if item.question_id == contract.question_id),
        ).supporting_evidence_ids
        assignments.append(
            MaterialAssignment(
                question_id=contract.question_id,
                primary_event_id=evidence.event_id,
                primary_experience_key=evidence.experience_key,
                supporting_event_id=supporting[0] if supporting else None,
                allocation_score=score,
                selection_reason=(
                    f"질문 적합성·근거 완결성·차별성·직무 전이·면접 방어성을 합산해 "
                    f"{evidence.experience_key}를 선택"
                ),
                rejected_candidates=rejected,
            )
        )
    return MaterialPortfolioPlan(
        assignments=assignments,
        searched_candidate_count=len(application.evidence),
    )


def _allocation_score(
    application: ApplicationInput,
    contract: QuestionNarrativeContract,
    evidence: EvidencePacket,
) -> float:
    question_tokens = _tokens(
        " ".join([contract.direct_answer_required, *contract.preferred_evidence_traits])
    )
    evidence_tokens = _tokens(
        " ".join(
            [
                evidence.title,
                evidence.problem,
                evidence.judgment,
                *evidence.capabilities,
                *evidence.actions,
                *evidence.results,
            ]
        )
    )
    question_fit = (
        5.0
        if contract.question_id in evidence.best_for_questions
        else min(4.5, 1.5 + 0.45 * len(question_tokens & evidence_tokens))
    )
    preferred = application.preferred_evidence_ids_by_question.get(contract.question_id)
    if preferred == evidence.event_id:
        question_fit = 5.0
    completeness = min(
        5.0,
        1.5
        + bool(evidence.problem)
        + bool(evidence.judgment)
        + min(1.0, len(evidence.actions) / 2)
        + min(1.0, len(evidence.results) / 2)
        + 0.5 * bool(evidence.numeric_authorities),
    )
    differentiation = min(5.0, 2.0 + 0.5 * len(set(evidence.capabilities)))
    company_tokens = _tokens(f"{application.job_description} {application.company_context}")
    transfer = min(5.0, 1.5 + 0.45 * len(company_tokens & evidence_tokens))
    defensibility = 5.0 if evidence.boundaries and evidence.source_hash else 3.5
    weighted = (
        question_fit * 0.35
        + completeness * 0.25
        + differentiation * 0.15
        + transfer * 0.15
        + defensibility * 0.10
    )
    # A preferred item is a curated human hint, not a pin. It can resolve close
    # candidates but cannot rescue a materially weak evidence packet.
    if preferred == evidence.event_id:
        weighted += 0.2
    return round(min(5.0, weighted), 3)


def _blocked_plan(
    application: ApplicationInput,
    missing_questions: list[str],
    duplicates: list[str],
) -> MaterialPortfolioPlan:
    return MaterialPortfolioPlan(
        status="blocked_insufficient_evidence",
        code="INSUFFICIENT_DISTINCT_EVIDENCE",
        unassigned_questions=missing_questions,
        missing_questions=missing_questions,
        duplicate_experience_keys=duplicates,
        required_evidence_traits={item.question_id: [] for item in application.questions},
        searched_candidate_count=len(application.evidence),
    )


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9가-힣]+", text) if len(token) > 1}
