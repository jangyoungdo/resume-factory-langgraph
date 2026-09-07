from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping

from .schemas import (
    ApplicationInput,
    DraftAnswer,
    EvidencePacket,
    SentenceRole,
    ValidationIssue,
    ValidationReport,
)

NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")
GENERIC_PHRASES = ("열정으로 기여하겠습니다", "최선을 다하겠습니다")


def validate_answers(application: ApplicationInput, answers: list[DraftAnswer]) -> ValidationReport:
    issues: list[ValidationIssue] = []
    evidence_by_id = {item.event_id: item for item in application.evidence}
    authority_values = {
        str(authority.value)
        for evidence in application.evidence
        for authority in evidence.numeric_authorities
        if authority.status == "verified"
    }
    question_limits = {
        question.question_id: question.character_limit for question in application.questions
    }

    for answer in answers:
        if answer.character_count > question_limits[answer.question_id]:
            issues.append(
                ValidationIssue(
                    code="CHARACTER_LIMIT",
                    severity="hard_fail",
                    message=f"{answer.character_count}/{question_limits[answer.question_id]}자",
                )
            )
        if not answer.sentence_plans:
            issues.append(
                ValidationIssue(
                    code="NO_SENTENCE_PLAN",
                    severity="hard_fail",
                    message="문장 역할표가 없습니다.",
                )
            )
        if application.company not in answer.body:
            issues.append(
                ValidationIssue(
                    code="COMPANY_SPECIFICITY",
                    severity="hard_fail",
                    message="회사명이 포함된 구체적 직무 적용 문장이 없습니다.",
                )
            )
        if not any(item.role is SentenceRole.TRANSFER for item in answer.sentence_plans):
            issues.append(
                ValidationIssue(
                    code="NO_TRANSFER",
                    severity="hard_fail",
                    message="입사 후 구체적인 직무 전이 문장이 없습니다.",
                )
            )
        for phrase in GENERIC_PHRASES:
            if phrase in answer.body:
                issues.append(
                    ValidationIssue(
                        code="GENERIC_CLAIM",
                        severity="hard_fail",
                        message=f"금지된 일반론 표현: {phrase}",
                    )
                )
        for sentence in answer.sentence_plans:
            if sentence.evidence_event_id and sentence.evidence_event_id not in evidence_by_id:
                issues.append(
                    ValidationIssue(
                        code="UNKNOWN_EVIDENCE",
                        severity="hard_fail",
                        message=f"알 수 없는 근거 {sentence.evidence_event_id}",
                        sentence_id=sentence.sentence_id,
                    )
                )
            if sentence.evidence_event_id and not sentence.interview_defensible:
                issues.append(
                    ValidationIssue(
                        code="NOT_INTERVIEW_DEFENSIBLE",
                        severity="hard_fail",
                        message="근거 문장을 면접에서 방어할 수 없습니다.",
                        sentence_id=sentence.sentence_id,
                    )
                )
            for number in NUMBER_RE.findall(sentence.text):
                if number not in authority_values:
                    issues.append(
                        ValidationIssue(
                            code="UNVERIFIED_NUMBER",
                            severity="hard_fail",
                            message=f"검증되지 않은 수치 {number}",
                            sentence_id=sentence.sentence_id,
                        )
                    )
        _validate_forbidden_combinations(answer, evidence_by_id, issues)

    role_counts = Counter(
        sentence.role for answer in answers for sentence in answer.sentence_plans
    )
    total = sum(role_counts.values()) or 1
    evidence_roles = {
        SentenceRole.PROBLEM,
        SentenceRole.JUDGMENT,
        SentenceRole.ACTION,
        SentenceRole.RESULT,
        SentenceRole.VALIDATION,
    }
    perspective_roles = {SentenceRole.PERSPECTIVE, SentenceRole.DIFFERENTIATION}
    transfer_roles = {SentenceRole.COMPANY_NEED, SentenceRole.TRANSFER}
    metrics: dict[str, float | int | bool] = {
        "unsupported_claim_count": sum(i.code == "UNKNOWN_EVIDENCE" for i in issues),
        "numeric_mismatch_count": sum(i.code == "UNVERIFIED_NUMBER" for i in issues),
        "fact_collision_count": sum(i.code == "FORBIDDEN_COMBINATION" for i in issues),
        "sentence_role_coverage": sum(role_counts.values()) / total,
        "evidence_action_result_ratio": sum(role_counts[r] for r in evidence_roles) / total,
        "perspective_differentiation_ratio": sum(role_counts[r] for r in perspective_roles) / total,
        "company_transfer_ratio": sum(role_counts[r] for r in transfer_roles) / total,
    }
    ratio_limits = (
        ("EVIDENCE_RATIO", metrics["evidence_action_result_ratio"], 0.50, 0.60),
        ("PERSPECTIVE_RATIO", metrics["perspective_differentiation_ratio"], 0.20, 0.25),
        ("TRANSFER_RATIO", metrics["company_transfer_ratio"], 0.15, 0.20),
    )
    for code, value, lower, upper in ratio_limits:
        if not lower <= float(value) <= upper:
            issues.append(
                ValidationIssue(
                    code=code,
                    severity="hard_fail",
                    message=f"문장 구성비 {value:.1%}, 허용 범위 {lower:.0%}~{upper:.0%}",
                )
            )
    return ValidationReport(
        passed=not any(issue.severity == "hard_fail" for issue in issues),
        issues=issues,
        metrics=metrics,
    )


def _validate_forbidden_combinations(
    answer: DraftAnswer,
    evidence_by_id: Mapping[str, EvidencePacket],
    issues: list[ValidationIssue],
) -> None:
    used = set(answer.evidence_ids)
    for evidence_id in used:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            continue
        forbidden = getattr(evidence, "forbidden_combinations", [])
        for other in forbidden:
            if other in used:
                issues.append(
                    ValidationIssue(
                        code="FORBIDDEN_COMBINATION",
                        severity="hard_fail",
                        message=f"분리해야 하는 사건이 함께 사용됨: {evidence_id}, {other}",
                    )
                )
