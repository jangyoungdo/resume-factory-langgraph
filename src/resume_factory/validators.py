from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping

from .editorial_policy import has_low_value_caveat, has_scope_qualifier, has_vague_result
from .schemas import (
    ApplicationInput,
    DraftAnswer,
    EvidencePacket,
    SentenceRole,
    ValidationIssue,
    ValidationReport,
)

NUMBER_RE = re.compile(r"(?<![A-Za-z])\d{1,3}(?:,\d{3})*(?:\.\d+)?|(?<![A-Za-z])\d+(?:\.\d+)?")
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
        limit = question_limits[answer.question_id]
        hard_min = math.ceil(limit * 0.95)
        target_min = math.ceil(limit * 0.97)
        target_max = math.floor(limit * 0.98)
        if answer.character_count < hard_min:
            issues.append(
                ValidationIssue(
                    code="CHARACTER_UNDERFILL",
                    severity="hard_fail",
                    message=f"{answer.character_count}/{limit}자; 최소 {hard_min}자",
                )
            )
        elif not target_min <= answer.character_count <= target_max:
            issues.append(
                ValidationIssue(
                    code="CHARACTER_TARGET_MISS",
                    severity="warning",
                    message=(
                        f"{answer.character_count}/{limit}자; 권장 목표 {target_min}~{target_max}자"
                    ),
                )
            )
        if answer.character_count > limit:
            issues.append(
                ValidationIssue(
                    code="CHARACTER_LIMIT",
                    severity="hard_fail",
                    message=f"{answer.character_count}/{limit}자",
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
        scope_sentences = [
            sentence
            for sentence in answer.sentence_plans
            if has_scope_qualifier(sentence.text)
        ]
        if len(scope_sentences) > 1:
            issues.append(
                ValidationIssue(
                    code="REPEATED_SCOPE_QUALIFIER",
                    severity="hard_fail",
                    message="사실 범위 설명은 최초 맥락에서 한 번만 사용해야 합니다.",
                    sentence_id=scope_sentences[1].sentence_id,
                )
            )
        judgment_indexes = [
            index
            for index, sentence in enumerate(answer.sentence_plans)
            if sentence.role is SentenceRole.JUDGMENT
        ]
        action_indexes = [
            index
            for index, sentence in enumerate(answer.sentence_plans)
            if sentence.role is SentenceRole.ACTION
        ]
        result_indexes = [
            index
            for index, sentence in enumerate(answer.sentence_plans)
            if sentence.role in {SentenceRole.RESULT, SentenceRole.VALIDATION}
        ]
        if not any(
            judgment_index < action_index < result_index
            for judgment_index in judgment_indexes
            for action_index in action_indexes
            for result_index in result_indexes
        ):
            issues.append(
                ValidationIssue(
                    code="NO_ACTION_RESULT_CHAIN",
                    severity="hard_fail",
                    message="판단 뒤의 구체 행동과 그 이후 검증 결과가 연결되지 않았습니다.",
                )
            )
        for sentence in answer.sentence_plans:
            if not sentence.selling_point.strip() or not (
                sentence.evidence_event_id or sentence.company_connection
            ):
                issues.append(
                    ValidationIssue(
                        code="NO_SENTENCE_VALUE",
                        severity="hard_fail",
                        message="문장에 판매 가치와 근거 또는 회사 연결이 필요합니다.",
                        sentence_id=sentence.sentence_id,
                    )
                )
            if has_low_value_caveat(sentence.text):
                issues.append(
                    ValidationIssue(
                        code="LOW_VALUE_DEFENSIVE_CAVEAT",
                        severity="hard_fail",
                        message=(
                            "사실 경계는 내부 검증에 남기고 독자 가치가 없는 보험 문장은 "
                            "본문에서 제거해야 합니다."
                        ),
                        sentence_id=sentence.sentence_id,
                    )
                )
            if has_vague_result(sentence.text):
                issues.append(
                    ValidationIssue(
                        code="VAGUE_RESULT",
                        severity="hard_fail",
                        message="결과를 추상적으로 평가하지 말고 관찰 가능한 변화를 써야 합니다.",
                        sentence_id=sentence.sentence_id,
                    )
                )
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
            for raw_number in NUMBER_RE.findall(sentence.text):
                number = raw_number.replace(",", "")
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
        question = next(
            item for item in application.questions if item.question_id == answer.question_id
        )
        if question.supporting_evidence_ids:
            supporting_plans = [
                sentence
                for sentence in answer.sentence_plans
                if sentence.evidence_event_id in question.supporting_evidence_ids
            ]
            if not supporting_plans:
                issues.append(
                    ValidationIssue(
                        code="LEARNING_TRANSFER_MISSING",
                        severity="hard_fail",
                        message="후행 경험에서 배운 방식의 재적용을 근거로 증명해야 합니다.",
                    )
                )
            elif not any(
                sentence.role
                in {SentenceRole.CAUSAL_BRIDGE, SentenceRole.RESULT, SentenceRole.VALIDATION}
                for sentence in supporting_plans
            ):
                issues.append(
                    ValidationIssue(
                        code="LEARNING_TRANSFER_RESULT_MISSING",
                        severity="hard_fail",
                        message="후행 적용의 행동뿐 아니라 확인된 결과까지 연결해야 합니다.",
                    )
                )

    role_counts = Counter(sentence.role for answer in answers for sentence in answer.sentence_plans)
    total = sum(role_counts.values()) or 1
    evidence_roles = {
        SentenceRole.PROBLEM,
        SentenceRole.JUDGMENT,
        SentenceRole.ACTION,
        SentenceRole.RESULT,
        SentenceRole.VALIDATION,
        SentenceRole.CAUSAL_BRIDGE,
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
        "character_target_hit_count": sum(
            math.ceil(question_limits[item.question_id] * 0.97)
            <= item.character_count
            <= math.floor(question_limits[item.question_id] * 0.98)
            for item in answers
        ),
    }
    for answer in answers:
        metrics[f"{answer.question_id}_character_count"] = answer.character_count
        metrics[f"{answer.question_id}_character_utilization"] = round(
            answer.character_count / question_limits[answer.question_id], 4
        )
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
