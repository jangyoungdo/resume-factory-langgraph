from __future__ import annotations

from pathlib import Path

from .character_budget import bounds_for
from .schemas import SubmissionBundle


def validate_submission_bundle(bundle: SubmissionBundle) -> list[str]:
    errors: list[str] = []
    for answer in bundle.answers:
        bounds = bounds_for(answer.character_limit)
        if answer.character_count < bounds.hard_min:
            errors.append(
                f"{answer.question_id}: {answer.character_count}/{bounds.limit}자; "
                f"최소 {bounds.hard_min}자"
            )
        if answer.character_count > bounds.limit:
            errors.append(
                f"{answer.question_id}: {answer.character_count}/{bounds.limit}자; 상한 초과"
            )
    return errors


def render_submission_markdown(bundle: SubmissionBundle) -> str:
    lines = [
        f"# {bundle.company} {bundle.job} — Resume Factory 최종 검토본",
        "",
        f"- revision: {bundle.revision}",
        f"- source run: `{bundle.source_run_id}`",
        f"- 상태: {bundle.status} / 실제 제출 미수행",
        "- 계산 기준: 소제목 + 줄바꿈 1자 + 본문",
        "",
    ]
    for answer in bundle.answers:
        lines.extend(
            [
                f"## {answer.question_id}. {answer.prompt}",
                "",
                (
                    f"- 글자 수: {answer.character_count}/{answer.character_limit}"
                    f" ({answer.utilization_ratio:.1%})"
                ),
                f"- 근거 ID: {', '.join(f'`{item}`' for item in answer.evidence_ids)}",
                "",
                "```text",
                answer.submission_text,
                "```",
                "",
            ]
        )
    if bundle.eligibility_warnings:
        lines.extend(["## 제출 전 경고", ""])
        lines.extend(f"- {warning}" for warning in bundle.eligibility_warnings)
        lines.append("")
    return "\n".join(lines)


def render_bundle_file(input_path: Path, output_path: Path) -> Path:
    bundle = SubmissionBundle.model_validate_json(input_path.read_text(encoding="utf-8"))
    errors = validate_submission_bundle(bundle)
    if errors:
        raise ValueError("; ".join(errors))
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing deliverable: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_submission_markdown(bundle), encoding="utf-8")
    return output_path
