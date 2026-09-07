from pathlib import Path

from resume_factory.agents import DeterministicBackend
from resume_factory.graph import run_resume_graph
from resume_factory.schemas import ApplicationInput, ExecutionMode
from resume_factory.validators import NUMBER_RE, validate_answers

FIXTURE = Path(__file__).parent / "fixtures" / "sample_application.json"


def test_numeric_parser_preserves_thousands_separators() -> None:
    assert [item.replace(",", "") for item in NUMBER_RE.findall("8,192개와 93.6Hz")] == [
        "8192",
        "93.6",
    ]


async def test_unverified_number_is_hard_failure() -> None:
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    result = await run_resume_graph(application, DeterministicBackend(), ExecutionMode.ECONOMY)
    result.answers[0].sentence_plans[0].text += " 999회 검증했습니다."
    report = validate_answers(application, result.answers)
    assert not report.passed
    assert any(issue.code == "UNVERIFIED_NUMBER" for issue in report.issues)


async def test_forbidden_event_combination_is_hard_failure() -> None:
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    second = application.evidence[0].model_copy(
        update={
            "event_id": "SYNTH-PHM-02",
            "forbidden_combinations": ["SYNTH-PHM-01"],
        }
    )
    application.evidence.append(second)
    result = await run_resume_graph(application, DeterministicBackend(), ExecutionMode.ECONOMY)
    result.answers[0].evidence_ids.append("SYNTH-PHM-02")
    report = validate_answers(application, result.answers)
    assert not report.passed
    assert any(issue.code == "FORBIDDEN_COMBINATION" for issue in report.issues)


async def test_character_budget_hard_and_target_boundaries() -> None:
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    result = await run_resume_graph(application, DeterministicBackend(), ExecutionMode.ECONOMY)
    source = result.answers[0]

    def answer_with_total(total: int):  # type: ignore[no-untyped-def]
        prefix = "Sample Steel "
        body_length = total - len(source.headline.strip()) - 1
        return source.model_copy(update={"body": prefix + "가" * (body_length - len(prefix))})

    expected = {
        569: "CHARACTER_UNDERFILL",
        570: "CHARACTER_TARGET_MISS",
        582: None,
        588: None,
        600: "CHARACTER_TARGET_MISS",
        601: "CHARACTER_LIMIT",
    }
    for count, code in expected.items():
        report = validate_answers(application, [answer_with_total(count)])
        character_codes = {
            issue.code for issue in report.issues if issue.code.startswith("CHARACTER_")
        }
        if code is None:
            assert not character_codes
        else:
            assert code in character_codes
