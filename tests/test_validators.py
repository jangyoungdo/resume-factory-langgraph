from pathlib import Path

from resume_factory.agents import DeterministicBackend
from resume_factory.graph import run_resume_graph
from resume_factory.schemas import ApplicationInput, ExecutionMode, SentenceRole
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


async def test_reader_facing_defensive_caveat_is_rejected() -> None:
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    result = await run_resume_graph(application, DeterministicBackend(), ExecutionMode.ECONOMY)
    result.answers[0].sentence_plans[2].text = (
        "이 수치는 설정 주기일 뿐 종단 시간을 보장한 값이 아닙니다."
    )

    report = validate_answers(application, result.answers)

    assert any(issue.code == "LOW_VALUE_DEFENSIVE_CAVEAT" for issue in report.issues)


async def test_scope_qualifier_may_appear_only_once() -> None:
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    result = await run_resume_graph(application, DeterministicBackend(), ExecutionMode.ECONOMY)
    result.answers[0].sentence_plans[0].text += " 교육용 가상 공정에서 수행했습니다."
    result.answers[0].sentence_plans[1].text += " 교육용 환경의 결과입니다."

    report = validate_answers(application, result.answers)

    assert any(issue.code == "REPEATED_SCOPE_QUALIFIER" for issue in report.issues)


async def test_vague_success_claim_is_rejected() -> None:
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    result = await run_resume_graph(application, DeterministicBackend(), ExecutionMode.ECONOMY)
    result.answers[0].sentence_plans[6].text = "이후 좋은 결과를 얻었습니다."

    report = validate_answers(application, result.answers)

    assert any(issue.code == "VAGUE_RESULT" for issue in report.issues)


async def test_judgment_must_lead_to_action_and_result_in_order() -> None:
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    result = await run_resume_graph(application, DeterministicBackend(), ExecutionMode.ECONOMY)
    plans = result.answers[0].sentence_plans
    judgment = next(index for index, item in enumerate(plans) if item.role is SentenceRole.JUDGMENT)
    action = next(index for index, item in enumerate(plans) if item.role is SentenceRole.ACTION)
    plans[judgment], plans[action] = plans[action], plans[judgment]

    report = validate_answers(application, result.answers)

    assert any(issue.code == "NO_ACTION_RESULT_CHAIN" for issue in report.issues)


async def test_learning_transfer_requires_supported_followup_result() -> None:
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    followup = application.evidence[0].model_copy(
        update={"event_id": "SYNTH-FOLLOWUP-02", "title": "후행 프로젝트"}
    )
    application.evidence.append(followup)
    application.questions[0].supporting_evidence_ids = [followup.event_id]
    result = await run_resume_graph(application, DeterministicBackend(), ExecutionMode.ECONOMY)

    missing = validate_answers(application, result.answers)
    assert any(issue.code == "LEARNING_TRANSFER_MISSING" for issue in missing.issues)

    result.answers[0].sentence_plans[7].evidence_event_id = followup.event_id
    result.answers[0].sentence_plans[7].role = SentenceRole.CAUSAL_BRIDGE
    grounded = validate_answers(application, result.answers)
    assert not any(
        issue.code in {"LEARNING_TRANSFER_MISSING", "LEARNING_TRANSFER_RESULT_MISSING"}
        for issue in grounded.issues
    )
