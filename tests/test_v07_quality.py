from pathlib import Path

from resume_factory.agents import DeterministicBackend
from resume_factory.editorial_quality import apply_edit_operations
from resume_factory.graph import run_resume_graph
from resume_factory.material_planning import allocate_materials
from resume_factory.question_planning import build_question_contracts, classify_question
from resume_factory.schemas import (
    ApplicationInput,
    EditOperation,
    ExecutionMode,
    MaterialSelectionMode,
    QuestionArchetype,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_application.json"


def _application() -> ApplicationInput:
    return ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


async def test_duplicate_experience_keys_block_before_model_calls() -> None:
    application = _application()
    application.questions.append(application.questions[0].model_copy(update={"question_id": "Q2"}))
    alias = application.evidence[0].model_copy(
        update={"event_id": "SYNTH-ALIAS-02", "best_for_questions": ["Q2"]}
    )
    application.evidence.append(alias)
    application.pinned_evidence_ids_by_question = {
        "Q1": application.evidence[0].event_id,
        "Q2": alias.event_id,
    }
    backend = DeterministicBackend()

    result = await run_resume_graph(application, backend, ExecutionMode.BALANCED)

    assert result.status == "blocked_insufficient_evidence"
    assert result.material_plan is not None
    assert result.material_plan.code == "INSUFFICIENT_DISTINCT_EVIDENCE"
    assert backend.calls == []


def test_auto_allocator_is_global_unique_and_stable() -> None:
    application = _application()
    application.material_selection_mode = MaterialSelectionMode.AUTO_UNIQUE
    application.questions.extend(
        [
            application.questions[0].model_copy(
                update={"question_id": "Q2", "text": "팀 활동의 역할과 공동 목표를 쓰십시오."}
            ),
            application.questions[0].model_copy(
                update={"question_id": "Q3", "text": "입사 후 어떻게 기여할지 쓰십시오."}
            ),
        ]
    )
    for index, question_id in enumerate(("Q2", "Q3"), start=2):
        application.evidence.append(
            application.evidence[0].model_copy(
                update={
                    "event_id": f"SYNTH-{question_id}",
                    "experience_key": f"SYNTH_EXPERIENCE_{index}",
                    "best_for_questions": [question_id],
                }
            )
        )
    contracts, _ = build_question_contracts(application)

    first = allocate_materials(application, contracts)
    second = allocate_materials(application, contracts)

    assert first == second
    assert len({item.primary_experience_key for item in first.assignments}) == 3


def test_question_archetypes_do_not_force_one_narrative() -> None:
    assert (
        classify_question("경쟁력과 이를 갖추기 위해 기울인 노력을 쓰십시오.")
        is QuestionArchetype.COMPETENCY_EFFORT
    )
    assert (
        classify_question("팀 활동에서 맡은 역할과 공동 목표 달성 과정을 쓰십시오.")
        is QuestionArchetype.TEAMWORK_ROLE
    )
    assert (
        classify_question("실패 경험과 그 과정에서 무엇을 배웠는지 쓰십시오.")
        is QuestionArchetype.LEARNING_TRANSFER
    )


async def test_final_drafts_receive_independent_reader_critique() -> None:
    application = _application()
    backend = DeterministicBackend()

    result = await run_resume_graph(application, backend, ExecutionMode.BALANCED)

    critics = [call for call in backend.calls if call.agent_role == "human_reader_critic"]
    assert len(critics) == len(application.questions)
    assert all(item.passed for item in result.editorial_assessments)
    assert "editorial_quality_gate" in result.telemetry.graph_nodes_completed


async def test_sentence_scoped_edit_preserves_untouched_plans() -> None:
    application = _application()
    result = await run_resume_graph(application, DeterministicBackend(), ExecutionMode.BALANCED)
    original = result.answers[0]
    untouched = original.sentence_plans[1].model_dump()
    target = original.sentence_plans[0]
    updated = apply_edit_operations(
        original,
        [
            EditOperation(
                operation="replace_sentence",
                target_sentence_ids=[target.sentence_id],
                text="저의 경쟁력은 검증 기준부터 다시 세우는 판단입니다.",
            )
        ],
    )

    assert updated.sentence_plans[0].text != target.text
    assert updated.sentence_plans[1].model_dump() == untouched
    assert len(updated.sentence_plans) == len(original.sentence_plans)
