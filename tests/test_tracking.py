import sys
from pathlib import Path
from types import SimpleNamespace

from resume_factory.agents import DeterministicBackend
from resume_factory.config import Settings
from resume_factory.graph import run_resume_graph
from resume_factory.schemas import ApplicationInput, ExecutionMode
from resume_factory.tracking import Tracker

FIXTURE = Path(__file__).parent / "fixtures" / "sample_application.json"


async def test_mlflow_receives_aggregate_usage_without_draft_content(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {"tables": []}

    def log_metrics(metrics):  # type: ignore[no-untyped-def]
        captured["metrics"] = metrics

    def set_tags(tags):  # type: ignore[no-untyped-def]
        captured["tags"] = tags

    def log_table(data, artifact_file):  # type: ignore[no-untyped-def]
        captured["tables"].append((artifact_file, data))  # type: ignore[union-attr]

    def log_dict(data, artifact_file):  # type: ignore[no-untyped-def]
        captured["summary"] = (artifact_file, data)

    fake_mlflow = SimpleNamespace(
        log_params=lambda params: captured.update(params=params),
        log_metrics=log_metrics,
        set_tags=set_tags,
        log_table=log_table,
        log_dict=log_dict,
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    settings = Settings(
        model_luna="gpt-5.6-luna",
        model_terra="gpt-5.6-terra",
        model_sol="gpt-5.6-sol",
        execution_mode=ExecutionMode.BALANCED,
        dual_brain_root=None,
        notion_snapshot_dir=None,
        notion_token=None,
        openai_api_key=None,
        enable_mlflow=True,
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        local_dir=tmp_path / ".local",
    )
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    result = await run_resume_graph(application, DeterministicBackend(), ExecutionMode.BALANCED)

    Tracker(settings).log_result(result)

    assert captured["metrics"]["total_tokens"] == 0  # type: ignore[index]
    assert len(captured["tables"]) == 4  # type: ignore[arg-type]
    serialized = repr(captured)
    assert result.answers[0].body not in serialized
    assert "evidence_ids" not in serialized
