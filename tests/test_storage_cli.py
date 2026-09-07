from pathlib import Path

from typer.testing import CliRunner

from resume_factory.agents import DeterministicBackend
from resume_factory.cli import app
from resume_factory.graph import run_resume_graph
from resume_factory.schemas import ApplicationInput, ExecutionMode
from resume_factory.storage import RunStore

FIXTURE = Path(__file__).parent / "fixtures" / "sample_application.json"


async def test_run_store_never_overwrites_input(tmp_path: Path) -> None:
    original = FIXTURE.read_text(encoding="utf-8")
    application = ApplicationInput.model_validate_json(original)
    result = await run_resume_graph(application, DeterministicBackend(), ExecutionMode.ECONOMY)
    store = RunStore(tmp_path / ".local")
    output = store.save(result)
    assert output.parent.name == "runs"
    assert store.load(result.run_id).run_id == result.run_id
    assert FIXTURE.read_text(encoding="utf-8") == original


def test_doctor_runs_without_credentials(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Python 3.12" in result.output


async def test_usage_cli_lists_shows_and_compares_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    application = ApplicationInput.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    first = await run_resume_graph(
        application, DeterministicBackend(), ExecutionMode.ECONOMY, run_id="usage-one"
    )
    second = await run_resume_graph(
        application, DeterministicBackend(), ExecutionMode.BALANCED, run_id="usage-two"
    )
    store = RunStore(tmp_path / ".local")
    store.save(first)
    store.save(second)
    runner = CliRunner()

    listed = runner.invoke(app, ["usage", "list"])
    shown = runner.invoke(app, ["usage", "show", "usage-one", "--group-by", "question"])
    compared = runner.invoke(
        app, ["usage", "compare", "usage-one", "usage-two", "--group-by", "model"]
    )
    feedback = runner.invoke(
        app, ["feedback", "usage-one", "--decision", "accepted", "--rating", "5"]
    )
    assert listed.exit_code == 0
    assert "usage-one" in listed.output
    assert shown.exit_code == 0
    assert "Q1" in shown.output
    assert compared.exit_code == 0
    assert "Quality is shown as a vector" in compared.output
    assert feedback.exit_code == 0
    assert store.load_feedback("usage-one")["rating"] == 5  # type: ignore[index]
