from __future__ import annotations

import asyncio
import platform
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .agents import DeterministicBackend, OpenAIBackend
from .config import Settings
from .graph import run_resume_graph
from .indexing import build_curated_index
from .schemas import ApplicationInput, ExecutionMode
from .storage import RunStore
from .tracking import Tracker

app = typer.Typer(no_args_is_help=True, help="Evidence-grounded Resume Factory")
console = Console()


@app.command()
def doctor() -> None:
    """Check runtime, privacy boundaries, credentials, and optional services."""
    settings = Settings.from_env()
    checks = {
        "Python 3.12": sys.version_info[:2] == (3, 12),
        "OpenAI key (optional offline)": bool(settings.openai_api_key),
        "Dual Brain root": bool(settings.dual_brain_root and settings.dual_brain_root.exists()),
        "Notion snapshot": bool(
            settings.notion_snapshot_dir and settings.notion_snapshot_dir.exists()
        ),
        "Local data excluded": settings.local_dir.name == ".local",
    }
    table = Table(title=f"Resume Factory doctor · {platform.python_version()}")
    table.add_column("Check")
    table.add_column("Status")
    for name, passed in checks.items():
        table.add_row(name, "PASS" if passed else "OPTIONAL/NOT CONFIGURED")
    console.print(table)


@app.command("index")
def index_command(
    source: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    semantic: Annotated[
        bool, typer.Option(help="Build local BGE-M3 embeddings after curating JSON evidence")
    ] = True,
) -> None:
    """Build a curated evidence index inside the private vault."""
    settings = Settings.from_env()
    if settings.dual_brain_root is None:
        raise typer.BadParameter("RF_DUAL_BRAIN_ROOT is required")
    output, count = build_curated_index(
        source.resolve(), settings.dual_brain_root, semantic=semantic
    )
    console.print(f"Indexed {count} evidence packets into {output}")


@app.command()
def run(
    input: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    offline: Annotated[
        bool, typer.Option(help="Use deterministic agents and no external API")
    ] = False,
    mode: Annotated[ExecutionMode | None, typer.Option()] = None,
) -> None:
    """Run the graph and save a private, non-overwriting draft result."""
    settings = Settings.from_env()
    application_input = ApplicationInput.model_validate_json(input.read_text(encoding="utf-8"))
    backend = DeterministicBackend() if offline else OpenAIBackend(settings)
    selected_mode = mode or settings.execution_mode
    tracker = Tracker(settings)
    with tracker.run(f"{application_input.company}-{application_input.job}"):
        result = asyncio.run(run_resume_graph(application_input, backend, selected_mode))
        tracker.log_result(result)
    path = RunStore(settings.local_dir).save(result)
    console.print(f"run_id={result.run_id} status={result.status} saved={path}")


@app.command()
def review(run_id: str) -> None:
    """Show evidence, warnings, validation, and model cost for one run."""
    result = RunStore(Settings.from_env().local_dir).load(run_id)
    console.print_json(result.model_dump_json(indent=2))


@app.command("eval")
def eval_command(suite: Annotated[str, typer.Option()] = "golden") -> None:
    """Run the public, synthetic regression suite."""
    if suite != "golden":
        raise typer.BadParameter("only the golden suite is available")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/golden"], check=False
    )
    raise typer.Exit(completed.returncode)


@app.command()
def diagram() -> None:
    """Render Mermaid sources into reproducible SVG and PNG files."""
    completed = subprocess.run(["npm", "run", "diagrams"], check=False)
    raise typer.Exit(completed.returncode)


if __name__ == "__main__":
    app()
