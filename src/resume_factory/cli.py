from __future__ import annotations

import asyncio
import platform
import subprocess
import sys
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.table import Table

from .agents import DeterministicBackend, OpenAIBackend
from .config import Settings
from .deliverables import render_bundle_file
from .graph import run_resume_graph
from .indexing import build_curated_index
from .schemas import ApplicationInput, ExecutionMode, FeedbackDecision
from .storage import RunStore
from .tracking import Tracker
from .usage import GroupBy, build_feedback, compare_reports, usage_report

app = typer.Typer(no_args_is_help=True, help="Evidence-grounded Resume Factory")
usage_app = typer.Typer(no_args_is_help=True, help="Inspect local model token and cost usage")
app.add_typer(usage_app, name="usage")
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


def _group_by(value: str) -> GroupBy:
    if value not in {"question", "team", "agent", "model"}:
        raise typer.BadParameter("group-by must be question, team, agent, or model")
    return cast(GroupBy, value)


@usage_app.command("list")
def usage_list(limit: Annotated[int, typer.Option(min=1, max=500)] = 20) -> None:
    """List locally retained usage summaries."""
    rows = RunStore(Settings.from_env().local_dir).list_usage_runs(limit)
    table = Table(title="Resume Factory usage runs")
    columns = ("run_id", "company", "job", "mode", "total_tokens", "total_cost_usd")
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*(str(row[column]) for column in columns))
    console.print(table)


@usage_app.command("show")
def usage_show(
    run_id: str,
    group_by: Annotated[str, typer.Option()] = "team",
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show one run grouped by question, team, agent, or model."""
    store = RunStore(Settings.from_env().local_dir)
    report = usage_report(store.load(run_id), _group_by(group_by))
    feedback = store.load_feedback(run_id)
    if feedback:
        report["human_feedback"] = feedback
    if json_output:
        console.print_json(data=report)
        return
    _print_usage_report(report)


@usage_app.command("compare")
def usage_compare(
    run_ids: Annotated[list[str], typer.Argument()],
    group_by: Annotated[str, typer.Option()] = "team",
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Compare usage and QA vectors across two or more runs."""
    if len(run_ids) < 2:
        raise typer.BadParameter("provide at least two run IDs")
    store = RunStore(Settings.from_env().local_dir)
    report = compare_reports([store.load(item) for item in run_ids], _group_by(group_by))
    for item in report["runs"]:
        feedback = store.load_feedback(str(item["run_id"]))
        if feedback:
            item["human_feedback"] = feedback
    if json_output:
        console.print_json(data=report)
        return
    table = Table(title="Usage and quality comparison")
    for column in (
        "run_id",
        "mode",
        "tokens",
        "cost_usd",
        "cost_complete",
        "QA",
        "target_hits",
        "fact_errors",
        "rating",
        "edit_ratio",
    ):
        table.add_column(column)
    for item in report["runs"]:
        total = item["total"]
        validation = item["validation"]
        feedback = item.get("human_feedback") or {}
        table.add_row(
            str(item["run_id"]),
            str(item["mode"]),
            str(total["total_tokens"]),
            f"{float(cast(float | int, total['estimated_cost_usd'])):.6f}",
            str(total["cost_complete"]),
            str(item["status"]),
            str(validation.get("character_target_hit_count", "-")),
            str(
                int(validation.get("unsupported_claim_count", 0))
                + int(validation.get("numeric_mismatch_count", 0))
                + int(validation.get("fact_collision_count", 0))
            ),
            str(feedback.get("rating", "-")),
            str(feedback.get("overall_edit_ratio", "-")),
        )
    console.print(table)
    console.print("Quality is shown as a vector; cost does not determine the best draft.")


@app.command()
def feedback(
    run_id: str,
    decision: Annotated[str, typer.Option()],
    rating: Annotated[int, typer.Option(min=1, max=5)],
    final_draft: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    """Store private human preference and edit-distance metadata for a run."""
    try:
        resolved_decision = FeedbackDecision(decision)
    except ValueError as error:
        raise typer.BadParameter("decision must be accepted, revised, or rejected") from error
    store = RunStore(Settings.from_env().local_dir)
    record = build_feedback(
        store.load(run_id),
        decision=resolved_decision,
        rating=rating,
        final_draft=final_draft,
    )
    store.save_feedback(record)
    console.print(
        f"feedback_saved={run_id} decision={record.decision.value} rating={record.rating}"
    )


def _print_usage_report(report: dict[str, object]) -> None:
    total = cast(dict[str, object], report["total"])
    console.print(
        f"run={report['run_id']} mode={report['mode']} status={report['status']} "
        f"tokens={total['total_tokens']} "
        f"cost_usd={float(cast(float | int, total['estimated_cost_usd'])):.6f} "
        f"cost_complete={total['cost_complete']}"
    )
    table = Table(title="Usage breakdown")
    columns = (
        "group",
        "calls",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "offline_calls",
        "missing_usage_calls",
    )
    for column in columns:
        table.add_column(column)
    for row in cast(list[dict[str, object]], report["groups"]):
        table.add_row(*(str(row[column]) for column in columns))
    console.print(table)
    total_unknown = int(cast(int, total["unknown_price_calls"]))
    total_missing = int(cast(int, total["missing_usage_calls"]))
    if total_unknown or total_missing:
        console.print(
            f"WARNING unknown_price_calls={total_unknown} missing_usage_calls={total_missing}"
        )
    top = cast(list[dict[str, object]], report["top_agents"])
    if top:
        console.print(
            "Top agents: "
            + ", ".join(
                f"{item['group']}=${float(cast(float | int, item['estimated_cost_usd'])):.6f}"
                for item in top[:10]
            )
        )
    console.print_json(data=cast(dict[str, object], report["validation"]))


@app.command("render-draft")
def render_draft(
    input: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    """Render a canonical private submission JSON into a non-overwriting Markdown file."""
    try:
        path = render_bundle_file(input.resolve(), output.resolve())
    except (ValueError, FileExistsError) as error:
        raise typer.BadParameter(str(error)) from error
    console.print(f"rendered={path}")


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
