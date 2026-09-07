from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .schemas import HumanFeedback, RunResult
from .usage import aggregate_calls


class RunStore:
    def __init__(self, local_dir: Path) -> None:
        self.local_dir = local_dir
        self.runs_dir = local_dir / "runs"
        self.db_path = local_dir / "resume_factory.sqlite"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    company TEXT NOT NULL,
                    job TEXT NOT NULL,
                    total_cost_usd REAL NOT NULL,
                    result_path TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    company TEXT NOT NULL,
                    job TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    cached_input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    reasoning_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    total_cost_usd REAL NOT NULL,
                    cost_complete INTEGER NOT NULL,
                    price_catalog_version TEXT,
                    character_rewrite_count INTEGER NOT NULL,
                    validation_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS model_calls (
                    call_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    question_id TEXT,
                    team TEXT NOT NULL,
                    agent_role TEXT NOT NULL,
                    call_kind TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    cached_input_tokens INTEGER NOT NULL,
                    cache_creation_input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    reasoning_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    estimated_cost_usd REAL,
                    price_catalog_version TEXT,
                    latency_ms INTEGER NOT NULL,
                    retry_count INTEGER,
                    usage_status TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    error_code TEXT,
                    started_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS human_feedback (
                    run_id TEXT PRIMARY KEY,
                    decision TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    final_draft_path TEXT,
                    final_draft_hash TEXT,
                    overall_edit_ratio REAL,
                    per_question_edit_ratio_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (2)"
            )
            self._backfill_legacy_runs(connection)

    def _backfill_legacy_runs(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT result_path FROM runs
            WHERE run_id NOT IN (SELECT run_id FROM usage_runs)
            """
        ).fetchall()
        for (raw_path,) in rows:
            try:
                result = RunResult.model_validate_json(
                    Path(str(raw_path)).read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            self._save_usage(connection, result)

    def save(self, result: RunResult) -> Path:
        path = self.runs_dir / f"{result.run_id}.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO runs
                (run_id, status, company, job, total_cost_usd, result_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    result.run_id,
                    result.status,
                    result.input_summary["company"],
                    result.input_summary["job"],
                    result.telemetry.total_cost_usd,
                    str(path),
                ),
            )
            self._save_usage(connection, result)
        return path

    def _save_usage(self, connection: sqlite3.Connection, result: RunResult) -> None:
        total = aggregate_calls(result.telemetry.model_calls)[0]
        connection.execute(
            """
            INSERT OR REPLACE INTO usage_runs
            (run_id, status, company, job, mode, input_tokens, cached_input_tokens,
             output_tokens, reasoning_tokens, total_tokens, total_cost_usd,
             cost_complete, price_catalog_version, character_rewrite_count,
             validation_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.run_id,
                result.status,
                result.input_summary["company"],
                result.input_summary["job"],
                result.telemetry.mode.value,
                total["input_tokens"],
                total["cached_input_tokens"],
                total["output_tokens"],
                total["reasoning_tokens"],
                total["total_tokens"],
                total["estimated_cost_usd"],
                int(total["cost_complete"]),
                result.telemetry.price_catalog_version,
                len(result.telemetry.metadata.get("character_rewrite_attempts", [])),
                json.dumps(result.validation.metrics, ensure_ascii=False),
            ),
        )
        connection.execute("DELETE FROM model_calls WHERE run_id = ?", (result.run_id,))
        connection.executemany(
            """
            INSERT INTO model_calls VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    (
                        call.call_id
                        if call.call_id != "legacy"
                        else f"{result.run_id}-legacy-{index}"
                    ),
                    result.run_id,
                    call.sequence,
                    call.question_id,
                    call.team,
                    call.agent_role,
                    call.call_kind.value,
                    call.tier.value,
                    call.model,
                    call.input_tokens,
                    call.cached_input_tokens,
                    call.cache_creation_input_tokens,
                    call.output_tokens,
                    call.reasoning_tokens,
                    call.total_tokens,
                    call.estimated_cost_usd,
                    call.price_catalog_version,
                    call.latency_ms,
                    call.retry_count,
                    call.usage_status.value,
                    int(call.success),
                    call.error_code,
                    call.started_at.isoformat(),
                )
                for index, call in enumerate(result.telemetry.model_calls, start=1)
            ],
        )

    def load(self, run_id: str) -> RunResult:
        path = self.runs_dir / f"{run_id}.json"
        return RunResult.model_validate_json(path.read_text(encoding="utf-8"))

    def list_runs(self) -> list[dict[str, object]]:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def list_usage_runs(self, limit: int = 20) -> list[dict[str, object]]:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM usage_runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def save_feedback(self, feedback: HumanFeedback) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO human_feedback
                (run_id, decision, rating, final_draft_path, final_draft_hash,
                 overall_edit_ratio, per_question_edit_ratio_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback.run_id,
                    feedback.decision.value,
                    feedback.rating,
                    feedback.final_draft_path,
                    feedback.final_draft_hash,
                    feedback.overall_edit_ratio,
                    json.dumps(feedback.per_question_edit_ratio, ensure_ascii=False),
                    feedback.created_at.isoformat(),
                ),
            )

    def load_feedback(self, run_id: str) -> dict[str, object] | None:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM human_feedback WHERE run_id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None
