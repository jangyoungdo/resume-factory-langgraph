from __future__ import annotations

import sqlite3
from pathlib import Path

from .schemas import RunResult


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
        return path

    def load(self, run_id: str) -> RunResult:
        path = self.runs_dir / f"{run_id}.json"
        return RunResult.model_validate_json(path.read_text(encoding="utf-8"))

    def list_runs(self) -> list[dict[str, object]]:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]
