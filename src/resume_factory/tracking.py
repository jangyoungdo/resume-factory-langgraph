from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from .config import Settings
from .schemas import RunResult


class Tracker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @contextmanager
    def run(self, name: str) -> Iterator[None]:
        if not self.settings.enable_mlflow:
            yield
            return
        import mlflow

        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        mlflow.set_experiment("resume-factory")
        with mlflow.start_run(run_name=name):
            yield

    def log_result(self, result: RunResult) -> None:
        if not self.settings.enable_mlflow:
            return
        import mlflow

        mlflow.log_params(
            {
                "mode": result.telemetry.mode.value,
                "company": result.input_summary["company"],
                "job": result.input_summary["job"],
                "status": result.status,
            }
        )
        metrics = {
            key: float(value)
            for key, value in result.validation.metrics.items()
            if isinstance(value, (int, float, bool))
        }
        metrics["total_cost_usd"] = result.telemetry.total_cost_usd
        metrics["model_calls"] = len(result.telemetry.model_calls)
        mlflow.log_metrics(metrics)
