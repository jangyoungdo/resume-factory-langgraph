from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from .config import Settings
from .schemas import RunResult
from .usage import aggregate_calls


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
        experiment = mlflow.get_experiment_by_name("resume-factory")
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                "resume-factory",
                artifact_location=(self.settings.local_dir / "mlflow-artifacts").resolve().as_uri(),
            )
        else:
            experiment_id = experiment.experiment_id
        with mlflow.start_run(run_name=name, experiment_id=experiment_id):
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
                "provider": result.telemetry.provider.value,
                "billing_mode": result.telemetry.billing_mode.value,
                "graph_version": result.telemetry.graph_version,
            }
        )
        metrics = {
            key: float(value)
            for key, value in result.validation.metrics.items()
            if isinstance(value, (int, float, bool))
        }
        if result.telemetry.cost_status.value == "estimated":
            metrics["total_cost_usd"] = result.telemetry.total_cost_usd
        metrics["model_calls"] = len(result.telemetry.model_calls)
        total = aggregate_calls(result.telemetry.model_calls)[0]
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
            "latency_ms",
            "failures",
            "missing_usage_calls",
            "offline_calls",
        ):
            metrics[key] = float(total[key])
        metrics["character_rewrite_count"] = float(
            len(result.telemetry.metadata.get("character_rewrite_attempts", []))
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "company": result.input_summary["company"],
                "job": result.input_summary["job"],
                "price_catalog_version": result.telemetry.price_catalog_version or "unknown",
                "cost_complete": str(result.telemetry.cost_complete).lower(),
                "cost_status": result.telemetry.cost_status.value,
            }
        )
        for group_by in ("question", "team", "agent", "model"):
            rows = aggregate_calls(result.telemetry.model_calls, group_by)
            mlflow.log_table(
                data=_table_dict(rows),
                artifact_file=f"usage/by_{group_by}.json",
            )
        mlflow.log_dict(
            {
                "run_id": result.run_id,
                "price_catalog_version": result.telemetry.price_catalog_version,
                "cost_complete": result.telemetry.cost_complete,
                "totals": total,
            },
            "usage/summary.json",
        )


def _table_dict(rows: list[dict[str, object]]) -> dict[str, list[object]]:
    if not rows:
        return {}
    return {key: [row[key] for row in rows] for key in rows[0]}
