from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .schemas import ExecutionMode, ModelTier


@dataclass(frozen=True)
class Settings:
    model_luna: str
    model_terra: str
    model_sol: str
    execution_mode: ExecutionMode
    dual_brain_root: Path | None
    notion_snapshot_dir: Path | None
    notion_token: str | None
    openai_api_key: str | None
    enable_mlflow: bool
    mlflow_tracking_uri: str
    local_dir: Path

    @classmethod
    def from_env(cls, cwd: Path | None = None) -> Settings:
        load_dotenv()
        base = cwd or Path.cwd()
        return cls(
            model_luna=os.getenv("RF_MODEL_LUNA", "gpt-5.6-luna"),
            model_terra=os.getenv("RF_MODEL_TERRA", "gpt-5.6-terra"),
            model_sol=os.getenv("RF_MODEL_SOL", "gpt-5.6-sol"),
            execution_mode=ExecutionMode(os.getenv("RF_EXECUTION_MODE", "balanced")),
            dual_brain_root=_optional_path("RF_DUAL_BRAIN_ROOT"),
            notion_snapshot_dir=_optional_path("RF_NOTION_SNAPSHOT_DIR"),
            notion_token=os.getenv("RF_NOTION_TOKEN") or None,
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            enable_mlflow=os.getenv("RF_ENABLE_MLFLOW", "false").lower() == "true",
            mlflow_tracking_uri=os.getenv(
                "RF_MLFLOW_TRACKING_URI", "sqlite:///.local/mlflow.db"
            ),
            local_dir=base / ".local",
        )

    def model_for(self, tier: ModelTier) -> str:
        return {
            ModelTier.LUNA: self.model_luna,
            ModelTier.TERRA: self.model_terra,
            ModelTier.SOL: self.model_sol,
            ModelTier.LOCAL: "deterministic-local",
        }[tier]


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else None

