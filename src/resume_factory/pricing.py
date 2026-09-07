from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any


@dataclass(frozen=True)
class PriceEstimate:
    cost_usd: float | None
    catalog_version: str


class PriceCatalog:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.version = str(payload["catalog_version"])
        self.models = dict(payload["models"])

    @classmethod
    def bundled(cls) -> PriceCatalog:
        resource = files("resume_factory").joinpath("model_prices.json")
        return cls(json.loads(resource.read_text(encoding="utf-8")))

    def estimate(
        self,
        model: str,
        *,
        input_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
    ) -> PriceEstimate:
        prices = self.models.get(model)
        if prices is None:
            return PriceEstimate(None, self.version)
        cached = min(max(cached_input_tokens, 0), max(input_tokens, 0))
        uncached = max(input_tokens - cached, 0)
        cost = (
            uncached * float(prices["input_per_million"])
            + cached * float(prices["cached_input_per_million"])
            + max(output_tokens, 0) * float(prices["output_per_million"])
        ) / 1_000_000
        return PriceEstimate(round(cost, 8), self.version)
