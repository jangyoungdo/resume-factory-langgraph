from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_openai import ChatOpenAI

from .config import Settings
from .schemas import AgentProposal, AgentScore, ModelCallRecord, ModelTier

PRICE_PER_MILLION: dict[ModelTier, tuple[float, float]] = {
    ModelTier.LOCAL: (0.0, 0.0),
    ModelTier.LUNA: (0.20, 1.20),
    ModelTier.TERRA: (2.00, 12.00),
    ModelTier.SOL: (4.00, 20.00),
}


class AgentBackend(Protocol):
    calls: list[ModelCallRecord]

    async def propose(
        self,
        *,
        role: str,
        team: str,
        brief: dict[str, Any],
        tier: ModelTier,
    ) -> AgentProposal: ...


@dataclass(frozen=True)
class AgentSpec:
    role: str
    tier: ModelTier
    focus: str


class DeterministicBackend:
    """Offline backend for tests and pipeline demonstrations.

    It deliberately produces traceable, conservative text instead of pretending to
    reproduce model quality. Every proposal is grounded in IDs supplied in the brief.
    """

    def __init__(self) -> None:
        self.calls: list[ModelCallRecord] = []

    async def propose(
        self,
        *,
        role: str,
        team: str,
        brief: dict[str, Any],
        tier: ModelTier,
    ) -> AgentProposal:
        await asyncio.sleep(0)
        evidence_ids = list(brief.get("evidence_ids", []))
        focus = str(brief.get("focus", role))
        company = str(brief.get("company", "대상 회사"))
        job = str(brief.get("job", "지원 직무"))
        recommendation = (
            f"{company} {job} 관점에서 {focus}을(를) 우선하고, "
            f"검증된 근거 {', '.join(evidence_ids) or '없음'}만 사용한다."
        )
        score = AgentScore(
            relevance=4.2,
            evidence_fidelity=4.8 if evidence_ids else 3.0,
            company_transfer=4.1,
            differentiation=3.8,
            sentence_efficiency=4.0,
            defensibility=4.5 if evidence_ids else 3.0,
        )
        self.calls.append(
            ModelCallRecord(
                agent_role=role,
                tier=ModelTier.LOCAL,
                model="deterministic-local",
                input_tokens=0,
                output_tokens=0,
                estimated_cost_usd=0,
                latency_ms=0,
            )
        )
        return AgentProposal(
            agent_role=role,
            proposal_id=f"{team}-{uuid.uuid4().hex[:8]}",
            recommendation=recommendation,
            claims=[recommendation],
            evidence_ids=evidence_ids,
            risks=[] if evidence_ids else ["근거 ID가 제공되지 않음"],
            score=score,
            confidence=0.88 if evidence_ids else 0.65,
            needs_escalation=not evidence_ids,
        )


class OpenAIBackend:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for online execution")
        self.settings = settings
        self.calls: list[ModelCallRecord] = []

    async def propose(
        self,
        *,
        role: str,
        team: str,
        brief: dict[str, Any],
        tier: ModelTier,
    ) -> AgentProposal:
        model_name = self.settings.model_for(tier)
        model = ChatOpenAI(model=model_name, temperature=0).with_structured_output(
            AgentProposal, method="json_schema", include_raw=True
        )
        prompt = (
            "당신은 근거 기반 자기소개서 버티컬 AI의 전문 에이전트다. "
            "제공되지 않은 사실이나 수치를 만들지 말고, evidence_ids를 주장에 연결하라. "
            "다른 에이전트와 상의하지 말고 독립적으로 제안하라.\n"
            f"team={team}\nrole={role}\nbrief={json.dumps(brief, ensure_ascii=False)}"
        )
        started = time.perf_counter()
        response = await model.ainvoke(prompt)
        elapsed = int((time.perf_counter() - started) * 1000)
        parsed = response["parsed"]
        result = (
            parsed if isinstance(parsed, AgentProposal) else AgentProposal.model_validate(parsed)
        )
        raw = response["raw"]
        usage = getattr(raw, "usage_metadata", None) or {}
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        input_price, output_price = PRICE_PER_MILLION[tier]
        estimated_cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
        self.calls.append(
            ModelCallRecord(
                agent_role=role,
                tier=tier,
                model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=round(estimated_cost, 8),
                latency_ms=elapsed,
            )
        )
        return result


async def gather_independent_proposals(
    backend: AgentBackend,
    specs: list[AgentSpec],
    team: str,
    brief: dict[str, Any],
) -> list[AgentProposal]:
    return list(
        await asyncio.gather(
            *(
                backend.propose(
                    role=spec.role,
                    team=team,
                    brief={**brief, "focus": spec.focus},
                    tier=spec.tier,
                )
                for spec in specs
            )
        )
    )
