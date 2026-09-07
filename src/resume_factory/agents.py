from __future__ import annotations

import asyncio
import itertools
import json
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from langchain_openai import ChatOpenAI

from .config import Settings
from .pricing import PriceCatalog
from .schemas import (
    AgentProposal,
    AgentScore,
    CallKind,
    ModelCallRecord,
    ModelTier,
    UsageStatus,
)


class AgentBackend(Protocol):
    calls: list[ModelCallRecord]
    price_catalog_version: str

    def begin_run(self, run_id: str) -> None: ...

    def record_local_operation(
        self,
        *,
        role: str,
        team: str,
        question_id: str | None,
        call_kind: CallKind,
        latency_ms: int = 0,
    ) -> None: ...

    async def propose(
        self,
        *,
        role: str,
        team: str,
        brief: dict[str, Any],
        tier: ModelTier,
        question_id: str | None = None,
        call_kind: CallKind = CallKind.SPECIALIST,
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
        self.run_id = "unbound"
        self._sequence = itertools.count(1)
        self.price_catalog = PriceCatalog.bundled()
        self.price_catalog_version = self.price_catalog.version

    def begin_run(self, run_id: str) -> None:
        self.run_id = run_id
        self._sequence = itertools.count(1)

    def record_local_operation(
        self,
        *,
        role: str,
        team: str,
        question_id: str | None,
        call_kind: CallKind,
        latency_ms: int = 0,
    ) -> None:
        self.calls.append(
            _local_call_record(
                run_id=self.run_id,
                sequence=next(self._sequence),
                role=role,
                team=team,
                question_id=question_id,
                call_kind=call_kind,
                latency_ms=latency_ms,
                price_catalog_version=self.price_catalog_version,
            )
        )

    async def propose(
        self,
        *,
        role: str,
        team: str,
        brief: dict[str, Any],
        tier: ModelTier,
        question_id: str | None = None,
        call_kind: CallKind = CallKind.SPECIALIST,
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
                run_id=self.run_id,
                call_id=uuid.uuid4().hex,
                sequence=next(self._sequence),
                question_id=question_id,
                team=team,
                agent_role=role,
                call_kind=call_kind,
                tier=ModelTier.LOCAL,
                model="deterministic-local",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                estimated_cost_usd=0,
                price_catalog_version=self.price_catalog_version,
                latency_ms=0,
                retry_count=0,
                usage_status=UsageStatus.OFFLINE,
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
        self.run_id = "unbound"
        self._sequence = itertools.count(1)
        self.price_catalog = PriceCatalog.bundled()
        self.price_catalog_version = self.price_catalog.version

    def begin_run(self, run_id: str) -> None:
        self.run_id = run_id
        self._sequence = itertools.count(1)

    def record_local_operation(
        self,
        *,
        role: str,
        team: str,
        question_id: str | None,
        call_kind: CallKind,
        latency_ms: int = 0,
    ) -> None:
        self.calls.append(
            _local_call_record(
                run_id=self.run_id,
                sequence=next(self._sequence),
                role=role,
                team=team,
                question_id=question_id,
                call_kind=call_kind,
                latency_ms=latency_ms,
                price_catalog_version=self.price_catalog_version,
            )
        )

    async def propose(
        self,
        *,
        role: str,
        team: str,
        brief: dict[str, Any],
        tier: ModelTier,
        question_id: str | None = None,
        call_kind: CallKind = CallKind.SPECIALIST,
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
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        call_id = uuid.uuid4().hex
        sequence = next(self._sequence)
        try:
            response = await model.ainvoke(prompt)
        except Exception as error:
            elapsed = int((time.perf_counter() - started) * 1000)
            self.calls.append(
                ModelCallRecord(
                    run_id=self.run_id,
                    call_id=call_id,
                    sequence=sequence,
                    question_id=question_id,
                    team=team,
                    agent_role=role,
                    call_kind=call_kind,
                    tier=tier,
                    model=model_name,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    estimated_cost_usd=None,
                    price_catalog_version=self.price_catalog_version,
                    latency_ms=elapsed,
                    retry_count=None,
                    usage_status=UsageStatus.MISSING,
                    success=False,
                    error_code=type(error).__name__,
                    started_at=started_at,
                )
            )
            raise
        elapsed = int((time.perf_counter() - started) * 1000)
        parsed = response["parsed"]
        result = (
            parsed if isinstance(parsed, AgentProposal) else AgentProposal.model_validate(parsed)
        )
        raw = response["raw"]
        usage = getattr(raw, "usage_metadata", None) or {}
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        input_details = usage.get("input_token_details") or {}
        output_details = usage.get("output_token_details") or {}
        cached_input_tokens = int(input_details.get("cache_read", 0))
        cache_creation_tokens = int(input_details.get("cache_creation", 0))
        reasoning_tokens = int(output_details.get("reasoning", 0))
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens))
        estimate = self.price_catalog.estimate(
            model_name,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
        )
        usage_status = UsageStatus.REPORTED if usage else UsageStatus.MISSING
        self.calls.append(
            ModelCallRecord(
                run_id=self.run_id,
                call_id=call_id,
                sequence=sequence,
                question_id=question_id,
                team=team,
                agent_role=role,
                call_kind=call_kind,
                tier=tier,
                model=model_name,
                input_tokens=input_tokens,
                cached_input_tokens=cached_input_tokens,
                cache_creation_input_tokens=cache_creation_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=estimate.cost_usd,
                price_catalog_version=estimate.catalog_version,
                latency_ms=elapsed,
                retry_count=None,
                usage_status=usage_status,
                started_at=started_at,
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
                    question_id=(str(brief["question_id"]) if brief.get("question_id") else None),
                    call_kind=CallKind.SPECIALIST,
                )
                for spec in specs
            )
        )
    )


def _local_call_record(
    *,
    run_id: str,
    sequence: int,
    role: str,
    team: str,
    question_id: str | None,
    call_kind: CallKind,
    latency_ms: int,
    price_catalog_version: str,
) -> ModelCallRecord:
    return ModelCallRecord(
        run_id=run_id,
        call_id=uuid.uuid4().hex,
        sequence=sequence,
        question_id=question_id,
        team=team,
        agent_role=role,
        call_kind=call_kind,
        tier=ModelTier.LOCAL,
        model="deterministic-local",
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        estimated_cost_usd=0,
        price_catalog_version=price_catalog_version,
        latency_ms=latency_ms,
        retry_count=0,
        usage_status=UsageStatus.OFFLINE,
    )
