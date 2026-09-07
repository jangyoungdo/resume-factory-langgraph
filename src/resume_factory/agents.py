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
    BackendProvider,
    BillingMode,
    CallKind,
    CostStatus,
    DraftProposal,
    ModelCallRecord,
    ModelTier,
    PrepSoaraStructure,
    SentencePlan,
    SentenceRole,
    UsageStatus,
)


class AgentBackend(Protocol):
    calls: list[ModelCallRecord]
    price_catalog_version: str
    provider: BackendProvider
    billing_mode: BillingMode
    cost_status: CostStatus

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

    provider = BackendProvider.LOCAL
    billing_mode = BillingMode.OFFLINE
    cost_status = CostStatus.NOT_APPLICABLE

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
        draft = _deterministic_draft(brief) if brief.get("writing_contract") else None
        integrated = []
        if role == "integration_editor":
            for raw in brief.get("drafts", []):
                integrated.append(DraftProposal.model_validate(raw))
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
            draft=draft,
            drafts=integrated,
        )


class OpenAIBackend:
    provider = BackendProvider.OPENAI_API
    billing_mode = BillingMode.API
    cost_status = CostStatus.ESTIMATED

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
                    provider=self.provider,
                    billing_mode=self.billing_mode,
                    cost_status=CostStatus.UNKNOWN,
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
                provider=self.provider,
                billing_mode=self.billing_mode,
                cost_status=(
                    CostStatus.ESTIMATED if estimate.cost_usd is not None else CostStatus.UNKNOWN
                ),
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
        provider=BackendProvider.LOCAL,
        billing_mode=BillingMode.OFFLINE,
        cost_status=CostStatus.NOT_APPLICABLE,
    )


def _deterministic_draft(brief: dict[str, Any]) -> DraftProposal:
    question_id = str(brief["question_id"])
    company = str(brief["company"])
    job = str(brief["job"])
    evidence = dict(brief["evidence"])
    transfer = dict(brief["transfer"])
    event_id = str(evidence["event_id"])
    raw = [
        (
            SentenceRole.ANSWER,
            f"이 경험에서 증명한 판단 방식을 {company} {job}에 적용하겠습니다.",
            "직접 답변",
            None,
        ),
        (
            SentenceRole.COMPANY_NEED,
            f"{company} {job}에는 부분 최적화보다 전체 흐름을 보는 판단이 필요합니다.",
            "회사 직무 수요",
            None,
        ),
        (
            SentenceRole.PERSPECTIVE,
            "저는 결과보다 먼저 확인 가능한 근거와 판단의 전제를 고정합니다.",
            "지원자의 관점",
            event_id,
        ),
        (SentenceRole.PROBLEM, str(evidence["problem"]), "해결 대상", event_id),
        (SentenceRole.JUDGMENT, str(evidence["judgment"]), "판단 이유", event_id),
        (SentenceRole.ACTION, str(evidence["actions"][0]), "구체 행동", event_id),
        (SentenceRole.RESULT, str(evidence["results"][0]), "검증 결과", event_id),
        (
            SentenceRole.DIFFERENTIATION,
            "판단 조건과 검증 결과를 함께 남겨 팀이 같은 기준으로 다음 행동을 정했습니다.",
            "차별점",
            event_id,
        ),
        (SentenceRole.TRANSFER, str(transfer["first_action"]), "입사 후 첫 행동", event_id),
        (
            SentenceRole.VALIDATION,
            f"성과는 {transfer['output_or_kpi']}로 확인하겠습니다.",
            "검증 기준",
            event_id,
        ),
    ]
    plans = [
        SentencePlan(
            sentence_id=f"{question_id}-S{index:02d}",
            text=text,
            role=role,
            selling_point=selling_point,
            evidence_event_id=evidence_id,
            claim_ids=[f"{question_id}-C{index:02d}"],
            company_connection=company
            if role in {SentenceRole.ANSWER, SentenceRole.COMPANY_NEED, SentenceRole.TRANSFER}
            else None,
            interview_defensible=evidence_id is not None,
        )
        for index, (role, text, selling_point, evidence_id) in enumerate(raw, start=1)
    ]
    return DraftProposal(
        question_id=question_id,
        headline=f"[{evidence['title']}]",
        direct_answer=plans[0].text,
        prep_soara_structure=PrepSoaraStructure(
            p=plans[0].text,
            r=plans[4].text,
            e_soara=" ".join(item.text for item in plans[3:8]),
            p2=plans[8].text,
        ),
        sentence_plans=plans,
        evidence_ids=[event_id],
        company_transfer=plans[8].text,
        interview_defense=[str(item) for item in evidence.get("boundaries", [])],
        confidence=0.88,
    )
