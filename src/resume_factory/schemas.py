from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ExecutionMode(StrEnum):
    ECONOMY = "economy"
    BALANCED = "balanced"
    PREMIUM = "premium"


class ModelTier(StrEnum):
    LOCAL = "local"
    LUNA = "luna"
    TERRA = "terra"
    SOL = "sol"


class SentenceRole(StrEnum):
    ANSWER = "answer"
    PERSPECTIVE = "perspective"
    COMPANY_NEED = "company_need"
    PROBLEM = "problem"
    JUDGMENT = "judgment"
    ACTION = "action"
    RESULT = "result"
    VALIDATION = "validation"
    DIFFERENTIATION = "differentiation"
    TRANSFER = "transfer"
    BOUNDARY = "boundary"
    CAUSAL_BRIDGE = "causal_bridge"


class ApplicationQuestion(BaseModel):
    question_id: str
    text: str
    character_limit: int = Field(default=600, ge=100, le=5000)
    required: bool = True


class NumericAuthority(BaseModel):
    authority_id: str
    value: float | int | str
    unit: str | None = None
    status: Literal["verified", "unverified"] = "verified"


class EvidencePacket(BaseModel):
    event_id: str
    title: str
    problem: str
    judgment: str
    actions: list[str]
    results: list[str]
    capabilities: list[str] = Field(default_factory=list)
    source_ref: str
    source_hash: str
    numeric_authorities: list[NumericAuthority] = Field(default_factory=list)
    boundaries: list[str] = Field(default_factory=list)
    forbidden_combinations: list[str] = Field(default_factory=list)


class ApplicationInput(BaseModel):
    company: str
    job: str
    industry: str = "manufacturing"
    job_description: str
    company_context: str
    questions: list[ApplicationQuestion]
    evidence: list[EvidencePacket]
    eligibility_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_questions_and_evidence(self) -> ApplicationInput:
        if not self.questions:
            raise ValueError("at least one application question is required")
        if not self.evidence:
            raise ValueError("at least one evidence packet is required")
        return self


class AgentScore(BaseModel):
    relevance: float = Field(ge=0, le=5)
    evidence_fidelity: float = Field(ge=0, le=5)
    company_transfer: float = Field(ge=0, le=5)
    differentiation: float = Field(ge=0, le=5)
    sentence_efficiency: float = Field(ge=0, le=5)
    defensibility: float = Field(ge=0, le=5)

    @property
    def weighted(self) -> float:
        return round(
            self.evidence_fidelity * 0.30
            + self.relevance * 0.20
            + self.company_transfer * 0.20
            + self.differentiation * 0.15
            + self.sentence_efficiency * 0.10
            + self.defensibility * 0.05,
            3,
        )


class AgentProposal(BaseModel):
    agent_role: str
    proposal_id: str
    recommendation: str
    claims: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    rejected_alternatives: list[str] = Field(default_factory=list)
    score: AgentScore
    confidence: float = Field(ge=0, le=1)
    needs_escalation: bool = False


class CritiqueReport(BaseModel):
    proposal_id: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    adjusted_score: float = Field(ge=0, le=5)


class TeamDecision(BaseModel):
    team: str
    selected_proposal_id: str
    decision: str
    rationale: str
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list)
    escalated_to: ModelTier | None = None


class DemandBrief(BaseModel):
    company_problems: list[str]
    responsibilities: list[str]
    kpis: list[str]
    failure_risks: list[str]
    behavioral_competencies: list[str]
    source_refs: list[str]


class QuestionContract(BaseModel):
    question_id: str
    direct_answer_required: str
    buyer_intent: str
    required_evidence_type: str
    required_company_connection: str
    likely_objections: list[str]
    forbidden_generic_claims: list[str]
    character_budget: int


class TransferContract(BaseModel):
    question_id: str
    evidence_event_id: str
    past_problem: str
    reusable_judgment: str
    reusable_action: str
    company_task: str
    first_action: str
    output_or_kpi: str
    boundary: str


class PositioningBrief(BaseModel):
    hire_me_reason: str
    primary_differentiator: str
    supporting_differentiators: list[str]
    proof_events: list[str]
    company_application: str
    avoid_language: list[str]
    anticipated_objections: list[str]


class SentencePlan(BaseModel):
    sentence_id: str
    text: str
    role: SentenceRole
    selling_point: str
    evidence_event_id: str | None = None
    claim_ids: list[str] = Field(default_factory=list)
    company_connection: str | None = None
    interview_defensible: bool = False


class DraftAnswer(BaseModel):
    question_id: str
    headline: str
    body: str
    sentence_plans: list[SentencePlan]
    evidence_ids: list[str]

    @property
    def character_count(self) -> int:
        return len(self.headline) + len(self.body)


class ValidationIssue(BaseModel):
    code: str
    severity: Literal["warning", "hard_fail"]
    message: str
    sentence_id: str | None = None


class ValidationReport(BaseModel):
    passed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    metrics: dict[str, float | int | bool] = Field(default_factory=dict)


class ModelCallRecord(BaseModel):
    agent_role: str
    tier: ModelTier
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    latency_ms: int
    retry_count: int = 0


class RunTelemetry(BaseModel):
    run_id: str
    mode: ExecutionMode
    model_calls: list[ModelCallRecord] = Field(default_factory=list)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    total_cost_usd: float = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    run_id: str
    status: Literal["validated", "needs_review", "failed"]
    input_summary: dict[str, str]
    demand_brief: DemandBrief
    question_contracts: list[QuestionContract]
    transfer_contracts: list[TransferContract]
    positioning_brief: PositioningBrief
    answers: list[DraftAnswer]
    validation: ValidationReport
    team_decisions: list[TeamDecision]
    eligibility_warnings: list[str]
    telemetry: RunTelemetry

