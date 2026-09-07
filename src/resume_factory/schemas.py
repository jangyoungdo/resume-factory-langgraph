from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, model_validator


class ExecutionMode(StrEnum):
    ECONOMY = "economy"
    BALANCED = "balanced"
    PREMIUM = "premium"


class ModelTier(StrEnum):
    LOCAL = "local"
    LUNA = "luna"
    TERRA = "terra"
    SOL = "sol"


class CallKind(StrEnum):
    SPECIALIST = "specialist"
    CRITIC = "critic"
    LEAD = "lead"
    ADJUDICATOR = "adjudicator"
    CHARACTER_REWRITE = "character_rewrite"
    SEMANTIC_CRITIC = "semantic_critic"
    SEMANTIC_REPAIR = "semantic_repair"


class UsageStatus(StrEnum):
    REPORTED = "reported"
    OFFLINE = "offline"
    MISSING = "missing"
    LEGACY = "legacy"


class BackendProvider(StrEnum):
    LOCAL = "local"
    OPENAI_API = "openai_api"
    CODEX_CLI = "codex_cli"


class BillingMode(StrEnum):
    OFFLINE = "offline"
    API = "api"
    CHATGPT_SUBSCRIPTION = "chatgpt_subscription"


class CostStatus(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


class FeedbackDecision(StrEnum):
    ACCEPTED = "accepted"
    REVISED = "revised"
    REJECTED = "rejected"


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


class QuestionArchetype(StrEnum):
    COMPETENCY_EFFORT = "competency_effort"
    TEAMWORK_ROLE = "teamwork_role"
    CONTRIBUTION_TRANSFER = "contribution_transfer"
    LEARNING_TRANSFER = "learning_transfer"
    MOTIVATION_FIT = "motivation_fit"
    GROWTH_VALUES = "growth_values"
    FREEFORM = "freeform"


class MaterialSelectionMode(StrEnum):
    AUTO_UNIQUE = "auto_unique"
    PINNED = "pinned"


class SentencePlan(BaseModel):
    sentence_id: str
    text: str
    role: SentenceRole
    selling_point: str
    evidence_event_id: str | None = None
    claim_ids: list[str] = Field(default_factory=list)
    company_connection: str | None = None
    interview_defensible: bool = False


class PrepSoaraStructure(BaseModel):
    p: str = ""
    r: str = ""
    e_soara: str = ""
    p2: str = ""


class DraftProposal(BaseModel):
    question_id: str
    headline: str
    direct_answer: str
    prep_soara_structure: PrepSoaraStructure = Field(default_factory=PrepSoaraStructure)
    sentence_plans: list[SentencePlan]
    evidence_ids: list[str]
    company_transfer: str
    interview_defense: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class ApplicationQuestion(BaseModel):
    question_id: str
    text: str
    character_limit: int = Field(default=600, ge=100, le=5000)
    required: bool = True
    supporting_evidence_ids: list[str] = Field(default_factory=list)


class NumericAuthority(BaseModel):
    authority_id: str
    value: float | int | str
    unit: str | None = None
    status: Literal["verified", "unverified"] = "verified"


class EvidencePacket(BaseModel):
    event_id: str
    # Legacy artifacts fall back to event_id. New auto-unique intake requires an
    # explicit immutable key and validates it before any model call.
    experience_key: str = ""
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
    best_for_questions: list[str] = Field(default_factory=list)


class ApplicationInput(BaseModel):
    company: str
    job: str
    industry: str = "manufacturing"
    job_description: str
    company_context: str
    questions: list[ApplicationQuestion]
    evidence: list[EvidencePacket]
    eligibility_notes: list[str] = Field(default_factory=list)
    existing_draft: str | None = None
    previous_outcomes: list[str] = Field(default_factory=list)
    material_selection_mode: MaterialSelectionMode = MaterialSelectionMode.PINNED
    pinned_evidence_ids_by_question: dict[str, str] = Field(default_factory=dict)
    preferred_evidence_ids_by_question: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_questions_and_evidence(self) -> ApplicationInput:
        if not self.questions:
            raise ValueError("at least one application question is required")
        if not self.evidence:
            raise ValueError("at least one evidence packet is required")
        known_evidence = {item.event_id for item in self.evidence}
        missing_support = {
            evidence_id
            for question in self.questions
            for evidence_id in question.supporting_evidence_ids
            if evidence_id not in known_evidence
        }
        if missing_support:
            raise ValueError(
                "unknown supporting evidence IDs: " + ", ".join(sorted(missing_support))
            )
        for evidence in self.evidence:
            if not evidence.experience_key:
                evidence.experience_key = evidence.event_id
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
    draft: DraftProposal | None = None
    drafts: list[DraftProposal] = Field(default_factory=list)
    structured_payload: AgentStructuredPayload | None = None


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
    selected_draft: DraftProposal | None = None
    structured_payload: AgentStructuredPayload | None = None


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
    character_limit: int | None = None
    character_hard_min: int | None = None
    character_target_min: int | None = None
    character_target_max: int | None = None
    counting_policy: Literal["headline_newline_body"] = "headline_newline_body"

    @model_validator(mode="after")
    def populate_character_bounds(self) -> QuestionContract:
        import math

        limit = self.character_limit or self.character_budget
        self.character_limit = limit
        self.character_hard_min = self.character_hard_min or math.ceil(limit * 0.95)
        self.character_target_min = self.character_target_min or math.ceil(limit * 0.97)
        self.character_target_max = self.character_target_max or math.floor(limit * 0.98)
        return self


class QuestionNarrativeContract(QuestionContract):
    archetype: QuestionArchetype = QuestionArchetype.FREEFORM
    direct_answer: str = ""
    core_message: str = ""
    required_elements: list[str] = Field(default_factory=list)
    optional_elements: list[str] = Field(default_factory=list)
    forbidden_detours: list[str] = Field(default_factory=list)
    preferred_evidence_traits: list[str] = Field(default_factory=list)
    narrative_sequence: list[str] = Field(default_factory=list)


class MaterialAssignment(BaseModel):
    question_id: str
    primary_event_id: str
    primary_experience_key: str
    supporting_event_id: str | None = None
    allocation_score: float = Field(ge=0, le=5)
    selection_reason: str
    rejected_candidates: list[str] = Field(default_factory=list)


class MaterialPortfolioPlan(BaseModel):
    assignments: list[MaterialAssignment] = Field(default_factory=list)
    unassigned_questions: list[str] = Field(default_factory=list)
    duplicate_experience_keys: list[str] = Field(default_factory=list)
    status: Literal["ready", "blocked_insufficient_evidence"] = "ready"
    code: str | None = None
    missing_questions: list[str] = Field(default_factory=list)
    required_evidence_traits: dict[str, list[str]] = Field(default_factory=dict)
    searched_candidate_count: int = 0


class EditorialAssessment(BaseModel):
    question_id: str
    inferred_takeaway: str
    question_directness: int = Field(ge=1, le=5)
    thesis_clarity: int = Field(ge=1, le=5)
    logical_continuity: int = Field(ge=1, le=5)
    evidence_to_claim: int = Field(ge=1, le=5)
    effort_or_action_specificity: int = Field(ge=1, le=5)
    company_transfer: int = Field(ge=1, le=5)
    redundant_sentence_ids: list[str] = Field(default_factory=list)
    low_value_sentence_ids: list[str] = Field(default_factory=list)
    logical_gap_after_sentence_ids: list[str] = Field(default_factory=list)
    excessive_technical_detail_ids: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    verdict: Literal["pass", "repair", "replan"]

    @property
    def passed(self) -> bool:
        scores = (
            self.question_directness,
            self.thesis_clarity,
            self.logical_continuity,
            self.evidence_to_claim,
            self.effort_or_action_specificity,
            self.company_transfer,
        )
        issue_ids = (
            self.redundant_sentence_ids
            + self.low_value_sentence_ids
            + self.logical_gap_after_sentence_ids
            + self.excessive_technical_detail_ids
        )
        return (
            self.verdict == "pass"
            and min(scores) >= 4
            and not issue_ids
            and not self.missing_information
        )


class EditOperation(BaseModel):
    operation: Literal[
        "replace_sentence",
        "delete_sentence",
        "insert_after",
        "merge_sentences",
        "reorder_span",
    ]
    target_sentence_ids: list[str] = Field(min_length=1)
    text: str | None = None
    role: SentenceRole | None = None
    selling_point: str | None = None
    evidence_event_id: str | None = None
    company_connection: str | None = None


class AgentStructuredPayload(BaseModel):
    demand_brief: DemandBrief | None = None
    question_narratives: list[QuestionNarrativeContract] = Field(default_factory=list)
    editorial_assessment: EditorialAssessment | None = None
    edit_operations: list[EditOperation] = Field(default_factory=list)


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


class DraftAnswer(BaseModel):
    question_id: str
    headline: str
    body: str
    sentence_plans: list[SentencePlan]
    evidence_ids: list[str]
    character_limit: int | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def submission_text(self) -> str:
        headline = self.headline.strip().replace("\r\n", "\n").replace("\r", "\n")
        body = self.body.strip().replace("\r\n", "\n").replace("\r", "\n")
        return f"{headline}\n{body}" if headline else body

    @computed_field  # type: ignore[prop-decorator]
    @property
    def character_count(self) -> int:
        return len(self.submission_text)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def utilization_ratio(self) -> float | None:
        if not self.character_limit:
            return None
        return round(self.character_count / self.character_limit, 4)


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
    run_id: str = "legacy"
    call_id: str = "legacy"
    sequence: int = 0
    question_id: str | None = None
    team: str = "legacy"
    agent_role: str
    call_kind: CallKind = CallKind.SPECIALIST
    tier: ModelTier
    model: str
    input_tokens: int
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    output_tokens: int
    reasoning_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None
    price_catalog_version: str | None = None
    latency_ms: int
    retry_count: int | None = None
    usage_status: UsageStatus = UsageStatus.LEGACY
    success: bool = True
    error_code: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    provider: BackendProvider = BackendProvider.LOCAL
    billing_mode: BillingMode = BillingMode.OFFLINE
    cost_status: CostStatus = CostStatus.NOT_APPLICABLE
    provider_run_id: str | None = None
    queued_at: datetime | None = None
    process_started_at: datetime | None = None
    queue_latency_ms: int = 0
    first_event_latency_ms: int | None = None
    provider_execution_ms: int | None = None

    @model_validator(mode="after")
    def populate_legacy_total_tokens(self) -> ModelCallRecord:
        if self.total_tokens == 0 and self.input_tokens + self.output_tokens > 0:
            self.total_tokens = self.input_tokens + self.output_tokens
        if (
            self.estimated_cost_usd is None
            and self.usage_status is UsageStatus.MISSING
            and self.billing_mode is not BillingMode.CHATGPT_SUBSCRIPTION
        ):
            self.cost_status = CostStatus.UNKNOWN
        return self


class RunPhaseSpan(BaseModel):
    phase: str
    started_at: datetime
    completed_at: datetime
    duration_ms: int = Field(ge=0)
    status: Literal["completed", "failed", "skipped"] = "completed"
    detail: str | None = None


class RunTelemetry(BaseModel):
    run_id: str
    mode: ExecutionMode
    model_calls: list[ModelCallRecord] = Field(default_factory=list)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    total_cost_usd: float = 0
    cost_complete: bool = True
    price_catalog_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    provider: BackendProvider = BackendProvider.LOCAL
    billing_mode: BillingMode = BillingMode.OFFLINE
    cost_status: CostStatus = CostStatus.NOT_APPLICABLE
    graph_version: str = "v0.4"
    graph_nodes_completed: list[str] = Field(default_factory=list)
    base_call_budget: int = 0
    optional_calls_used: int = 0
    hard_call_cap: int = 0
    command_started_at: datetime | None = None
    completed_at: datetime | None = None
    wall_time_ms: int | None = None
    phase_spans: list[RunPhaseSpan] = Field(default_factory=list)
    network_status: Literal["healthy", "degraded", "unknown"] = "unknown"


class RunResult(BaseModel):
    run_id: str
    status: Literal[
        "validated",
        "needs_review",
        "failed",
        "network_degraded",
        "blocked_insufficient_evidence",
    ]
    input_summary: dict[str, str]
    demand_brief: DemandBrief
    question_contracts: list[QuestionNarrativeContract]
    transfer_contracts: list[TransferContract]
    positioning_brief: PositioningBrief
    answers: list[DraftAnswer]
    validation: ValidationReport
    team_decisions: list[TeamDecision]
    eligibility_warnings: list[str]
    telemetry: RunTelemetry
    material_plan: MaterialPortfolioPlan | None = None
    editorial_assessments: list[EditorialAssessment] = Field(default_factory=list)


class SubmissionAnswer(BaseModel):
    question_id: str
    prompt: str
    character_limit: int
    headline: str = ""
    body: str
    evidence_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def submission_text(self) -> str:
        headline = self.headline.strip().replace("\r\n", "\n").replace("\r", "\n")
        body = self.body.strip().replace("\r\n", "\n").replace("\r", "\n")
        return f"{headline}\n{body}" if headline else body

    @computed_field  # type: ignore[prop-decorator]
    @property
    def character_count(self) -> int:
        return len(self.submission_text)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def utilization_ratio(self) -> float:
        return round(self.character_count / self.character_limit, 4)


class SubmissionBundle(BaseModel):
    company: str
    job: str
    revision: int = Field(ge=1)
    source_run_id: str
    status: Literal["user_review", "approved"] = "user_review"
    answers: list[SubmissionAnswer]
    eligibility_warnings: list[str] = Field(default_factory=list)
    actual_submission_performed: bool = False


class HumanFeedback(BaseModel):
    run_id: str
    decision: FeedbackDecision
    rating: int = Field(ge=1, le=5)
    final_draft_path: str | None = None
    final_draft_hash: str | None = None
    overall_edit_ratio: float | None = Field(default=None, ge=0, le=1)
    per_question_edit_ratio: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


AgentProposal.model_rebuild()
TeamDecision.model_rebuild()
