"""
SICPS Pydantic Schemas — Single source of truth for all agent data contracts.
Every agent reads/writes JSON that validates against these models.
REQ-030–033: Unified findings schema.
REQ-038–039: File-based coordination contract.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


# =============================================================================
# ENUMS
# =============================================================================

class Severity(str, Enum):
    critical = "critical"
    high     = "high"
    medium   = "medium"
    low      = "low"
    info     = "info"


class ClaimType(str, Enum):
    auto       = "auto"
    homeowners = "homeowners"
    medical    = "medical"


class Priority(str, Enum):
    urgent = "urgent"
    high   = "high"
    normal = "normal"


class RoutingDecision(str, Enum):
    auto_settle          = "auto_settle"
    siu_referral         = "siu_referral"
    coverage_denial      = "coverage_denial"
    late_notice_review   = "late_notice_review"
    total_loss_routing   = "total_loss_routing"
    cat_surge_processing = "cat_surge_processing"
    manual_review_route  = "manual_review_route"
    senior_review        = "senior_review"
    adjuster_review      = "adjuster_review"


class RiskLevel(str, Enum):
    critical = "critical"
    high     = "high"
    medium   = "medium"
    low      = "low"


# =============================================================================
# PRIMITIVES — REQ-015: bbox for every extracted field
# =============================================================================

class BoundingBox(BaseModel):
    """Source location of extracted text in a PDF. REQ-015."""
    model_config = ConfigDict(extra="forbid")

    page: int
    x1:   float
    y1:   float
    x2:   float
    y2:   float

    def as_evidence_pointer(self, doc: str) -> str:
        return f"{doc}:page{self.page}:bbox({self.x1},{self.y1},{self.x2},{self.y2})"


class ExtractedField(BaseModel):
    """A single field extracted from a document with confidence + traceability. REQ-014–015."""
    model_config = ConfigDict(extra="forbid")

    field_name:      str
    value:           Any
    confidence:      float
    bbox:            BoundingBox | None = None
    source_doc:      str
    source_page:     int = 1
    requires_review: bool = False

    @field_validator("confidence")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return round(v, 4)


# =============================================================================
# FINDINGS SCHEMA — REQ-030–033: used by ALL 6 agents
# =============================================================================

class Finding(BaseModel):
    """Universal finding record emitted by every agent. REQ-030–033."""
    model_config = ConfigDict(extra="forbid")

    finding_id:            str
    agent:                 str
    severity:              Severity
    confidence:            float
    evidence_links:        list[str]
    recommendation:        str
    open_questions:        list[str] = []
    requires_human_review: bool = False
    timestamp:             str

    @field_validator("confidence")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return round(v, 4)


# =============================================================================
# POLICY PACK — mirrors config/policy.yaml. REQ-034–037.
# extra="ignore" so future yaml additions never crash agents.
# =============================================================================

class ApprovalConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    auto_settle_max_amount:         float
    homeowners_auto_approve_limit:  float
    senior_review_threshold:        float
    total_loss_ratio_threshold:     float
    catastrophe_auto_approve_limit: float


class TolerancesConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    repair_cost_variance_pct:       float
    medical_cost_variance_pct:      float
    labor_rate_variance_pct:        float
    replacement_value_variance_pct: float


class FraudRing(BaseModel):
    model_config = ConfigDict(extra="ignore")
    ring_id:      str
    description:  str
    provider_ids: list[str]
    tax_ids:      list[str]


class BlacklistedProvider(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id:     str
    name:   str
    tax_id: str
    reason: str


class FraudConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    high_risk_threshold:               float
    medium_risk_threshold:             float
    low_risk_threshold:                float
    prior_claims_window_months:        int
    prior_claims_siu_trigger:          int
    provider_collusion_min_matches:    int
    staged_accident_vehicle_threshold: int
    inflated_repair_multiplier:        float
    known_fraud_rings:                 list[FraudRing]
    blacklisted_providers:             list[BlacklistedProvider]


class CoverageConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    late_notice_days_auto:       int
    late_notice_days_homeowners: int
    late_notice_days_medical:    int
    waiting_period_enforcement:  bool
    pre_existing_lookback_days:  int


class ExtractionConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    min_confidence_auto_process:   float
    low_confidence_flag_threshold: float
    ocr_field_completeness_min:    float


class CATEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    event_id:           str
    description:        str
    event_type:         str
    affected_states:    list[str]
    affected_zip_codes: list[str]
    active_from:        str
    active_until:       str


class CATEventConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    surge_mode_enabled:       bool
    surge_auto_approve_limit: float
    active_events:            list[CATEvent]


class PrivacyConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    mask_ssn:                bool
    mask_dob:                bool
    audit_log_pii_redaction: bool
    data_retention_days:     int


class StateOverride(BaseModel):
    model_config = ConfigDict(extra="ignore")
    late_notice_days: int


class ComplianceConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    jurisdiction:         str
    regulatory_framework: str
    state_overrides:      dict[str, StateOverride]


class PolicyPack(BaseModel):
    """Full config/policy.yaml deserialized. Agents load this, never hardcode values."""
    model_config = ConfigDict(extra="ignore")

    version:        str
    region:         str
    effective_date: str
    approval:       ApprovalConfig
    tolerances:     TolerancesConfig
    fraud:          FraudConfig
    coverage:       CoverageConfig
    extraction:     ExtractionConfig
    cat_event:      CATEventConfig
    privacy:        PrivacyConfig
    compliance:     ComplianceConfig


# =============================================================================
# MANIFEST SCHEMA — mirrors manifest.yaml. REQ-005–006.
# extra="ignore" — manifests have fraud_assessment/audit/ocr extra sections.
# =============================================================================

class ClaimantInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    claimant_id:       str
    full_name:         str
    policy_number:     str
    prior_claims_18mo: int
    prior_claim_ids:   list[str] = []
    fraud_history:     bool
    new_claimant:      bool


class PolicyInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    policy_number:    str
    policy_type:      str
    effective_date:   str
    expiration_date:  str
    is_active:        bool
    policy_expired:   bool
    deductible:       float
    coverage_limit:   float
    late_notice_days: int
    exclusions:       list[str] = []
    state:            str


class IncidentInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    date:                  str
    report_date:           str
    days_to_report:        int
    late_notice_violation: bool
    location:              str
    zip_code:              str
    type:                  str
    severity:              str
    cat_event:             bool
    cat_event_id:          str | None = None
    police_report_filed:   bool
    witness_count:         int = 0


class VehicleInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    make:         str | None = None
    model:        str | None = None
    year:         int | None = None
    vin:          str | None = None
    market_value: float | None = None


class FinancialInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    repair_estimate:  float
    medical_total:    float
    gross_amount:     float
    deductible:       float
    net_settlement:   float
    market_value:     float
    total_loss_ratio: float
    total_loss_flag:  bool
    injury_claim:     float = 0.0


class ProviderInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    provider_id:    str
    provider_name:  str
    tax_id:         str
    address:        str
    is_blacklisted: bool
    fraud_ring_id:  str | None = None


class DocumentRef(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type:                str
    file:                str
    processable:         bool
    expected_confidence: str


class ManifestFlags(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cat_event:          bool
    fraud_history:      bool
    new_claimant:       bool
    late_notice:        bool
    coverage_denied:    bool
    siu_referral:       bool
    manual_review:      bool
    total_loss:         bool
    missing_documents:  bool
    duplicate_claim:    bool
    policy_expired:     bool
    low_ocr_confidence: bool


class ManifestSchema(BaseModel):
    """Full claims bundle manifest.yaml. REQ-005–006."""
    model_config = ConfigDict(extra="ignore")

    schema_version:   str
    claim_id:         str
    scenario:         str
    claim_type:       str
    priority:         str
    expected_outcome: str
    expected_routing: str
    claimant:         ClaimantInfo
    policy:           PolicyInfo
    incident:         IncidentInfo
    vehicle:          VehicleInfo | None = None
    financials:       FinancialInfo
    provider:         ProviderInfo | None = None
    documents:        list[DocumentRef]
    flags:            ManifestFlags
    adjuster_notes:   str = ""


# =============================================================================
# AGENT A OUTPUT — runs/{id}/context_packet.json
# =============================================================================

class EvidenceIndexEntry(BaseModel):
    """Maps each document to the fields it provides. REQ-010."""
    model_config = ConfigDict(extra="forbid")

    doc_type:    str
    file:        str
    provides:    list[str]
    processable: bool
    page_count:  int = 1


class ContextPacket(BaseModel):
    """Agent A output. Single context object used by all downstream agents. REQ-008–011."""
    model_config = ConfigDict(extra="forbid")

    claim_id:       str
    run_id:         str
    claim_type:     str
    priority:       str
    claimant:       ClaimantInfo
    policy:         PolicyInfo
    incident:       IncidentInfo
    vehicle:        VehicleInfo | None = None
    financials:     FinancialInfo
    provider:       ProviderInfo | None = None
    flags:          ManifestFlags
    evidence_index: list[EvidenceIndexEntry]
    findings:       list[Finding] = []
    input_hash:     str
    created_at:     str


# =============================================================================
# AGENT B OUTPUT — runs/{id}/claim_summary.json
# =============================================================================

class ClaimSummary(BaseModel):
    """Agent B output — extracted fields with confidence + bbox. REQ-012–016."""
    model_config = ConfigDict(extra="forbid")

    claim_id:               str
    extracted_fields:       list[ExtractedField]
    overall_confidence:     float
    requires_manual_review: bool
    low_confidence_fields:  list[str]
    extraction_errors:      list[str]
    findings:               list[Finding] = []

    # Denormalized for downstream agents — avoids re-parsing extracted_fields
    claimant_name:     str
    policy_number:     str
    claim_type:        str
    incident_date:     str
    repair_estimate:   float
    medical_total:     float
    gross_amount:      float
    market_value:      float
    days_to_report:    int
    zip_code:          str
    prior_claims_18mo: int
    vehicle_vin:       str | None = None
    provider_id:       str | None = None
    provider_tax_id:   str | None = None

    @field_validator("overall_confidence")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("overall_confidence must be 0.0–1.0")
        return round(v, 4)


# =============================================================================
# AGENT C OUTPUT — runs/{id}/coverage_result.json
# =============================================================================

class CoverageResult(BaseModel):
    """Agent C output — coverage validation findings. REQ-017–018."""
    model_config = ConfigDict(extra="forbid")

    claim_id:                 str
    coverage_active:          bool
    policy_expired:           bool
    denial_reason:            str | None = None
    late_notice_violation:    bool
    late_notice_days_allowed: int
    days_to_report:           int
    exclusions_triggered:     list[str]
    deductible:               float
    coverage_limit:           float
    waiting_period_active:    bool
    applicable_state:         str
    findings:                 list[Finding] = []


# =============================================================================
# AGENT D OUTPUT — runs/{id}/settlement_calc.json (preliminary)
# =============================================================================

class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str
    amount:      float
    category:    str


class SettlementCalc(BaseModel):
    """Agent D preliminary output, finalized by Agent H. REQ-019–020."""
    model_config = ConfigDict(extra="forbid")

    claim_id:                 str
    gross_amount:             float
    deductible:               float
    net_settlement:           float
    line_items:               list[LineItem]
    repair_estimate:          float
    medical_total:            float
    replacement_value:        float = 0.0
    market_value:             float
    total_loss_ratio:         float
    total_loss_flag:          bool
    repair_variance_pct:      float
    medical_variance_pct:     float
    repair_within_tolerance:  bool
    medical_within_tolerance: bool
    findings:                 list[Finding] = []
    finalized:                bool = False


# =============================================================================
# AGENT E OUTPUT — runs/{id}/fraud_score.json
# =============================================================================

class FraudIndicator(BaseModel):
    model_config = ConfigDict(extra="forbid")
    indicator: str
    weight:    float
    evidence:  str


class FraudScore(BaseModel):
    """Agent E output — fraud probability + SIU signals. REQ-021–025."""
    model_config = ConfigDict(extra="forbid")

    claim_id:             str
    fraud_score:          float
    risk_level:           RiskLevel
    siu_referral:         bool
    adjuster_review:      bool
    provider_blacklisted: bool
    provider_ring_match:  bool
    provider_ring_id:     str | None = None
    prior_claims_trigger: bool
    billing_variance_pct: float
    indicators:           list[FraudIndicator]
    evidence_links:       list[str]
    findings:             list[Finding] = []

    @field_validator("fraud_score")
    @classmethod
    def score_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("fraud_score must be 0.0–1.0")
        return round(v, 4)


# =============================================================================
# AGENT H OUTPUT — exceptions.md, approval_packet.json, metrics.json
# =============================================================================

class ExceptionEntry(BaseModel):
    """Single exception item. REQ-026."""
    model_config = ConfigDict(extra="forbid")

    exception_id:       str
    category:           str
    severity:           Severity
    description:        str
    recommended_action: str
    evidence_links:     list[str]
    assigned_to:        str | None = None


class ApprovalPacket(BaseModel):
    """Agent H final output — routing decision + full evidence bundle. REQ-026–028."""
    model_config = ConfigDict(extra="forbid")

    claim_id:         str
    run_id:           str
    routing_decision: RoutingDecision
    net_settlement:   float
    adjuster_notes:   str
    exceptions:       list[ExceptionEntry]
    all_findings:     list[Finding]
    evidence_bundle:  list[str]
    input_hash:       str
    finalized_at:     str


class RunMetrics(BaseModel):
    """Pipeline run metrics. REQ-029, REQ-042."""
    model_config = ConfigDict(extra="forbid")

    claim_id:                   str
    run_id:                     str
    pipeline_duration_seconds:  float
    agents_executed:            list[str]
    total_findings:             int
    findings_by_severity:       dict[str, int]
    extraction_field_count:     int
    extraction_accuracy:        float
    low_confidence_field_count: int
    exception_count:            int
    routing_decision:           str
    input_hash:                 str
    cache_hit:                  bool
