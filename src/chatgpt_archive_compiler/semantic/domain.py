"""Versioned, evidence-linked domain records for longitudinal organization and review."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import ConfigDict, Field

from chatgpt_archive_compiler.semantic.models import SemanticModel

SCHEMA_VERSION = "1.0"


class EvidenceKind(StrEnum):
    """How a conclusion was produced."""

    DETERMINISTIC = "deterministic"
    HEURISTIC = "heuristic"
    MODEL_GENERATED = "model-generated"


class EvidenceReference(SemanticModel):
    """Auditable source-derived support without hidden reasoning."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    schema_version: str = SCHEMA_VERSION
    conversation_id: str
    title: str
    timestamp: datetime | None = None
    excerpt: str = Field(max_length=2_000)
    relationship: str
    kind: EvidenceKind
    provider: str | None = None
    model: str | None = None
    archive_anchor: str | None = None


class ConversationRecord(SemanticModel):
    """Conversation features used by hierarchical and longitudinal stages."""

    schema_version: str = SCHEMA_VERSION
    conversation_id: str
    title: str
    timestamp: datetime | None = None
    theme_ids: tuple[str, ...] = ()
    primary_project_id: str | None = None
    entities: tuple[str, ...] = ()
    evidence: tuple[EvidenceReference, ...] = ()


class Subtheme(SemanticModel):
    """Leaf theme with multi-label conversation membership."""

    schema_version: str = SCHEMA_VERSION
    subtheme_id: str
    theme_id: str
    title: str
    conversation_ids: tuple[str, ...]


class Theme(SemanticModel):
    """Stable hierarchical theme beneath a broad domain."""

    schema_version: str = SCHEMA_VERSION
    theme_id: str
    domain: str
    title: str
    description: str
    subtheme_ids: tuple[str, ...] = ()
    conversation_ids: tuple[str, ...]
    evidence: tuple[EvidenceReference, ...] = ()


class TimelineEvent(SemanticModel):
    """Evidence-linked event ordered without implying causality."""

    schema_version: str = SCHEMA_VERSION
    event_id: str
    title: str
    occurred_at: datetime | None = None
    event_type: str
    evidence: tuple[EvidenceReference, ...] = Field(min_length=1)


class ProjectMilestone(TimelineEvent):
    """Candidate project milestone requiring human confirmation."""

    review_status: str = "unreviewed"


class Project(SemanticModel):
    """Longitudinal action thread distinct from a similarity cluster."""

    schema_version: str = SCHEMA_VERSION
    project_id: str
    title: str
    description: str
    conversation_ids: tuple[str, ...]
    milestone_ids: tuple[str, ...] = ()
    related_theme_ids: tuple[str, ...] = ()
    review_signal: float = Field(ge=0, le=1)
    evidence: tuple[EvidenceReference, ...] = ()


class Insight(SemanticModel):
    """A concise conclusion that must carry source evidence."""

    schema_version: str = SCHEMA_VERSION
    insight_id: str
    statement: str
    evidence: tuple[EvidenceReference, ...] = Field(min_length=1)
    review_status: str = "unreviewed"


class OpenLoop(SemanticModel):
    """Candidate unresolved task or question, never asserted as fact."""

    schema_version: str = SCHEMA_VERSION
    open_loop_id: str
    summary: str
    unresolved_reason: str
    evidence: tuple[EvidenceReference, ...] = Field(min_length=1)
    review_status: str = "unreviewed"


class ContradictionCandidate(SemanticModel):
    """Paired evidence for a possible contradiction, revision, or context change."""

    schema_version: str = SCHEMA_VERSION
    candidate_id: str
    classification: str
    summary: str
    earlier_evidence: tuple[EvidenceReference, ...] = Field(min_length=1)
    later_evidence: tuple[EvidenceReference, ...] = Field(min_length=1)
    review_status: str = "unreviewed"


class ReviewDecision(SemanticModel):
    """Explicit human review of a semantic object."""

    schema_version: str = SCHEMA_VERSION
    object_id: str
    decision: str
    timestamp: datetime
    note: str | None = None


class UserCorrection(SemanticModel):
    """Transparent supervision reapplied deterministically on later runs."""

    schema_version: str = SCHEMA_VERSION
    correction_type: str
    object_id: str
    old_value: object | None = None
    new_value: object
    timestamp: datetime
    source: str = "user"
