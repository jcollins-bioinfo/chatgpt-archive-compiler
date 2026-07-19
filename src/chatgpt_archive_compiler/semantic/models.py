"""Typed records used by the semantic-atlas pipeline and provider adapters."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class SemanticModel(BaseModel):
    """Strict base model for durable semantic-atlas artifacts."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class SemanticAtlasOptions(SemanticModel):
    """Configuration for representation, graph, selective refinement, cache, and review."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    include_reasoning: bool = False
    max_conversations: int | None = Field(default=None, gt=0)
    max_characters_per_conversation: int = Field(default=32_000, ge=1_000)
    embedding_batch_size: int = Field(default=64, gt=0, le=2_048)
    analysis_batch_size: int = Field(default=8, gt=0, le=128)
    graph_brute_force_threshold: int = Field(default=256, gt=1)
    graph_candidate_dimensions: int = Field(default=12, gt=0, le=128)
    max_neighbors: int = Field(default=10, gt=0, le=100)
    min_similarity: float = Field(default=0.32, ge=-1.0, le=1.0)
    community_resolution: float = Field(default=1.0, gt=0.0, le=10.0)
    max_leaf_categories: int = Field(default=64, ge=1, le=512)
    refined_conversations_per_category: int = Field(default=3, ge=1, le=16)
    max_refined_conversations: int = Field(default=0, ge=0, le=4_096)
    review_confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    cache_enabled: bool = True


class ConversationRepresentation(SemanticModel):
    """Deterministic, current-path representation supplied to semantic providers."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    conversation_key: str = Field(min_length=16)
    source_index: int = Field(ge=0)
    source_conversation_id: str | None = None
    title: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    text: str
    message_count: int = Field(ge=0)
    character_count: int = Field(ge=0)
    original_character_count: int = Field(ge=0)
    truncated: bool = False
    text_sha256: str = Field(min_length=64, max_length=64)


class SemanticRunEstimate(SemanticModel):
    """Content-free size estimate suitable for display before any provider call."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    conversation_count: int = Field(ge=0)
    message_count: int = Field(ge=0)
    character_count: int = Field(ge=0)
    original_character_count: int = Field(ge=0)
    approximate_input_tokens: int = Field(ge=0)
    embedding_item_count: int = Field(ge=0)
    analysis_item_count: int = Field(ge=0)
    truncated_conversation_count: int = Field(ge=0)


class EmbeddingRecord(SemanticModel):
    """One provider-generated vector joined to a conversation by opaque key."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    conversation_key: str
    vector: tuple[float, ...] = Field(min_length=1)


class ConversationAnalysis(SemanticModel):
    """Rich multi-axis semantic profile for one conversation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    conversation_key: str
    synopsis: str
    primary_subject: str
    secondary_subjects: tuple[str, ...] = ()
    projects: tuple[str, ...] = ()
    conversation_type: str
    entities: tuple[str, ...] = ()
    goals: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    recurring_themes: tuple[str, ...] = ()
    temporal_role: str | None = None
    related_conversation_keys: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)


class GraphEdge(SemanticModel):
    """Weighted undirected semantic relationship between two conversations."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    source_key: str
    target_key: str
    similarity: float = Field(ge=-1.0, le=1.0)
    relation_type: str = "semantic_similarity"
    shared_signals: tuple[str, ...] = ()


class CategoryDraft(SemanticModel):
    """Local community supplied to the interpretation provider for naming."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    category_id: str
    conversation_keys: tuple[str, ...]
    representative_conversation_keys: tuple[str, ...]
    candidate_labels: tuple[str, ...]
    earliest_at: datetime | None = None
    latest_at: datetime | None = None


class TaxonomyInterpretationRequest(SemanticModel):
    """Provider request containing local clusters and content-light profiles."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    clusters: tuple[CategoryDraft, ...]
    analyses: tuple[ConversationAnalysis, ...]


class CategoryInterpretation(SemanticModel):
    """Human-readable interpretation of one locally discovered category."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    category_id: str
    name: str
    description: str
    parent_name: str
    defining_concepts: tuple[str, ...] = ()
    related_category_ids: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)


class TaxonomyInterpretation(SemanticModel):
    """Provider response naming every local community and its broad parent."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    categories: tuple[CategoryInterpretation, ...]


class Category(SemanticModel):
    """One node in the final hierarchical archive taxonomy."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    category_id: str
    name: str
    description: str
    parent_id: str | None = None
    level: int = Field(ge=0)
    conversation_keys: tuple[str, ...] = ()
    child_ids: tuple[str, ...] = ()
    defining_concepts: tuple[str, ...] = ()
    related_category_ids: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)


class CategoryAssignment(SemanticModel):
    """Primary and secondary taxonomy placement for one conversation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    conversation_key: str
    primary_category_id: str
    secondary_category_ids: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str | None = None


class Taxonomy(SemanticModel):
    """Complete hierarchical category set and conversation assignments."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    categories: tuple[Category, ...]
    assignments: tuple[CategoryAssignment, ...]


class CategoryProfile(SemanticModel):
    """Archive-level interpretation of one category and its development."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    category_id: str
    overview: str
    characteristic_questions: tuple[str, ...] = ()
    major_conclusions: tuple[str, ...] = ()
    unresolved_threads: tuple[str, ...] = ()
    temporal_evolution: str | None = None
    representative_conversation_keys: tuple[str, ...] = ()


class TimelineEvent(SemanticModel):
    """One semantic milestone in a recurring project's history."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    label: str
    summary: str
    conversation_keys: tuple[str, ...]
    occurred_at: datetime | None = None


class ProjectTimeline(SemanticModel):
    """Chronological synthesis of a project spanning multiple conversations."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    project_name: str
    overview: str
    events: tuple[TimelineEvent, ...]
    current_state: str | None = None
    unresolved_work: tuple[str, ...] = ()


class ArchiveSynthesis(SemanticModel):
    """Global interpretation of the archive across categories and time."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    executive_summary: str
    dominant_domains: tuple[str, ...] = ()
    cross_domain_connections: tuple[str, ...] = ()
    recurring_patterns: tuple[str, ...] = ()
    temporal_evolution: tuple[str, ...] = ()
    tensions_and_reversals: tuple[str, ...] = ()
    dormant_threads: tuple[str, ...] = ()
    opportunities: tuple[str, ...] = ()


class GraphSummary(SemanticModel):
    """Content-free graph statistics supplied to global synthesis."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    category_count: int = Field(ge=0)
    isolated_node_count: int = Field(ge=0)


class ConversationReference(SemanticModel):
    """Compact dated pointer used to ground timelines without resending source text."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    conversation_key: str
    title: str
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ArchiveSynthesisRequest(SemanticModel):
    """Provider request for category profiles, project timelines, and global synthesis."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    analyses: tuple[ConversationAnalysis, ...]
    references: tuple[ConversationReference, ...]
    taxonomy: Taxonomy
    graph_summary: GraphSummary


class ArchiveSynthesisBundle(SemanticModel):
    """Structured provider response for all archive-level interpretive products."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    category_profiles: tuple[CategoryProfile, ...]
    project_timelines: tuple[ProjectTimeline, ...]
    synthesis: ArchiveSynthesis


class ReviewCode(StrEnum):
    """Stable reason that an item needs human review."""

    LOW_ANALYSIS_CONFIDENCE = "low_analysis_confidence"
    LOW_CATEGORY_CONFIDENCE = "low_category_confidence"
    AMBIGUOUS_ASSIGNMENT = "ambiguous_assignment"
    ISOLATED_CONVERSATION = "isolated_conversation"
    SINGLETON_CATEGORY = "singleton_category"
    CONSOLIDATED_CATEGORY = "consolidated_category"


class ReviewPriority(StrEnum):
    """Relative human-review priority."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReviewItem(SemanticModel):
    """Privacy-safe pointer to a classification condition needing attention."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    code: ReviewCode
    priority: ReviewPriority
    conversation_keys: tuple[str, ...] = ()
    category_ids: tuple[str, ...] = ()
    message: str
    diagnostics: dict[str, JsonValue] = Field(default_factory=dict)


class ReviewQueue(SemanticModel):
    """All deterministic review signals emitted by one semantic run."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    items: tuple[ReviewItem, ...]
    counts_by_code: dict[str, int]
    counts_by_priority: dict[str, int]


class SemanticAtlas(SemanticModel):
    """Complete in-memory semantic result used by renderers and downstream tools."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    atlas_version: str = "1.0"
    representations: tuple[ConversationRepresentation, ...]
    analyses: tuple[ConversationAnalysis, ...]
    embeddings: tuple[EmbeddingRecord, ...]
    graph_edges: tuple[GraphEdge, ...]
    taxonomy: Taxonomy
    category_profiles: tuple[CategoryProfile, ...]
    project_timelines: tuple[ProjectTimeline, ...]
    synthesis: ArchiveSynthesis
    review_queue: ReviewQueue


class ArtifactRecord(SemanticModel):
    """Integrity metadata for one generated semantic-atlas artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    filename: str
    sha256: str
    size_bytes: int = Field(ge=0)


class SemanticAtlasManifest(SemanticModel):
    """Privacy-safe provenance, options, counts, and checksums for one run."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    atlas_version: str
    package_version: str
    created_at: datetime
    source_archive_version: str
    source_fingerprint: str
    embedding_provider: str
    embedding_model: str
    analysis_provider: str
    analysis_model: str
    options: dict[str, JsonValue]
    estimate: SemanticRunEstimate
    graph_summary: GraphSummary
    review_counts_by_code: dict[str, int]
    artifacts: tuple[ArtifactRecord, ...]


class SemanticAtlasResult(SemanticModel):
    """In-memory atlas plus paths to all exported products."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    output_directory: Path
    catalog_path: Path
    embeddings_path: Path
    graph_path: Path
    taxonomy_path: Path
    category_profiles_path: Path
    project_timelines_path: Path
    synthesis_path: Path
    review_queue_path: Path
    atlas_html_path: Path
    manifest_path: Path
    estimate: SemanticRunEstimate
    atlas: SemanticAtlas
