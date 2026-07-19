"""Typeset a semantic atlas as a self-contained, category-organized book.

The renderer deliberately depends only on the stable :class:`~chatgpt_archive_compiler.models.Archive`
IR and a mapping-like semantic atlas.  It does not depend on a particular embedding, clustering,
or model provider.  This keeps semantic analysis replaceable while giving every analysis backend
the same polished publication target.
"""

# ruff: noqa: E501 -- Embedded print CSS remains readable as complete declarations.

from __future__ import annotations

import hashlib
import html
import importlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import IO, Protocol, TypeAlias, cast
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from pydantic import BaseModel, ConfigDict, Field

from chatgpt_archive_compiler.exceptions import ArchiveCompilationError
from chatgpt_archive_compiler.models import (
    Archive,
    ContentBlock,
    ContentBlockType,
    Conversation,
    Message,
    Role,
)

AtlasInput: TypeAlias = Mapping[str, object] | BaseModel
"""A semantic-atlas model or its JSON-compatible mapping representation."""


class _PDFDocument(Protocol):
    """Structural type for a configured WeasyPrint document."""

    def write_pdf(self, target: str) -> object:
        """Write the rendered document to ``target``."""


class _HTMLFactory(Protocol):
    """Structural type for the small optional WeasyPrint surface used here."""

    def __call__(
        self,
        *,
        filename: str,
        base_url: str,
        url_fetcher: Callable[..., object],
    ) -> _PDFDocument:
        """Construct a local HTML document with a restricted fetcher."""


class BookPaperSize(StrEnum):
    """Supported print page sizes for the semantic book."""

    TRADE = "trade"
    A4 = "a4"


class SemanticBookOptions(BaseModel):
    """Publication and rendering options for a semantic-atlas book.

    Attributes
    ----------
    title
        Title printed on the cover, title page, and running folios.
    subtitle
        Optional explanatory line beneath the title.
    author
        Optional author or archive-owner credit.
    edition
        Optional edition label, such as ``"First semantic edition"``.
    paper_size
        Trade-book or A4 page geometry used for print CSS and PDF output.
    include_transcripts
        Whether each catalog entry includes its complete visible-path user/assistant transcript.
        This is disabled by default because thousands of full transcripts can exceed a Colab PDF
        layout process's memory; the semantic catalog remains complete without duplicating them.
    include_message_timestamps
        Whether transcript messages include their UTC timestamps.
    render_pdf
        Whether to invoke the optional WeasyPrint dependency after writing HTML.
    max_conversations
        Optional deterministic limit for previews and test editions.
    max_related_conversations
        Maximum number of cross-conversation links printed beneath each catalog entry.
    max_synopsis_entries_per_category
        Maximum number of representative or high-confidence conversation profiles expanded in
        each category. Every selected conversation still appears in its category directory. Set
        this to ``None`` only for a deliberately complete, potentially very large HTML edition.
    max_expanded_conversations
        Global print budget applied fairly across categories after the per-category cap. This
        prevents a broad taxonomy from expanding most of a large archive despite each individual
        category remaining under its local limit.
    max_project_timelines
        Maximum number of recurring projects rendered in the book.
    max_timeline_events_per_project
        Maximum milestones rendered for each recurring project.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    title: str = "A Semantic Atlas of ChatGPT Conversations"
    subtitle: str | None = "Subjects, projects, connections, and intellectual trajectories"
    author: str | None = None
    edition: str | None = "First semantic edition"
    paper_size: BookPaperSize = BookPaperSize.TRADE
    include_transcripts: bool = False
    include_message_timestamps: bool = False
    render_pdf: bool = False
    max_conversations: int | None = Field(default=None, gt=0)
    max_related_conversations: int = Field(default=8, ge=0, le=50)
    max_synopsis_entries_per_category: int | None = Field(default=12, gt=0)
    max_expanded_conversations: int | None = Field(default=240, gt=0)
    max_project_timelines: int | None = Field(default=24, gt=0)
    max_timeline_events_per_project: int | None = Field(default=12, gt=0)


class SemanticBookResult(BaseModel):
    """Paths, checksums, and non-content counts for one rendered semantic book."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    output_directory: Path
    html_path: Path
    html_sha256: str
    pdf_path: Path | None = None
    pdf_sha256: str | None = None
    manifest_path: Path
    category_count: int = Field(ge=0)
    conversation_count: int = Field(ge=0)
    expanded_conversation_count: int = Field(ge=0)
    unclassified_conversation_count: int = Field(ge=0)


@dataclass(frozen=True)
class _Relation:
    """One normalized directed semantic relationship between conversations."""

    target_id: str
    label: str | None = None
    score: float | None = None


@dataclass(frozen=True)
class _Category:
    """One normalized category independent of the semantic engine's schema."""

    category_id: str
    name: str
    summary: str | None
    parent_id: str | None
    conversation_ids: tuple[str, ...]
    representative_conversation_ids: tuple[str, ...]
    related_category_ids: tuple[str, ...]
    themes: tuple[str, ...]
    characteristic_questions: tuple[str, ...]
    major_conclusions: tuple[str, ...]
    unresolved_threads: tuple[str, ...]
    trajectory: str | None
    source_order: int


@dataclass(frozen=True)
class _CatalogEntry:
    """Semantic annotations for one Archive conversation."""

    conversation_id: str
    synopsis: str | None
    primary_category_id: str | None
    category_ids: tuple[str, ...]
    related: tuple[_Relation, ...]
    projects: tuple[str, ...]
    subjects: tuple[str, ...]
    themes: tuple[str, ...]
    entities: tuple[str, ...]
    goals: tuple[str, ...]
    decisions: tuple[str, ...]
    unresolved_questions: tuple[str, ...]
    status: str | None
    confidence: float | None


@dataclass(frozen=True)
class _SynthesisSection:
    """A heading and Markdown body from the cross-archive synthesis."""

    title: str
    body: str


@dataclass(frozen=True)
class _TimelineEvent:
    """One dated or undated milestone in a recurring project."""

    label: str
    summary: str
    conversation_ids: tuple[str, ...]
    occurred_at: datetime | None


@dataclass(frozen=True)
class _ProjectTimeline:
    """One archive-spanning project narrative and its ordered milestones."""

    name: str
    overview: str
    events: tuple[_TimelineEvent, ...]
    current_state: str | None
    unresolved_work: tuple[str, ...]


@dataclass(frozen=True)
class _NormalizedAtlas:
    """Renderer-owned normalized projection of any supported atlas mapping."""

    categories: tuple[_Category, ...]
    catalog: Mapping[str, _CatalogEntry]
    synthesis: tuple[_SynthesisSection, ...]
    project_timelines: tuple[_ProjectTimeline, ...]
    conversation_keys_by_source_index: Mapping[int, str]


@dataclass(frozen=True)
class _PlacedConversation:
    """An Archive conversation and its resolved semantic placement."""

    conversation_id: str
    conversation: Conversation
    annotation: _CatalogEntry
    primary_category_id: str


_MARKDOWN = MarkdownIt("commonmark", {"html": False, "breaks": True})
_UNCLASSIFIED_ID = "__unclassified__"


@contextmanager
def _atomic_text_writer(destination: Path) -> Iterator[IO[str]]:
    """Yield a UTF-8 writer that replaces ``destination`` atomically on success."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            yield cast(IO[str], handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    except OSError as exc:
        raise ArchiveCompilationError(
            f"Could not write semantic-book artifact: {destination}"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of one generated artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_text(value: str) -> str:
    """Normalize separators known to trigger PDF layout assertions."""

    return value.replace("\u2028", "\n").replace("\u2029", "\n\n")


def _as_mapping(value: object) -> Mapping[str, object] | None:
    """Return a string-keyed mapping view of a mapping or Pydantic model."""

    if isinstance(value, BaseModel):
        return cast(Mapping[str, object], value.model_dump(mode="python"))
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return cast(Mapping[str, object], value)
    return None


def _as_sequence(value: object) -> Sequence[object] | None:
    """Return a non-text sequence view when available."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return cast(Sequence[object], value)
    return None


def _first(mapping: Mapping[str, object], *keys: str) -> object | None:
    """Return the first present, non-``None`` value among ``keys``."""

    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _text(value: object | None) -> str | None:
    """Normalize a non-empty scalar string without coercing containers."""

    if isinstance(value, str):
        normalized = _normalize_text(value).strip()
        return normalized or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _identifier(value: object | None) -> str | None:
    """Extract an identifier from a scalar or common identifier mapping."""

    scalar = _text(value)
    if scalar is not None:
        return scalar
    mapping = _as_mapping(value) if value is not None else None
    if mapping is None:
        return None
    return _text(
        _first(
            mapping,
            "conversation_id",
            "category_id",
            "target_conversation_id",
            "target_category_id",
            "target_id",
            "source_path",
            "id",
            "slug",
        )
    )


def _string_tuple(value: object | None) -> tuple[str, ...]:
    """Extract unique identifiers or labels from a scalar, sequence, or mapping."""

    if value is None:
        return ()
    sequence = _as_sequence(value)
    values: Sequence[object] = sequence if sequence is not None else (value,)
    result: list[str] = []
    for item in values:
        identifier = _identifier(item)
        if identifier is not None and identifier not in result:
            result.append(identifier)
    return tuple(result)


def _float(value: object | None) -> float | None:
    """Return a finite numeric value without treating booleans as numbers."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        if result == result and result not in {float("inf"), float("-inf")}:
            return result
    return None


def _integer(value: object | None) -> int | None:
    """Return an integer value without coercing booleans or fractional floats."""

    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _datetime(value: object | None) -> datetime | None:
    """Return a datetime from a typed value or ISO-8601 string."""

    if isinstance(value, datetime):
        return value
    text = _text(value)
    if text is None:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _mapping_records(value: object | None) -> list[tuple[str | None, Mapping[str, object]]]:
    """Normalize a record sequence or identifier-keyed record mapping."""

    sequence = _as_sequence(value) if value is not None else None
    if sequence is not None:
        return [(None, mapping) for item in sequence if (mapping := _as_mapping(item)) is not None]
    mapping = _as_mapping(value) if value is not None else None
    if mapping is None:
        return []
    return [
        (key, record) for key, item in mapping.items() if (record := _as_mapping(item)) is not None
    ]


def _category_source(root: Mapping[str, object]) -> object | None:
    """Locate category records in direct or taxonomy-wrapped atlas schemas."""

    direct = _first(root, "categories", "category_nodes")
    if direct is not None:
        return direct
    taxonomy = _first(root, "taxonomy", "category_taxonomy")
    taxonomy_mapping = _as_mapping(taxonomy) if taxonomy is not None else None
    if taxonomy_mapping is None:
        return taxonomy
    nested = _first(taxonomy_mapping, "categories", "nodes", "children", "roots")
    return nested if nested is not None else taxonomy_mapping


def _parse_categories(root: Mapping[str, object]) -> tuple[_Category, ...]:
    """Flatten nested or parent-linked category records in deterministic source order."""

    result: list[_Category] = []
    seen: set[str] = set()

    def visit(
        record: Mapping[str, object],
        *,
        fallback_id: str | None,
        parent_hint: str | None,
    ) -> None:
        category_id = _identifier(_first(record, "category_id", "id", "slug", "key")) or fallback_id
        name = _text(_first(record, "name", "label", "title"))
        if category_id is None and name is not None:
            category_id = name.casefold().replace(" ", "-")
        if category_id is None or category_id in seen:
            return
        seen.add(category_id)
        parent_id = _identifier(_first(record, "parent_id", "parent", "parent_category_id"))
        if parent_id is None:
            parent_id = parent_hint
        result.append(
            _Category(
                category_id=category_id,
                name=name or category_id.replace("_", " ").replace("-", " ").title(),
                summary=_text(_first(record, "summary", "description", "profile", "synthesis")),
                parent_id=parent_id,
                conversation_ids=_string_tuple(
                    _first(
                        record,
                        "conversation_ids",
                        "conversation_keys",
                        "members",
                        "conversations",
                    )
                ),
                representative_conversation_ids=_string_tuple(
                    _first(
                        record,
                        "representative_conversation_ids",
                        "representative_conversation_keys",
                        "representatives",
                    )
                ),
                related_category_ids=_string_tuple(
                    _first(record, "related_category_ids", "related_categories", "related")
                ),
                themes=_string_tuple(
                    _first(record, "themes", "key_themes", "keywords", "defining_concepts")
                ),
                characteristic_questions=(),
                major_conclusions=(),
                unresolved_threads=(),
                trajectory=_text(
                    _first(
                        record,
                        "trajectory",
                        "evolution",
                        "temporal_summary",
                        "temporal_evolution",
                    )
                ),
                source_order=len(result),
            )
        )
        children = _first(record, "children", "subcategories")
        for child_fallback, child in _mapping_records(children):
            visit(child, fallback_id=child_fallback, parent_hint=category_id)

    for fallback, record in _mapping_records(_category_source(root)):
        visit(record, fallback_id=fallback, parent_hint=None)

    profiles = _first(root, "category_profiles", "profiles")
    profile_by_id = {
        category_id: profile
        for fallback, profile in _mapping_records(profiles)
        if (category_id := _identifier(_first(profile, "category_id", "id")) or fallback)
        is not None
    }
    merged: list[_Category] = []
    for category in result:
        profile = profile_by_id.get(category.category_id)
        if profile is None:
            merged.append(category)
            continue
        merged.append(
            replace(
                category,
                summary=_text(
                    _first(
                        profile,
                        "summary",
                        "overview",
                        "description",
                        "profile",
                        "synthesis",
                    )
                )
                or category.summary,
                related_category_ids=category.related_category_ids
                or _string_tuple(
                    _first(profile, "related_category_ids", "related_categories", "related")
                ),
                representative_conversation_ids=(
                    category.representative_conversation_ids
                    or _string_tuple(
                        _first(
                            profile,
                            "representative_conversation_ids",
                            "representative_conversation_keys",
                            "representatives",
                        )
                    )
                ),
                themes=category.themes
                or _string_tuple(
                    _first(profile, "themes", "key_themes", "keywords", "defining_concepts")
                ),
                characteristic_questions=_string_tuple(
                    _first(profile, "characteristic_questions", "questions")
                ),
                major_conclusions=_string_tuple(
                    _first(profile, "major_conclusions", "conclusions")
                ),
                unresolved_threads=_string_tuple(
                    _first(profile, "unresolved_threads", "open_threads")
                ),
                trajectory=category.trajectory
                or _text(
                    _first(
                        profile,
                        "trajectory",
                        "evolution",
                        "temporal_summary",
                        "temporal_evolution",
                    )
                ),
            )
        )
    return tuple(merged)


def _parse_relations(value: object | None) -> tuple[_Relation, ...]:
    """Normalize conversation relationships from identifier or edge records."""

    if value is None:
        return ()
    sequence = _as_sequence(value)
    values: Sequence[object] = sequence if sequence is not None else (value,)
    relations: list[_Relation] = []
    seen: set[str] = set()
    for item in values:
        mapping = _as_mapping(item)
        target = _identifier(item)
        if target is None or target in seen:
            continue
        seen.add(target)
        relations.append(
            _Relation(
                target_id=target,
                label=(
                    _text(_first(mapping, "label", "reason", "relationship", "relation_type"))
                    if mapping is not None
                    else None
                ),
                score=(
                    _float(_first(mapping, "score", "weight", "similarity", "confidence"))
                    if mapping is not None
                    else None
                ),
            )
        )
    return tuple(relations)


def _catalog_source(root: Mapping[str, object]) -> object | None:
    """Locate catalog records in common semantic-atlas schemas."""

    return _first(
        root,
        "conversation_catalog",
        "catalog",
        "conversation_profiles",
        "analyses",
        "conversations",
    )


def _parse_catalog(root: Mapping[str, object]) -> Mapping[str, _CatalogEntry]:
    """Normalize multi-label conversation annotations and relationship edges."""

    result: dict[str, _CatalogEntry] = {}
    for fallback, record in _mapping_records(_catalog_source(root)):
        conversation_id = (
            _identifier(
                _first(
                    record,
                    "conversation_id",
                    "conversation_key",
                    "source_conversation_id",
                    "source_path",
                    "id",
                )
            )
            or fallback
        )
        if conversation_id is None:
            continue
        primary_category_id = _identifier(
            _first(record, "primary_category_id", "primary_category", "category_id")
        )
        category_ids = _string_tuple(
            _first(record, "category_ids", "categories", "assignments", "secondary_category_ids")
        )
        if primary_category_id is not None and primary_category_id not in category_ids:
            category_ids = (primary_category_id, *category_ids)
        primary_subject = _text(_first(record, "primary_subject"))
        subjects = list(_string_tuple(_first(record, "secondary_subjects", "subjects")))
        if primary_subject is not None and primary_subject not in subjects:
            subjects.insert(0, primary_subject)
        result[conversation_id] = _CatalogEntry(
            conversation_id=conversation_id,
            synopsis=_text(
                _first(record, "synopsis", "summary", "semantic_synopsis", "description")
            ),
            primary_category_id=primary_category_id,
            category_ids=category_ids,
            related=_parse_relations(
                _first(
                    record,
                    "related_conversations",
                    "related_conversation_ids",
                    "related_conversation_keys",
                    "connections",
                    "semantic_neighbors",
                )
            ),
            projects=_string_tuple(_first(record, "projects", "project_memberships")),
            subjects=tuple(subjects),
            themes=_string_tuple(_first(record, "themes", "motifs", "recurring_themes")),
            entities=_string_tuple(
                _first(record, "entities", "key_entities", "technologies", "concepts")
            ),
            goals=_string_tuple(_first(record, "goals")),
            decisions=_string_tuple(_first(record, "decisions", "major_decisions")),
            unresolved_questions=_string_tuple(
                _first(record, "unresolved_questions", "open_questions")
            ),
            status=_text(_first(record, "status", "temporal_role", "conversation_type", "purpose")),
            confidence=_float(
                _first(record, "confidence", "assignment_confidence", "category_confidence")
            ),
        )

    taxonomy = _as_mapping(_first(root, "taxonomy", "category_taxonomy"))
    assignments = _first(taxonomy, "assignments") if taxonomy is not None else None
    for fallback, assignment in _mapping_records(assignments):
        conversation_id = (
            _identifier(
                _first(assignment, "conversation_key", "conversation_id", "source_path", "id")
            )
            or fallback
        )
        if conversation_id is None:
            continue
        current = result.get(conversation_id, _empty_annotation(conversation_id))
        primary = _identifier(
            _first(assignment, "primary_category_id", "primary_category", "category_id")
        )
        secondary = _string_tuple(
            _first(assignment, "secondary_category_ids", "category_ids", "categories")
        )
        category_ids = tuple(
            dict.fromkeys(
                candidate
                for candidate in (primary, *secondary, *current.category_ids)
                if candidate is not None
            )
        )
        assignment_confidence = _float(_first(assignment, "confidence"))
        result[conversation_id] = replace(
            current,
            primary_category_id=primary or current.primary_category_id,
            category_ids=category_ids,
            confidence=(
                assignment_confidence if assignment_confidence is not None else current.confidence
            ),
        )

    for _, edge in _mapping_records(_first(root, "graph_edges", "semantic_graph", "edges")):
        source = _identifier(_first(edge, "source_key", "source_id", "source"))
        target = _identifier(_first(edge, "target_key", "target_id", "target"))
        if source is None or target is None or source == target:
            continue
        relation = _Relation(
            target_id=target,
            label=_text(_first(edge, "relation_type", "label", "reason")),
            score=_float(_first(edge, "similarity", "score", "weight")),
        )
        reverse = replace(relation, target_id=source)
        for conversation_id, candidate in ((source, relation), (target, reverse)):
            current = result.get(conversation_id, _empty_annotation(conversation_id))
            if candidate.target_id not in {item.target_id for item in current.related}:
                result[conversation_id] = replace(
                    current,
                    related=(*current.related, candidate),
                )
    return result


def _parse_project_timelines(root: Mapping[str, object]) -> tuple[_ProjectTimeline, ...]:
    """Normalize archive-spanning project narratives and their event references."""

    timelines: list[_ProjectTimeline] = []
    for fallback, record in _mapping_records(
        _first(root, "project_timelines", "projects", "timelines")
    ):
        name = _text(_first(record, "project_name", "name", "title")) or fallback
        overview = _text(_first(record, "overview", "summary", "description"))
        if name is None or overview is None:
            continue
        events: list[_TimelineEvent] = []
        for _, event in _mapping_records(_first(record, "events", "milestones", "stages")):
            label = _text(_first(event, "label", "title", "name"))
            summary = _text(_first(event, "summary", "description", "body"))
            if label is None or summary is None:
                continue
            events.append(
                _TimelineEvent(
                    label=label,
                    summary=summary,
                    conversation_ids=_string_tuple(
                        _first(
                            event,
                            "conversation_keys",
                            "conversation_ids",
                            "conversations",
                        )
                    ),
                    occurred_at=_datetime(_first(event, "occurred_at", "date", "timestamp")),
                )
            )
        timelines.append(
            _ProjectTimeline(
                name=name,
                overview=overview,
                events=tuple(events),
                current_state=_text(_first(record, "current_state", "status", "latest_state")),
                unresolved_work=_string_tuple(
                    _first(record, "unresolved_work", "next_steps", "open_work")
                ),
            )
        )
    return tuple(timelines)


def _parse_synthesis_value(value: object | None) -> tuple[_SynthesisSection, ...]:
    """Normalize a string, structured section list, or named-section mapping."""

    body = _text(value)
    if body is not None:
        return (_SynthesisSection(title="Across the archive", body=body),)
    sequence = _as_sequence(value) if value is not None else None
    if sequence is not None:
        sections: list[_SynthesisSection] = []
        for item in sequence:
            mapping = _as_mapping(item)
            if mapping is None:
                continue
            section_body = _text(_first(mapping, "body", "content", "summary", "text"))
            if section_body is None:
                continue
            sections.append(
                _SynthesisSection(
                    title=_text(_first(mapping, "title", "heading", "name")) or "Synthesis",
                    body=section_body,
                )
            )
        return tuple(sections)
    mapping = _as_mapping(value) if value is not None else None
    if mapping is None:
        return ()
    executive_summary = _text(_first(mapping, "executive_summary"))
    structured_fields = (
        ("Dominant domains", "dominant_domains"),
        ("Connections across domains", "cross_domain_connections"),
        ("Recurring patterns", "recurring_patterns"),
        ("Development over time", "temporal_evolution"),
        ("Tensions and reversals", "tensions_and_reversals"),
        ("Dormant threads", "dormant_threads"),
        ("Opportunities", "opportunities"),
    )
    if executive_summary is not None or any(mapping.get(key) for _, key in structured_fields):
        structured: list[_SynthesisSection] = []
        if executive_summary is not None:
            structured.append(_SynthesisSection(title="Executive summary", body=executive_summary))
        for title, key in structured_fields:
            items = _string_tuple(mapping.get(key))
            if items:
                structured.append(
                    _SynthesisSection(
                        title=title,
                        body="\n".join(f"- {item}" for item in items),
                    )
                )
        return tuple(structured)
    nested_sections = _first(mapping, "sections", "chapters")
    if nested_sections is not None:
        lead = _text(_first(mapping, "overview", "summary", "introduction"))
        parsed = list(_parse_synthesis_value(nested_sections))
        if lead is not None:
            parsed.insert(0, _SynthesisSection(title="Across the archive", body=lead))
        return tuple(parsed)
    sections = []
    for key, item in mapping.items():
        section_body = _text(item)
        if section_body is not None:
            sections.append(
                _SynthesisSection(
                    title=key.replace("_", " ").strip().title(),
                    body=section_body,
                )
            )
    return tuple(sections)


def _normalize_atlas(atlas: AtlasInput) -> _NormalizedAtlas:
    """Project a typed model or generic mapping into the renderer's stable schema."""

    root = _as_mapping(atlas)
    if root is None:
        raise ArchiveCompilationError("Semantic atlas must be a mapping or Pydantic model.")
    nested_atlas = _as_mapping(root.get("atlas"))
    if nested_atlas is not None:
        root = nested_atlas
    categories = _parse_categories(root)
    catalog = _parse_catalog(root)
    synthesis = _parse_synthesis_value(
        _first(root, "cross_archive_synthesis", "synthesis", "global_synthesis", "overview")
    )
    project_timelines = _parse_project_timelines(root)
    keys_by_source_index: dict[int, str] = {}
    for _, representation in _mapping_records(root.get("representations")):
        source_index = _integer(representation.get("source_index"))
        conversation_key = _text(representation.get("conversation_key"))
        if source_index is not None and source_index >= 0 and conversation_key is not None:
            keys_by_source_index[source_index] = conversation_key
    return _NormalizedAtlas(
        categories=categories,
        catalog=catalog,
        synthesis=synthesis,
        project_timelines=project_timelines,
        conversation_keys_by_source_index=keys_by_source_index,
    )


def _empty_annotation(conversation_id: str) -> _CatalogEntry:
    """Return a neutral annotation for an Archive conversation absent from the catalog."""

    return _CatalogEntry(
        conversation_id=conversation_id,
        synopsis=None,
        primary_category_id=None,
        category_ids=(),
        related=(),
        projects=(),
        subjects=(),
        themes=(),
        entities=(),
        goals=(),
        decisions=(),
        unresolved_questions=(),
        status=None,
        confidence=None,
    )


def _conversation_id(conversation: Conversation) -> str:
    """Return the public IR identifier, falling back to the archive-relative source path."""

    return conversation.conversation_id or str(conversation.source_path)


def _semantic_conversation_key(conversation: Conversation, source_index: int) -> str:
    """Reproduce the semantic pipeline's opaque conversation-key derivation.

    The source index is the conversation's position in Archive IR before any chronological sort.
    Matching the analysis pipeline's versioned preimage lets the renderer join a ``SemanticAtlas``
    back to Archive IR without persisting private identifiers in a second lookup artifact.
    """

    preimage = "\0".join(
        (
            "semantic-conversation-v1",
            str(source_index),
            str(conversation.source_path),
            conversation.conversation_id or "",
        )
    )
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:24]


def _conversation_sort_key(conversation: Conversation) -> tuple[float, str, str]:
    """Return a deterministic chronological key for conversation placement."""

    date = conversation.created_at or conversation.updated_at
    timestamp = date.timestamp() if date is not None else float("inf")
    return timestamp, str(conversation.source_path), conversation.conversation_id or ""


def _resolve_placements(
    archive: Archive,
    atlas: _NormalizedAtlas,
    *,
    max_conversations: int | None,
) -> tuple[tuple[_Category, ...], tuple[_PlacedConversation, ...], int]:
    """Resolve primary placements while retaining secondary multi-label associations."""

    categories = list(atlas.categories)
    category_ids = {category.category_id for category in categories}
    membership: defaultdict[str, list[str]] = defaultdict(list)
    for category in categories:
        for conversation_id in category.conversation_ids:
            membership[conversation_id].append(category.category_id)

    selected = sorted(
        enumerate(archive.conversations),
        key=lambda item: _conversation_sort_key(item[1]),
    )
    if max_conversations is not None:
        selected = selected[:max_conversations]
    placements: list[_PlacedConversation] = []
    unclassified_count = 0
    known_ids = {*atlas.catalog, *membership}
    for source_index, conversation in selected:
        source_id = _conversation_id(conversation)
        semantic_key = atlas.conversation_keys_by_source_index.get(
            source_index,
            _semantic_conversation_key(conversation, source_index),
        )
        conversation_id = next(
            (candidate for candidate in (semantic_key, source_id) if candidate in known_ids),
            source_id,
        )
        annotation = atlas.catalog.get(conversation_id, _empty_annotation(conversation_id))
        candidates = (
            annotation.primary_category_id,
            *annotation.category_ids,
            *membership.get(conversation_id, ()),
        )
        primary = next(
            (
                candidate
                for candidate in candidates
                if candidate is not None and candidate in category_ids
            ),
            _UNCLASSIFIED_ID,
        )
        if primary == _UNCLASSIFIED_ID:
            unclassified_count += 1
        placements.append(
            _PlacedConversation(
                conversation_id=conversation_id,
                conversation=conversation,
                annotation=annotation,
                primary_category_id=primary,
            )
        )
    if unclassified_count:
        categories.append(
            _Category(
                category_id=_UNCLASSIFIED_ID,
                name="Unclassified conversations",
                summary=(
                    "Conversations for which the current semantic analysis did not establish a "
                    "confident primary placement. They remain visible for review rather than "
                    "being silently discarded."
                ),
                parent_id=None,
                conversation_ids=(),
                representative_conversation_ids=(),
                related_category_ids=(),
                themes=(),
                characteristic_questions=(),
                major_conclusions=(),
                unresolved_threads=(),
                trajectory=None,
                source_order=len(categories),
            )
        )
    return tuple(categories), tuple(placements), unclassified_count


def _ordered_categories(categories: Sequence[_Category]) -> tuple[tuple[_Category, int], ...]:
    """Return a cycle-safe, parent-before-child taxonomy traversal with depth values."""

    by_id = {category.category_id: category for category in categories}
    children: defaultdict[str | None, list[_Category]] = defaultdict(list)
    for category in categories:
        parent = category.parent_id
        if parent == category.category_id or parent not in by_id:
            parent = None
        children[parent].append(category)
    for items in children.values():
        items.sort(key=lambda item: (item.source_order, item.name.casefold(), item.category_id))

    ordered: list[tuple[_Category, int]] = []
    visited: set[str] = set()

    def visit(category: _Category, depth: int, active: set[str]) -> None:
        if category.category_id in visited or category.category_id in active:
            return
        visited.add(category.category_id)
        ordered.append((category, depth))
        next_active = {*active, category.category_id}
        for child in children.get(category.category_id, ()):
            visit(child, depth + 1, next_active)

    for root in children.get(None, ()):
        visit(root, 0, set())
    for category in sorted(
        categories, key=lambda item: (item.source_order, item.name.casefold(), item.category_id)
    ):
        visit(category, 0, set())
    return tuple(ordered)


def _anchor(prefix: str, identifier: str) -> str:
    """Create a short stable HTML anchor without disclosing raw source identifiers."""

    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _display_date(conversation: Conversation) -> str:
    """Return a human-readable UTC date for one Archive conversation."""

    date = conversation.created_at or conversation.updated_at
    if date is None:
        return "Undated"
    utc_date = date.astimezone(UTC)
    return f"{utc_date:%B} {utc_date.day}, {utc_date.year}"


def _date_range(conversations: Sequence[_PlacedConversation]) -> str | None:
    """Return a compact UTC date span for a category's placed conversations."""

    dates = [
        date.astimezone(UTC)
        for item in conversations
        if (date := item.conversation.created_at or item.conversation.updated_at) is not None
    ]
    if not dates:
        return None
    earliest = min(dates)
    latest = max(dates)
    if earliest.year == latest.year and earliest.month == latest.month:
        return f"{earliest:%B %Y}"
    if earliest.year == latest.year:
        return f"{earliest:%B}–{latest:%B %Y}"
    return f"{earliest:%B %Y}–{latest:%B %Y}"


def _safe_markdown(value: str) -> str:
    """Render Markdown with raw HTML and embedded assets disabled."""

    fragment = _MARKDOWN.render(_normalize_text(value))
    soup = BeautifulSoup(fragment, "html.parser")
    for image in soup.find_all("img"):
        alt = image.get("alt")
        image.replace_with(f"[Image omitted: {alt}]" if alt else "[Image omitted]")
    for link in soup.find_all("a"):
        href = link.get("href")
        if not isinstance(href, str):
            link.unwrap()
            continue
        parsed = urlsplit(href)
        if href.startswith("#") or parsed.scheme.casefold() in {"http", "https", "mailto"}:
            link["rel"] = "noreferrer"
        else:
            link.unwrap()
    return str(soup)


def _render_block(block: ContentBlock) -> str:
    """Render a public transcript content block without loading referenced assets."""

    if block.type in {ContentBlockType.TEXT, ContentBlockType.MARKDOWN}:
        return _safe_markdown(block.text or "")
    if block.type is ContentBlockType.CODE:
        language = html.escape(_normalize_text(block.language or "text"))
        text = html.escape(_normalize_text(block.text or ""))
        return f'<div class="code-label">{language}</div><pre><code>{text}</code></pre>'
    if block.type in {
        ContentBlockType.FILE_REFERENCE,
        ContentBlockType.IMAGE_REFERENCE,
        ContentBlockType.AUDIO_REFERENCE,
    }:
        label = block.type.value.replace("_", " ").title()
        return f'<p class="asset-note">{html.escape(label)} omitted from this edition.</p>'
    return ""


def _visible_messages(conversation: Conversation) -> tuple[Message, ...]:
    """Return user and assistant messages from the selected visible path."""

    visible: list[Message] = []
    for message in conversation.current_path_messages:
        if message.role not in {Role.USER, Role.ASSISTANT}:
            continue
        if not any(
            block.type not in {ContentBlockType.THINKING_TRACE, ContentBlockType.REASONING_SUMMARY}
            for block in message.content
        ):
            continue
        visible.append(message)
    return tuple(visible)


def _write_badges(handle: IO[str], *, label: str, values: Sequence[str]) -> None:
    """Write one labeled, compact semantic metadata row when values are present."""

    if not values:
        return
    handle.write(
        f'<div class="metadata-row"><span class="metadata-label">{html.escape(label)}</span>'
    )
    for value in values:
        handle.write(f'<span class="badge">{html.escape(_normalize_text(value))}</span>')
    handle.write("</div>")


def _write_profile_list(handle: IO[str], *, title: str, values: Sequence[str]) -> None:
    """Write one compact interpretive list on a category opener."""

    if not values:
        return
    handle.write(f'<section class="profile-list"><h2>{html.escape(title)}</h2><ul>')
    for value in values:
        handle.write(f"<li>{html.escape(_normalize_text(value))}</li>")
    handle.write("</ul></section>")


def _write_transcript(
    handle: IO[str],
    conversation: Conversation,
    *,
    include_timestamps: bool,
) -> None:
    """Write a typographically restrained user/assistant transcript."""

    messages = _visible_messages(conversation)
    if not messages:
        handle.write('<p class="empty-note">No visible user/assistant messages.</p>')
        return
    handle.write('<div class="transcript">')
    for message in messages:
        role = message.role.value
        handle.write(f'<section class="message {role}"><header><span>{role.title()}</span>')
        if include_timestamps and message.created_at is not None:
            timestamp = message.created_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
            handle.write(f"<time>{html.escape(timestamp)}</time>")
        handle.write('</header><div class="message-body">')
        for block in message.content:
            if block.type not in {
                ContentBlockType.THINKING_TRACE,
                ContentBlockType.REASONING_SUMMARY,
            }:
                handle.write(_render_block(block))
        handle.write("</div></section>")
    handle.write("</div>")


def _write_category_toc(
    handle: IO[str], ordered: Sequence[tuple[_Category, int]], counts: Mapping[str, int]
) -> None:
    """Write the print-aware hierarchical table of contents."""

    handle.write(
        '<nav class="toc" aria-labelledby="contents-heading"><h1 id="contents-heading">Contents</h1><ol>'
    )
    for number, (category, depth) in enumerate(ordered, start=1):
        count = counts.get(category.category_id, 0)
        handle.write(
            f'<li class="toc-depth-{min(depth, 3)}"><a href="#{_anchor("category", category.category_id)}">'
            f'<span class="toc-number">{number:02d}</span>'
            f'<span class="toc-title">{html.escape(_normalize_text(category.name))}</span>'
            f'<span class="toc-count">{count:,} {"chat" if count == 1 else "chats"}</span></a></li>'
        )
    handle.write("</ol></nav>")


def _featured_conversations(
    category: _Category,
    conversations: Sequence[_PlacedConversation],
    limit: int | None,
) -> tuple[_PlacedConversation, ...]:
    """Select representative, then high-confidence, entries for expanded print profiles."""

    target_count = len(conversations) if limit is None else min(limit, len(conversations))
    if target_count == 0:
        return ()
    by_id = {item.conversation_id: item for item in conversations}
    featured: list[_PlacedConversation] = []
    selected_ids: set[str] = set()
    for conversation_id in category.representative_conversation_ids:
        item = by_id.get(conversation_id)
        if item is not None and conversation_id not in selected_ids:
            featured.append(item)
            selected_ids.add(conversation_id)
            if len(featured) == target_count:
                return tuple(featured)
    remainder = [item for item in conversations if item.conversation_id not in selected_ids]
    remainder.sort(
        key=lambda item: (
            -(item.annotation.confidence if item.annotation.confidence is not None else -1.0),
            _conversation_sort_key(item.conversation),
        )
    )
    featured.extend(remainder[: target_count - len(featured)])
    return tuple(featured)


def _aggregate_category_placements(
    categories: Sequence[_Category],
    direct: Mapping[str, Sequence[_PlacedConversation]],
) -> dict[str, tuple[_PlacedConversation, ...]]:
    """Aggregate descendant placements for structural parent statistics without duplication."""

    category_ids = {category.category_id for category in categories}
    children: defaultdict[str, list[str]] = defaultdict(list)
    for category in categories:
        if category.parent_id in category_ids and category.parent_id != category.category_id:
            children[category.parent_id].append(category.category_id)
    cache: dict[str, tuple[_PlacedConversation, ...]] = {}

    def collect(category_id: str, active: frozenset[str]) -> tuple[_PlacedConversation, ...]:
        cached = cache.get(category_id)
        if cached is not None:
            return cached
        if category_id in active:
            return ()
        gathered = list(direct.get(category_id, ()))
        next_active = active | {category_id}
        for child_id in children.get(category_id, ()):
            gathered.extend(collect(child_id, next_active))
        deduplicated = {
            item.conversation_id: item
            for item in sorted(
                gathered, key=lambda value: _conversation_sort_key(value.conversation)
            )
        }
        result = tuple(deduplicated.values())
        cache[category_id] = result
        return result

    return {
        category.category_id: collect(category.category_id, frozenset()) for category in categories
    }


def _apply_global_feature_budget(
    categories: Sequence[_Category],
    candidates: Mapping[str, Sequence[_PlacedConversation]],
    limit: int | None,
) -> dict[str, tuple[_PlacedConversation, ...]]:
    """Allocate a global expansion budget round-robin so every category receives fair coverage."""

    if limit is None:
        return {
            category.category_id: tuple(
                sorted(
                    candidates.get(category.category_id, ()),
                    key=lambda item: _conversation_sort_key(item.conversation),
                )
            )
            for category in categories
        }
    selected: dict[str, list[_PlacedConversation]] = {
        category.category_id: [] for category in categories
    }
    offsets = {category.category_id: 0 for category in categories}
    remaining = limit
    while remaining:
        progressed = False
        for category in categories:
            category_id = category.category_id
            values = candidates.get(category_id, ())
            offset = offsets[category_id]
            if offset >= len(values):
                continue
            selected[category_id].append(values[offset])
            offsets[category_id] = offset + 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return {
        category_id: tuple(
            sorted(values, key=lambda item: _conversation_sort_key(item.conversation))
        )
        for category_id, values in selected.items()
    }


def _write_category_opener(
    handle: IO[str],
    *,
    category: _Category,
    depth: int,
    number: int,
    conversations: Sequence[_PlacedConversation],
    directory_conversations: Sequence[_PlacedConversation],
    featured_ids: set[str],
    categories_by_id: Mapping[str, _Category],
) -> None:
    """Write one category chapter opener, profile, and compact conversation directory."""

    category_anchor = _anchor("category", category.category_id)
    level_label = "Part" if depth == 0 else "Chapter" if depth == 1 else "Section"
    handle.write(
        f'<section class="category-opener depth-{min(depth, 3)}" id="{category_anchor}">'
        f'<p class="eyebrow">{level_label} {number:02d}</p>'
        f"<h1>{html.escape(_normalize_text(category.name))}</h1>"
    )
    count = len(conversations)
    stats = [f"{count:,} {'conversation' if count == 1 else 'conversations'}"]
    date_range = _date_range(conversations)
    if date_range is not None:
        stats.append(date_range)
    handle.write(f'<p class="category-stats">{" · ".join(stats)}</p>')
    if category.summary:
        handle.write(f'<div class="category-summary">{_safe_markdown(category.summary)}</div>')
    if category.trajectory:
        handle.write(
            '<aside class="trajectory"><h2>Development over time</h2>'
            f"{_safe_markdown(category.trajectory)}</aside>"
        )
    _write_badges(handle, label="Defining themes", values=category.themes)
    _write_profile_list(
        handle,
        title="Characteristic questions",
        values=category.characteristic_questions,
    )
    _write_profile_list(
        handle,
        title="Major conclusions",
        values=category.major_conclusions,
    )
    _write_profile_list(
        handle,
        title="Unresolved threads",
        values=category.unresolved_threads,
    )
    related = [
        categories_by_id[category_id]
        for category_id in category.related_category_ids
        if category_id in categories_by_id and category_id != category.category_id
    ]
    if related:
        handle.write('<p class="category-links"><span>Related categories</span>')
        for related_category in related:
            handle.write(
                f'<a href="#{_anchor("category", related_category.category_id)}">'
                f"{html.escape(_normalize_text(related_category.name))}</a>"
            )
        handle.write("</p>")
    handle.write("</section>")
    if directory_conversations:
        handle.write(
            '<section class="chapter-directory"><h2>Conversations in this category</h2><ol>'
        )
        for placed in directory_conversations:
            title = placed.conversation.title or "Untitled conversation"
            handle.write(f'<li id="{_anchor("conversation", placed.conversation_id)}">')
            if placed.conversation_id in featured_ids:
                handle.write(f'<a href="#{_anchor("entry", placed.conversation_id)}">')
            else:
                handle.write('<div class="directory-record">')
            handle.write(
                f"<span>{html.escape(_normalize_text(title))}</span>"
                f"<time>{html.escape(_display_date(placed.conversation))}</time>"
            )
            handle.write("</a></li>" if placed.conversation_id in featured_ids else "</div></li>")
        handle.write("</ol></section>")


def _write_conversation_entry(
    handle: IO[str],
    *,
    item: _PlacedConversation,
    categories_by_id: Mapping[str, _Category],
    placements_by_id: Mapping[str, _PlacedConversation],
    options: SemanticBookOptions,
) -> None:
    """Write one synopsis-led catalog entry with multi-label and graph cross-references."""

    conversation = item.conversation
    annotation = item.annotation
    title = conversation.title or "Untitled conversation"
    handle.write(
        f'<article class="conversation-entry" id="{_anchor("entry", item.conversation_id)}">'
        '<header class="conversation-header">'
        f'<p class="conversation-date">{html.escape(_display_date(conversation))}</p>'
        f"<h2>{html.escape(_normalize_text(title))}</h2>"
    )
    if annotation.status:
        handle.write(f'<p class="conversation-status">{html.escape(annotation.status)}</p>')
    handle.write("</header>")
    if annotation.synopsis:
        handle.write(f'<div class="synopsis">{_safe_markdown(annotation.synopsis)}</div>')
    else:
        handle.write(
            '<p class="synopsis missing">No semantic synopsis was available in this analysis run.</p>'
        )
    secondary_ids = [
        category_id
        for category_id in annotation.category_ids
        if category_id != item.primary_category_id and category_id in categories_by_id
    ]
    if secondary_ids:
        handle.write(
            '<div class="metadata-row"><span class="metadata-label">Also appears in</span>'
        )
        for category_id in secondary_ids:
            category = categories_by_id[category_id]
            handle.write(
                f'<a class="badge linked" href="#{_anchor("category", category_id)}">'
                f"{html.escape(_normalize_text(category.name))}</a>"
            )
        handle.write("</div>")
    _write_badges(handle, label="Projects", values=annotation.projects)
    _write_badges(handle, label="Subjects", values=annotation.subjects)
    _write_badges(handle, label="Themes", values=annotation.themes)
    _write_badges(handle, label="Entities", values=annotation.entities)
    _write_badges(handle, label="Goals", values=annotation.goals)
    _write_badges(handle, label="Decisions", values=annotation.decisions)
    _write_badges(
        handle,
        label="Open questions",
        values=annotation.unresolved_questions,
    )
    related = [
        relation
        for relation in annotation.related
        if relation.target_id != item.conversation_id and relation.target_id in placements_by_id
    ][: options.max_related_conversations]
    if related:
        handle.write('<aside class="cross-references"><h3>Connected conversations</h3><ul>')
        for relation in related:
            target = placements_by_id[relation.target_id]
            target_title = target.conversation.title or "Untitled conversation"
            handle.write(
                f'<li><a href="#{_anchor("conversation", relation.target_id)}">'
                f"{html.escape(_normalize_text(target_title))}</a>"
            )
            descriptors: list[str] = []
            if relation.label:
                descriptors.append(relation.label)
            if relation.score is not None:
                descriptors.append(f"affinity {relation.score:.2f}")
            if descriptors:
                handle.write(f'<span>{html.escape(" · ".join(descriptors))}</span>')
            handle.write("</li>")
        handle.write("</ul></aside>")
    if options.include_transcripts:
        handle.write(
            '<details class="transcript-section" open><summary>Conversation transcript</summary>'
        )
        _write_transcript(
            handle,
            conversation,
            include_timestamps=options.include_message_timestamps,
        )
        handle.write("</details>")
    handle.write('<p class="back-link"><a href="#contents-heading">Return to contents</a></p>')
    handle.write("</article>")


def _write_project_timelines(
    handle: IO[str],
    timelines: Sequence[_ProjectTimeline],
    placements_by_id: Mapping[str, _PlacedConversation],
    *,
    max_timelines: int | None,
    max_events_per_project: int | None,
) -> None:
    """Write archive-spanning project profiles, milestones, and source cross-references."""

    if not timelines:
        return
    handle.write(
        '<section class="project-atlas"><p class="eyebrow">Longitudinal view</p>'
        '<h1>Project timelines</h1><p class="project-introduction">Recurring projects are '
        "presented as evolving lines of work rather than isolated subjects. Milestones link back "
        "to the conversation entries that document them.</p></section>"
    )
    selected_timelines = timelines[:max_timelines] if max_timelines is not None else timelines
    if len(selected_timelines) < len(timelines):
        handle.write(
            '<p class="editorial-note">This print edition includes '
            f"{len(selected_timelines):,} of {len(timelines):,} project timelines. "
            "The complete structured set remains in project_timelines.json.</p>"
        )
    for timeline in selected_timelines:
        handle.write(
            '<section class="project-profile">'
            '<p class="eyebrow">Project</p>'
            f"<h1>{html.escape(_normalize_text(timeline.name))}</h1>"
            f'<div class="project-overview">{_safe_markdown(timeline.overview)}</div>'
        )
        if timeline.current_state:
            handle.write(
                '<aside class="current-state"><h2>Current state</h2>'
                f"{_safe_markdown(timeline.current_state)}</aside>"
            )
        selected_events = (
            timeline.events[:max_events_per_project]
            if max_events_per_project is not None
            else timeline.events
        )
        if selected_events:
            handle.write('<ol class="timeline">')
            for event in selected_events:
                handle.write('<li><div class="timeline-marker"></div><div>')
                if event.occurred_at is not None:
                    handle.write(f'<time>{html.escape(event.occurred_at.strftime("%B %Y"))}</time>')
                handle.write(
                    f"<h2>{html.escape(_normalize_text(event.label))}</h2>"
                    f"{_safe_markdown(event.summary)}"
                )
                references = [
                    placements_by_id[conversation_id]
                    for conversation_id in event.conversation_ids
                    if conversation_id in placements_by_id
                ]
                if references:
                    handle.write('<p class="event-sources"><span>Source conversations</span>')
                    for item in references:
                        title = item.conversation.title or "Untitled conversation"
                        handle.write(
                            f'<a href="#{_anchor("conversation", item.conversation_id)}">'
                            f"{html.escape(_normalize_text(title))}</a>"
                        )
                    handle.write("</p>")
                handle.write("</div></li>")
            handle.write("</ol>")
        if len(selected_events) < len(timeline.events):
            handle.write(
                '<p class="editorial-note">'
                f"{len(timeline.events) - len(selected_events):,} additional milestones are "
                "retained in project_timelines.json.</p>"
            )
        _write_profile_list(
            handle,
            title="Unresolved work",
            values=timeline.unresolved_work,
        )
        handle.write("</section>")


def _book_css(options: SemanticBookOptions) -> str:
    """Return self-contained paged-media CSS for the selected paper geometry."""

    page_size = "7in 10in" if options.paper_size is BookPaperSize.TRADE else "A4"
    return _BOOK_CSS.replace("__PAGE_SIZE__", page_size)


def _write_book_html(
    destination: Path,
    *,
    atlas: _NormalizedAtlas,
    categories: Sequence[_Category],
    placements: Sequence[_PlacedConversation],
    unclassified_count: int,
    options: SemanticBookOptions,
) -> int:
    """Stream one self-contained semantic book and return expanded profile count."""

    ordered = _ordered_categories(categories)
    categories_by_id = {category.category_id: category for category in categories}
    placements_by_id = {item.conversation_id: item for item in placements}
    by_category: defaultdict[str, list[_PlacedConversation]] = defaultdict(list)
    for item in placements:
        by_category[item.primary_category_id].append(item)
    aggregate_by_category = _aggregate_category_placements(categories, by_category)
    counts = {
        category.category_id: len(aggregate_by_category[category.category_id])
        for category in categories
    }
    feature_candidates = {
        category.category_id: _featured_conversations(
            category,
            by_category.get(category.category_id, ()),
            options.max_synopsis_entries_per_category,
        )
        for category in categories
    }
    featured_by_category = _apply_global_feature_budget(
        tuple(category for category, _ in ordered),
        feature_candidates,
        options.max_expanded_conversations,
    )
    expanded_count = sum(len(items) for items in featured_by_category.values())
    title = html.escape(_normalize_text(options.title))
    subtitle = html.escape(_normalize_text(options.subtitle)) if options.subtitle else None

    with _atomic_text_writer(destination) as handle:
        handle.write(
            '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="Content-Security-Policy" '
            "content=\"default-src 'none'; style-src 'unsafe-inline'; img-src data:\">"
            f"<title>{title}</title><style>{_book_css(options)}</style></head><body>"
        )
        handle.write(
            '<section class="half-title"><p>Semantic archive</p>'
            f"<h1>{title}</h1></section>"
            '<section class="title-page"><div class="title-rule"></div>'
            '<p class="kicker">A private analytical edition</p>'
            f"<h1>{title}</h1>"
        )
        if subtitle:
            handle.write(f'<p class="subtitle">{subtitle}</p>')
        if options.author:
            handle.write(f'<p class="author">{html.escape(_normalize_text(options.author))}</p>')
        if options.edition:
            handle.write(f'<p class="edition">{html.escape(_normalize_text(options.edition))}</p>')
        handle.write("</section>")
        handle.write('<section class="colophon"><h1>About this edition</h1>')
        handle.write(
            f"<p>This semantic atlas organizes {len(placements):,} conversations into "
            f"{len(categories):,} categories. Each conversation has one primary placement while "
            "secondary subjects and graph relationships remain available as cross-references.</p>"
        )
        if expanded_count < len(placements):
            handle.write(
                f"<p>All {len(placements):,} conversations appear in the category directories; "
                f"{expanded_count:,} representative or high-confidence profiles are expanded in "
                "this bounded print edition.</p>"
            )
        if unclassified_count:
            handle.write(
                f"<p>{unclassified_count:,} conversations remain explicitly unclassified and "
                "are collected for later human or model-assisted review.</p>"
            )
        if not options.include_transcripts:
            handle.write(
                "<p>This atlas edition prints semantic synopses rather than duplicating full "
                "transcripts. The source Archive IR and chronological compilation remain the "
                "authoritative transcript record.</p>"
            )
        handle.write(
            "<p>Generated locally from normalized Archive IR. No remote fonts, scripts, images, "
            "or stylesheets are embedded in this document.</p></section>"
        )
        if atlas.synthesis:
            handle.write(
                '<section class="synthesis"><p class="eyebrow">Orientation</p><h1>The archive in perspective</h1>'
            )
            for section in atlas.synthesis:
                handle.write(
                    f"<section><h2>{html.escape(_normalize_text(section.title))}</h2>"
                    f"{_safe_markdown(section.body)}</section>"
                )
            handle.write("</section>")
        _write_project_timelines(
            handle,
            atlas.project_timelines,
            placements_by_id,
            max_timelines=options.max_project_timelines,
            max_events_per_project=options.max_timeline_events_per_project,
        )
        _write_category_toc(handle, ordered, counts)
        handle.write('<main class="book-body">')
        for number, (category, depth) in enumerate(ordered, start=1):
            category_conversations = aggregate_by_category[category.category_id]
            directory_conversations = by_category.get(category.category_id, ())
            featured = featured_by_category[category.category_id]
            _write_category_opener(
                handle,
                category=category,
                depth=depth,
                number=number,
                conversations=category_conversations,
                directory_conversations=directory_conversations,
                featured_ids={item.conversation_id for item in featured},
                categories_by_id=categories_by_id,
            )
            for item in featured:
                _write_conversation_entry(
                    handle,
                    item=item,
                    categories_by_id=categories_by_id,
                    placements_by_id=placements_by_id,
                    options=options,
                )
        handle.write("</main></body></html>\n")
    return expanded_count


def _offline_url_fetcher(url: str, *args: object, **kwargs: object) -> object:
    """Reject all subsidiary URL loads during PDF generation."""

    del args, kwargs
    scheme = urlsplit(url).scheme.casefold() or "local"
    raise ArchiveCompilationError(f"Semantic-book PDF blocked a {scheme!r} asset request.")


def _render_pdf(html_path: Path, destination: Path) -> Path:
    """Render local book HTML to PDF using WeasyPrint with subsidiary fetching disabled."""

    try:
        weasyprint = importlib.import_module("weasyprint")
    except ImportError as exc:
        raise ArchiveCompilationError(
            "Semantic-book PDF rendering requires the optional 'pdf' dependency set."
        ) from exc

    temporary_path = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        html_factory = cast(_HTMLFactory, weasyprint.HTML)
        html_factory(
            filename=str(html_path),
            base_url=str(html_path.parent),
            url_fetcher=_offline_url_fetcher,
        ).write_pdf(str(temporary_path))
        os.replace(temporary_path, destination)
    except ArchiveCompilationError:
        raise
    except Exception as exc:
        raise ArchiveCompilationError(
            f"Semantic-book PDF rendering failed safely ({type(exc).__name__}): {destination}"
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def compile_semantic_book(
    archive: Archive,
    atlas: AtlasInput,
    output_directory: str | Path,
    *,
    options: SemanticBookOptions | None = None,
) -> SemanticBookResult:
    """Render a semantic atlas and Archive IR as polished, category-organized book artifacts.

    Parameters
    ----------
    archive
        Normalized Archive IR. Only its declared current conversation paths can enter optional
        transcript sections.
    atlas
        A Pydantic model or mapping. The adapter accepts direct or taxonomy-wrapped categories,
        conversation catalog records, category profiles, and structured or Markdown synthesis.
        Common field aliases are normalized so analysis implementations remain loosely coupled to
        the renderer.
    output_directory
        Private local directory in which HTML, optional PDF, and a non-content manifest are written.
    options
        Publication metadata, geometry, transcript policy, and PDF behavior.

    Returns
    -------
    SemanticBookResult
        Generated paths, SHA-256 digests, and coverage counts.

    Notes
    -----
    The HTML is self-contained and has a restrictive content-security policy. Markdown raw HTML and
    images are disabled, source reasoning records are omitted, and PDF rendering rejects every
    subsidiary asset request. Keep the output directory outside version control because synopses and
    optional transcripts contain private archive material.
    """

    active_options = options or SemanticBookOptions()
    normalized = _normalize_atlas(atlas)
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    categories, placements, unclassified_count = _resolve_placements(
        archive,
        normalized,
        max_conversations=active_options.max_conversations,
    )
    html_path = destination / "semantic-atlas.html"
    expanded_conversation_count = _write_book_html(
        html_path,
        atlas=normalized,
        categories=categories,
        placements=placements,
        unclassified_count=unclassified_count,
        options=active_options,
    )
    pdf_path = destination / "semantic-atlas.pdf" if active_options.render_pdf else None
    if pdf_path is not None:
        _render_pdf(html_path, pdf_path)
    manifest_path = destination / "semantic_book_manifest.json"
    html_digest = _sha256(html_path)
    pdf_digest = _sha256(pdf_path) if pdf_path is not None else None
    manifest = {
        "renderer_version": "1.0",
        "archive_version": archive.archive_version,
        "category_count": len(categories),
        "conversation_count": len(placements),
        "expanded_conversation_count": expanded_conversation_count,
        "unclassified_conversation_count": unclassified_count,
        "include_transcripts": active_options.include_transcripts,
        "paper_size": active_options.paper_size.value,
        "max_synopsis_entries_per_category": (active_options.max_synopsis_entries_per_category),
        "max_expanded_conversations": active_options.max_expanded_conversations,
        "max_project_timelines": active_options.max_project_timelines,
        "max_timeline_events_per_project": (active_options.max_timeline_events_per_project),
        "html": html_path.name,
        "html_sha256": html_digest,
        "pdf": pdf_path.name if pdf_path is not None else None,
        "pdf_sha256": pdf_digest,
    }
    with _atomic_text_writer(manifest_path) as handle:
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return SemanticBookResult(
        output_directory=destination,
        html_path=html_path,
        html_sha256=html_digest,
        pdf_path=pdf_path,
        pdf_sha256=pdf_digest,
        manifest_path=manifest_path,
        category_count=len(categories),
        conversation_count=len(placements),
        expanded_conversation_count=expanded_conversation_count,
        unclassified_conversation_count=unclassified_count,
    )


_BOOK_CSS = r"""
@page {
  size: __PAGE_SIZE__;
  margin: 19mm 17mm 21mm;
  @top-left {
    content: string(book-title);
    color: #667085;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 7.5pt;
    letter-spacing: .08em;
    text-transform: uppercase;
  }
  @top-right {
    content: string(chapter-title);
    color: #667085;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 7.5pt;
  }
  @bottom-center {
    content: counter(page);
    color: #667085;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 8pt;
  }
}
@page :first { @top-left { content: none; } @top-right { content: none; } @bottom-center { content: none; } }
@page :blank { @top-left { content: none; } @top-right { content: none; } @bottom-center { content: none; } }
@page title { @top-left { content: none; } @top-right { content: none; } @bottom-center { content: none; } }

:root {
  --ink: #17212b;
  --ink-soft: #475467;
  --muted: #667085;
  --line: #d7dce2;
  --paper: #fffefa;
  --wash: #f4f1ea;
  --navy: #19324a;
  --navy-bright: #244f72;
  --copper: #a8643a;
  --copper-soft: #d8ad8f;
  color: var(--ink);
  background: var(--paper);
  font-family: Charter, "Bitstream Charter", "Iowan Old Style", Palatino, "Palatino Linotype", Georgia, serif;
  font-size: 10.4pt;
  line-height: 1.54;
  font-kerning: normal;
  font-variant-ligatures: common-ligatures;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; background: var(--paper); }
a { color: var(--navy-bright); text-decoration: none; }
p, li { orphans: 3; widows: 3; }
p { margin: 0 0 3.6mm; }
h1, h2, h3 { color: var(--navy); font-weight: 600; }
h1 { string-set: chapter-title content(); }
.title-page h1 { string-set: book-title content(); }

.half-title, .title-page, .colophon, .synthesis, .project-atlas, .toc { break-after: page; }
.half-title {
  min-height: 205mm;
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding-bottom: 28mm;
}
.half-title p, .kicker, .eyebrow {
  margin: 0 0 6mm;
  color: var(--copper);
  font-family: "Helvetica Neue", Arial, sans-serif;
  font-size: 8.5pt;
  font-weight: 600;
  letter-spacing: .18em;
  text-transform: uppercase;
}
.half-title h1 { max-width: 130mm; margin: 0; font-size: 31pt; line-height: 1.06; }
.title-page {
  page: title;
  min-height: 205mm;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.title-rule { width: 24mm; height: 1.4mm; margin-bottom: 15mm; background: var(--copper); }
.title-page h1 { max-width: 145mm; margin: 0; font-size: 37pt; line-height: 1.03; letter-spacing: -.025em; }
.subtitle { max-width: 135mm; margin-top: 9mm; color: var(--ink-soft); font-size: 15pt; line-height: 1.36; }
.author { margin-top: 22mm; color: var(--navy); font-size: 12.5pt; font-variant: small-caps; letter-spacing: .06em; }
.edition { margin-top: 4mm; color: var(--muted); font-family: "Helvetica Neue", Arial, sans-serif; font-size: 8.5pt; letter-spacing: .08em; text-transform: uppercase; }
.colophon { max-width: 125mm; padding-top: 30mm; color: var(--ink-soft); font-size: 9.5pt; }
.colophon h1 { margin: 0 0 10mm; font-size: 20pt; }

.synthesis { padding-top: 8mm; }
.synthesis > h1, .toc h1 { margin: 0 0 13mm; font-size: 29pt; line-height: 1.08; }
.synthesis > section { margin: 0 0 11mm; }
.synthesis h2 { margin: 0 0 4mm; font-size: 15pt; }
.synthesis > section:first-of-type p:first-of-type::first-letter {
  color: var(--copper);
  font-size: 1.6em;
  line-height: 1;
}

.toc { break-before: page; padding-top: 8mm; }
.toc ol { margin: 0; padding: 0; list-style: none; }
.toc li { margin: 0; border-bottom: .2mm solid var(--line); }
.toc a { display: grid; grid-template-columns: 12mm 1fr auto 7mm; gap: 3mm; padding: 3mm 0 2.5mm; color: var(--ink); }
.toc a::after { content: target-counter(attr(href), page); min-width: 7mm; color: var(--muted); text-align: right; }
.toc-number { color: var(--copper); font-family: "Helvetica Neue", Arial, sans-serif; font-size: 8pt; letter-spacing: .08em; }
.toc-title { font-size: 11pt; }
.toc-count { color: var(--muted); font-family: "Helvetica Neue", Arial, sans-serif; font-size: 8pt; }
.toc-depth-1 a { padding-left: 8mm; }
.toc-depth-2 a { padding-left: 16mm; }
.toc-depth-3 a { padding-left: 24mm; }

.category-opener {
  min-height: 175mm;
  break-before: page;
  display: flex;
  flex-direction: column;
  justify-content: center;
  border-top: 1.3mm solid var(--copper);
  padding: 18mm 0 10mm;
}
.category-opener.depth-0 { break-before: right; }
.category-opener h1 { max-width: 145mm; margin: 0; font-size: 31pt; line-height: 1.05; letter-spacing: -.02em; }
.category-opener.depth-1 h1 { font-size: 27pt; }
.category-opener.depth-2 h1, .category-opener.depth-3 h1 { font-size: 24pt; }
.category-stats { margin: 6mm 0 11mm; color: var(--muted); font-family: "Helvetica Neue", Arial, sans-serif; font-size: 8.5pt; letter-spacing: .06em; text-transform: uppercase; }
.category-summary { max-width: 145mm; color: var(--ink-soft); font-size: 12pt; line-height: 1.55; }
.category-summary p:first-child::first-letter {
  color: var(--copper);
  font-size: 1.6em;
  line-height: 1;
}
.trajectory { max-width: 145mm; margin: 7mm 0; padding: 5mm 6mm; border-left: 1mm solid var(--copper-soft); background: var(--wash); }
.trajectory h2 { margin: 0 0 2mm; font: 600 8pt/1.3 "Helvetica Neue", Arial, sans-serif; letter-spacing: .1em; text-transform: uppercase; }
.trajectory p:last-child { margin-bottom: 0; }
.category-links { margin-top: 6mm; font-family: "Helvetica Neue", Arial, sans-serif; font-size: 8.5pt; }
.category-links span { margin-right: 4mm; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
.category-links a { margin-right: 3mm; border-bottom: .2mm solid var(--copper-soft); }
.profile-list { max-width: 145mm; margin: 5mm 0; }
.profile-list h2 { margin: 0 0 2mm; font: 600 8pt/1.3 "Helvetica Neue", Arial, sans-serif; letter-spacing: .1em; text-transform: uppercase; }
.profile-list ul { margin: 0; padding-left: 5mm; }
.profile-list li { margin: 1.3mm 0; }

.project-atlas { min-height: 185mm; display: flex; flex-direction: column; justify-content: center; }
.project-atlas h1 { margin: 0 0 8mm; font-size: 31pt; line-height: 1.06; }
.project-introduction { max-width: 135mm; color: var(--ink-soft); font-size: 12pt; }
.editorial-note { max-width: 125mm; margin: 4mm 0 7mm; color: var(--muted); font-family: "Helvetica Neue", Arial, sans-serif; font-size: 8.5pt; font-style: italic; }
.project-profile { break-before: page; }
.project-profile > h1 { margin: 0 0 7mm; font-size: 27pt; line-height: 1.08; }
.project-overview { max-width: 145mm; font-size: 11.2pt; }
.current-state { margin: 7mm 0; padding: 5mm 6mm; border-left: 1mm solid var(--copper); background: var(--wash); }
.current-state h2 { margin: 0 0 2mm; font: 600 8pt/1.3 "Helvetica Neue", Arial, sans-serif; letter-spacing: .1em; text-transform: uppercase; }
.current-state p:last-child { margin-bottom: 0; }
.timeline { margin: 10mm 0; padding: 0; list-style: none; }
.timeline > li { position: relative; break-inside: avoid; page-break-inside: avoid; min-height: 8mm; padding: 0 0 7mm 10mm; }
.timeline > li::before { content: ""; position: absolute; left: 2.45mm; top: 2mm; bottom: -1mm; width: .3mm; background: var(--copper-soft); }
.timeline > li:last-child::before { display: none; }
.timeline-marker { position: absolute; z-index: 1; left: 0; top: 0; width: 5.2mm; height: 5.2mm; border: .8mm solid var(--copper); border-radius: 50%; background: var(--paper); }
.timeline time { color: var(--copper); font: 600 7.5pt/1.3 "Helvetica Neue", Arial, sans-serif; letter-spacing: .08em; text-transform: uppercase; }
.timeline h2 { margin: 1mm 0 2mm; font-size: 14pt; }
.event-sources { margin-top: 3mm; font-family: "Helvetica Neue", Arial, sans-serif; font-size: 7.8pt; }
.event-sources span { margin-right: 3mm; color: var(--muted); letter-spacing: .07em; text-transform: uppercase; }
.event-sources a { display: inline-block; margin-right: 3mm; border-bottom: .2mm solid var(--copper-soft); }

.chapter-directory { break-after: page; }
.chapter-directory h2 { margin: 0 0 7mm; font-size: 18pt; }
.chapter-directory ol { margin: 0; padding: 0; list-style: none; columns: 260px 2; column-gap: 8mm; }
.chapter-directory li { break-inside: avoid; border-bottom: .2mm solid var(--line); }
.chapter-directory a, .directory-record { display: block; padding: 2.2mm 0; color: var(--ink); }
.directory-record { color: var(--ink-soft); }
.chapter-directory time { display: block; color: var(--muted); font-family: "Helvetica Neue", Arial, sans-serif; font-size: 7.5pt; }

.conversation-entry { padding: 8mm 0; border-bottom: .3mm solid var(--line); }
.conversation-entry:last-child { border-bottom: 0; }
.conversation-header { margin-bottom: 7mm; padding-bottom: 6mm; border-bottom: .7mm solid var(--copper); }
.conversation-date { margin: 0 0 2mm; color: var(--copper); font-family: "Helvetica Neue", Arial, sans-serif; font-size: 8pt; font-weight: 600; letter-spacing: .11em; text-transform: uppercase; }
.conversation-header h2 { margin: 0; font-size: 23pt; line-height: 1.12; letter-spacing: -.012em; }
.conversation-status { margin: 3mm 0 0; color: var(--muted); font-family: "Helvetica Neue", Arial, sans-serif; font-size: 8.5pt; text-transform: capitalize; }
.synopsis { max-width: 145mm; margin: 0 0 7mm; font-size: 11.2pt; }
.synopsis.missing { color: var(--muted); font-style: italic; }
.metadata-row { display: flex; flex-wrap: wrap; align-items: baseline; gap: 1.8mm; margin: 3mm 0; }
.metadata-label { min-width: 24mm; color: var(--muted); font-family: "Helvetica Neue", Arial, sans-serif; font-size: 7.3pt; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }
.badge { display: inline-block; padding: 1.1mm 2.4mm; border: .2mm solid #ccd4dc; border-radius: 5mm; color: var(--ink-soft); font: 7.8pt/1.2 "Helvetica Neue", Arial, sans-serif; }
.badge.linked { color: var(--navy-bright); border-color: #aebfce; }
.cross-references { margin: 8mm 0; padding: 5mm 6mm; border-top: .3mm solid var(--line); border-bottom: .3mm solid var(--line); }
.cross-references h3 { margin: 0 0 3mm; font: 600 8pt/1.3 "Helvetica Neue", Arial, sans-serif; letter-spacing: .11em; text-transform: uppercase; }
.cross-references ul { margin: 0; padding-left: 5mm; }
.cross-references li { margin: 1.5mm 0; }
.cross-references li span { display: block; color: var(--muted); font-family: "Helvetica Neue", Arial, sans-serif; font-size: 7.5pt; }
.back-link { margin-top: 9mm; font-family: "Helvetica Neue", Arial, sans-serif; font-size: 7.5pt; text-transform: uppercase; letter-spacing: .08em; }

.transcript-section { margin-top: 9mm; }
.transcript-section > summary { margin-bottom: 5mm; color: var(--navy); font: 600 9pt/1.3 "Helvetica Neue", Arial, sans-serif; letter-spacing: .1em; text-transform: uppercase; }
.message { margin: 6mm 0; break-inside: auto; }
.message > header { display: flex; justify-content: space-between; margin-bottom: 2mm; color: var(--muted); font-family: "Helvetica Neue", Arial, sans-serif; font-size: 7.5pt; font-weight: 600; letter-spacing: .09em; text-transform: uppercase; }
.message > header time { font-weight: 400; letter-spacing: 0; text-transform: none; }
.message.user { padding: 5mm 6mm; border-left: 1mm solid var(--copper); background: var(--wash); }
.message.assistant { padding-left: 6mm; border-left: .4mm solid #b9c3cd; }
.message-body > :last-child { margin-bottom: 0; }
.asset-note, .empty-note { color: var(--muted); font-style: italic; }

blockquote { margin: 5mm 0; padding-left: 5mm; border-left: .8mm solid var(--copper-soft); color: var(--ink-soft); }
pre { padding: 4mm; border-radius: 1.5mm; background: #18232e; color: #f5f2ea; white-space: pre-wrap; overflow-wrap: anywhere; font: 8pt/1.45 "SFMono-Regular", Consolas, "Liberation Mono", monospace; }
code { overflow-wrap: anywhere; font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace; font-size: .88em; }
.code-label { margin: 4mm 0 -2mm; color: var(--muted); font: 7pt/1.2 "Helvetica Neue", Arial, sans-serif; letter-spacing: .09em; text-transform: uppercase; }
table { width: 100%; margin: 5mm 0; border-collapse: collapse; font-size: 8.5pt; }
th, td { padding: 2mm; border: .2mm solid var(--line); vertical-align: top; }
th { color: var(--navy); background: var(--wash); }

@media screen {
  body { width: calc(100% - 32px); max-width: 820px; margin: 28px auto; padding: clamp(22px, 6vw, 64px); box-shadow: 0 18px 70px rgba(23,33,43,.13); }
  .half-title, .title-page, .category-opener { min-height: 760px; }
  .toc a:hover, .chapter-directory a:hover, .back-link a:hover { color: var(--copper); }
}
"""
