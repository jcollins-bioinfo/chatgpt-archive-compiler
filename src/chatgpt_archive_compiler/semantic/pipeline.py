"""Resumable semantic-atlas orchestration over normalized Archive IR."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar, cast

from pydantic import BaseModel, JsonValue, ValidationError

from chatgpt_archive_compiler.exceptions import ApiBudgetExceededError
from chatgpt_archive_compiler.models import (
    Archive,
    ContentBlock,
    ContentBlockType,
    Conversation,
    Role,
)
from chatgpt_archive_compiler.semantic.errors import SemanticAtlasError
from chatgpt_archive_compiler.semantic.export import write_semantic_artifacts
from chatgpt_archive_compiler.semantic.graph import (
    build_similarity_graph,
    consolidate_communities,
    discover_communities,
)
from chatgpt_archive_compiler.semantic.models import (
    ArchiveSynthesisBundle,
    ArchiveSynthesisRequest,
    ArtifactRecord,
    Category,
    CategoryAssignment,
    CategoryDraft,
    ConversationAnalysis,
    ConversationReference,
    ConversationRepresentation,
    EmbeddingRecord,
    GraphEdge,
    GraphSummary,
    ReviewCode,
    ReviewItem,
    ReviewPriority,
    ReviewQueue,
    SemanticAtlas,
    SemanticAtlasManifest,
    SemanticAtlasOptions,
    SemanticAtlasResult,
    SemanticRunEstimate,
    Taxonomy,
    TaxonomyInterpretation,
    TaxonomyInterpretationRequest,
)
from chatgpt_archive_compiler.semantic.providers import (
    EmbeddingProvider,
    StructuredAnalysisProvider,
)
from chatgpt_archive_compiler.version import __version__

_REASONING_TYPES = {
    ContentBlockType.THINKING_TRACE,
    ContentBlockType.REASONING_SUMMARY,
}
_TEXT_TYPES = {
    ContentBlockType.TEXT,
    ContentBlockType.MARKDOWN,
    ContentBlockType.CODE,
    ContentBlockType.UNKNOWN,
}
_SPACE_PATTERN = re.compile(r"[ \t\f\v]+")
_ModelT = TypeVar("_ModelT", bound=BaseModel)
SemanticProgressCallback = Callable[[str, int, int, int], None]


def _report_progress(
    callback: SemanticProgressCallback | None,
    stage: str,
    completed: int,
    total: int,
    cached: int = 0,
) -> None:
    """Emit one content-free progress event and suppress callback-derived source details."""

    if callback is None:
        return
    try:
        callback(stage, completed, total, cached)
    except Exception as exc:
        raise SemanticAtlasError(
            f"Semantic progress callback failed safely ({type(exc).__name__})."
        ) from exc


def conversation_key_for(conversation: Conversation, source_index: int) -> str:
    """Return the stable opaque join key used throughout semantic artifacts."""

    identity = "\0".join(
        (
            "semantic-conversation-v1",
            str(source_index),
            str(conversation.source_path),
            conversation.conversation_id or "",
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _block_text(block: ContentBlock, *, include_reasoning: bool) -> str | None:
    """Select source text suitable for a user/assistant semantic representation."""

    if block.type in _REASONING_TYPES and not include_reasoning:
        return None
    if block.type in _TEXT_TYPES or (include_reasoning and block.type in _REASONING_TYPES):
        return block.text
    if block.type in {
        ContentBlockType.FILE_REFERENCE,
        ContentBlockType.IMAGE_REFERENCE,
        ContentBlockType.AUDIO_REFERENCE,
    }:
        label = block.type.value.replace("_", " ")
        return f"[{label} omitted]"
    return None


def _normalize_text(text: str) -> str:
    """Normalize separators and horizontal whitespace without rewriting semantics."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u2028", "\n").replace("\u2029", "\n\n")
    lines = [_SPACE_PATTERN.sub(" ", line).strip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()


def _conversation_text(conversation: Conversation, *, include_reasoning: bool) -> tuple[str, int]:
    """Build deterministic role-labelled text from current-path user/assistant messages."""

    rendered: list[str] = []
    message_count = 0
    for message in conversation.current_path_messages:
        if message.role not in {Role.USER, Role.ASSISTANT}:
            continue
        blocks = [
            text
            for block in message.content
            if (text := _block_text(block, include_reasoning=include_reasoning))
        ]
        body = _normalize_text("\n\n".join(blocks))
        if not body:
            continue
        rendered.append(f"{message.role.value.title()}:\n{body}")
        message_count += 1
    return "\n\n".join(rendered), message_count


def _truncate_representation(text: str, limit: int) -> tuple[str, bool]:
    """Retain both beginning and ending context when a representation exceeds its limit."""

    if len(text) <= limit:
        return text, False
    marker = "\n\n[... middle omitted by deterministic character limit ...]\n\n"
    remaining = max(0, limit - len(marker))
    head_size = (remaining * 2) // 3
    tail_size = remaining - head_size
    return f"{text[:head_size]}{marker}{text[-tail_size:] if tail_size else ''}", True


def prepare_conversation_representations(
    archive: Archive,
    options: SemanticAtlasOptions | None = None,
) -> tuple[ConversationRepresentation, ...]:
    """Create stable current-path user/assistant representations.

    Reasoning and non-user-facing roles are excluded by default.
    """

    active_options = options or SemanticAtlasOptions()
    indexed = list(enumerate(archive.conversations))
    indexed.sort(
        key=lambda item: (
            (item[1].created_at or item[1].updated_at) is None,
            item[1].created_at or item[1].updated_at or datetime.max.replace(tzinfo=UTC),
            item[0],
        )
    )
    if active_options.max_conversations is not None:
        indexed = indexed[: active_options.max_conversations]

    representations: list[ConversationRepresentation] = []
    for source_index, conversation in indexed:
        body, message_count = _conversation_text(
            conversation,
            include_reasoning=active_options.include_reasoning,
        )
        title = _normalize_text(conversation.title or "Untitled conversation")
        complete_text = f"Title:\n{title}\n\n{body}" if body else f"Title:\n{title}"
        original_count = len(complete_text)
        text, truncated = _truncate_representation(
            complete_text,
            active_options.max_characters_per_conversation,
        )
        representations.append(
            ConversationRepresentation(
                conversation_key=conversation_key_for(conversation, source_index),
                source_index=source_index,
                source_conversation_id=conversation.conversation_id,
                title=title,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
                text=text,
                message_count=message_count,
                character_count=len(text),
                original_character_count=original_count,
                truncated=truncated,
                text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        )
    return tuple(representations)


def _estimate_representations(
    representations: Sequence[ConversationRepresentation],
) -> SemanticRunEstimate:
    """Summarize provider input size without retaining any source-derived strings."""

    character_count = sum(item.character_count for item in representations)
    return SemanticRunEstimate(
        conversation_count=len(representations),
        message_count=sum(item.message_count for item in representations),
        character_count=character_count,
        original_character_count=sum(item.original_character_count for item in representations),
        approximate_input_tokens=(character_count + 3) // 4,
        embedding_item_count=len(representations),
        analysis_item_count=len(representations),
        truncated_conversation_count=sum(item.truncated for item in representations),
    )


def estimate_semantic_run(
    archive: Archive,
    options: SemanticAtlasOptions | None = None,
) -> SemanticRunEstimate:
    """Return a content-free volume estimate before any external provider is invoked."""

    return _estimate_representations(prepare_conversation_representations(archive, options))


def _canonical_json(value: object) -> str:
    """Serialize a JSON-compatible value deterministically for cache keys."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _hash_value(value: object) -> str:
    """Return the SHA-256 of one canonical JSON value."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _load_record_cache(path: Path, model_type: type[_ModelT]) -> dict[str, tuple[str, _ModelT]]:
    """Load valid completed JSONL records, ignoring a possible interrupted final line."""

    records: dict[str, tuple[str, _ModelT]] = {}
    if not path.is_file():
        return records
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    wrapper = json.loads(line)
                    conversation_key = str(wrapper["conversation_key"])
                    cache_key = str(wrapper["cache_key"])
                    record = model_type.model_validate(wrapper["record"])
                    records[conversation_key] = (cache_key, record)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError, ValidationError):
                    continue
    except OSError as exc:
        raise SemanticAtlasError("Could not read a semantic cache file.") from exc
    return records


def _append_record_cache_batch(
    path: Path,
    cache_keys: Mapping[str, str],
    records: Sequence[EmbeddingRecord | ConversationAnalysis],
) -> None:
    """Append and fsync one completed provider batch for interruption-safe resume."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            for record in records:
                wrapper = {
                    "conversation_key": record.conversation_key,
                    "cache_key": cache_keys[record.conversation_key],
                    "record": record.model_dump(mode="json"),
                }
                handle.write(_canonical_json(wrapper))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise SemanticAtlasError("Could not checkpoint a semantic provider result.") from exc


def _atomic_json(path: Path, value: BaseModel | Mapping[str, object]) -> Path:
    """Atomically write canonical pretty JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        data: object = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        return path
    except (OSError, TypeError, ValueError) as exc:
        raise SemanticAtlasError("Could not write a semantic JSON artifact.") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_stage_cache(path: Path, cache_key: str, model_type: type[_ModelT]) -> _ModelT | None:
    """Load a whole-stage cache object only when its request fingerprint matches."""

    if not path.is_file():
        return None
    try:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        if wrapper.get("cache_key") != cache_key:
            return None
        return model_type.model_validate(wrapper["record"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, ValidationError):
        return None


def _write_stage_cache(path: Path, cache_key: str, record: BaseModel) -> None:
    """Atomically checkpoint one complete taxonomy or synthesis provider response."""

    _atomic_json(
        path,
        {
            "cache_key": cache_key,
            "record": record.model_dump(mode="json"),
        },
    )


def _batched(items: Sequence[_ModelT], size: int) -> Iterable[Sequence[_ModelT]]:
    """Yield stable non-empty slices of a model sequence."""

    for start in range(0, len(items), size):
        yield items[start : start + size]


def _expected_record_keys(
    records: Sequence[EmbeddingRecord | ConversationAnalysis],
) -> tuple[str, ...]:
    """Extract provider response keys with a safe invariant error on malformed objects."""

    try:
        return tuple(record.conversation_key for record in records)
    except (AttributeError, TypeError) as exc:
        raise SemanticAtlasError("A provider response omitted its conversation key.") from exc


def _get_embeddings(
    representations: Sequence[ConversationRepresentation],
    provider: EmbeddingProvider,
    *,
    options: SemanticAtlasOptions,
    cache_directory: Path,
    progress_callback: SemanticProgressCallback | None,
) -> tuple[EmbeddingRecord, ...]:
    """Load or obtain all embeddings with per-batch durable checkpoints."""

    cache_path = cache_directory / "embeddings.jsonl"
    cached = _load_record_cache(cache_path, EmbeddingRecord) if options.cache_enabled else {}
    results: dict[str, EmbeddingRecord] = {}
    missing: list[ConversationRepresentation] = []
    cache_keys: dict[str, str] = {}
    context = f"embedding-v1\0{provider.provider_name}\0{provider.model_name}\0"
    for item in representations:
        cache_key = hashlib.sha256(f"{context}{item.text_sha256}".encode()).hexdigest()
        cache_keys[item.conversation_key] = cache_key
        cached_item = cached.get(item.conversation_key)
        if cached_item is not None and cached_item[0] == cache_key:
            results[item.conversation_key] = cached_item[1]
        else:
            missing.append(item)
    cached_count = len(results)
    _report_progress(
        progress_callback,
        "embeddings",
        cached_count,
        len(representations),
        cached_count,
    )
    for batch in _batched(missing, options.embedding_batch_size):
        try:
            response = tuple(provider.embed(batch))
        except ApiBudgetExceededError:
            raise
        except Exception as exc:
            raise SemanticAtlasError(
                f"Embedding provider failed safely ({type(exc).__name__})."
            ) from exc
        expected = tuple(item.conversation_key for item in batch)
        actual = _expected_record_keys(response)
        if len(set(actual)) != len(actual) or set(actual) != set(expected):
            raise SemanticAtlasError("Embedding provider keys did not match the requested batch.")
        for record in response:
            results[record.conversation_key] = record
        if options.cache_enabled:
            _append_record_cache_batch(cache_path, cache_keys, response)
        _report_progress(
            progress_callback,
            "embeddings",
            len(results),
            len(representations),
            cached_count,
        )
    ordered = tuple(results[item.conversation_key] for item in representations)
    dimensions = {len(record.vector) for record in ordered}
    if len(dimensions) > 1:
        raise SemanticAtlasError("Embedding provider returned inconsistent vector dimensions.")
    return ordered


def _get_analyses(
    representations: Sequence[ConversationRepresentation],
    provider: StructuredAnalysisProvider,
    *,
    options: SemanticAtlasOptions,
    cache_directory: Path,
    progress_callback: SemanticProgressCallback | None,
    cache_filename: str = "analyses.jsonl",
    progress_stage: str = "conversation_profiles",
) -> tuple[ConversationAnalysis, ...]:
    """Load or obtain per-conversation profiles with batch-level resume."""

    cache_path = cache_directory / cache_filename
    cached = _load_record_cache(cache_path, ConversationAnalysis) if options.cache_enabled else {}
    results: dict[str, ConversationAnalysis] = {}
    missing: list[ConversationRepresentation] = []
    cache_keys: dict[str, str] = {}
    context = f"analysis-v1\0{provider.provider_name}\0{provider.model_name}\0"
    for item in representations:
        cache_key = hashlib.sha256(f"{context}{item.text_sha256}".encode()).hexdigest()
        cache_keys[item.conversation_key] = cache_key
        cached_item = cached.get(item.conversation_key)
        if cached_item is not None and cached_item[0] == cache_key:
            results[item.conversation_key] = cached_item[1]
        else:
            missing.append(item)
    cached_count = len(results)
    _report_progress(
        progress_callback,
        progress_stage,
        cached_count,
        len(representations),
        cached_count,
    )
    for batch in _batched(missing, options.analysis_batch_size):
        try:
            response = tuple(provider.analyze_conversations(batch))
        except ApiBudgetExceededError:
            raise
        except Exception as exc:
            raise SemanticAtlasError(
                f"Analysis provider failed safely ({type(exc).__name__})."
            ) from exc
        expected = tuple(item.conversation_key for item in batch)
        actual = _expected_record_keys(response)
        if len(set(actual)) != len(actual) or set(actual) != set(expected):
            raise SemanticAtlasError("Analysis provider keys did not match the requested batch.")
        for record in response:
            results[record.conversation_key] = record
        if options.cache_enabled:
            _append_record_cache_batch(cache_path, cache_keys, response)
        _report_progress(
            progress_callback,
            progress_stage,
            len(results),
            len(representations),
            cached_count,
        )
    return tuple(results[item.conversation_key] for item in representations)


def _category_drafts(
    memberships: Mapping[str, str],
    analyses: Sequence[ConversationAnalysis],
    representations: Sequence[ConversationRepresentation],
    edges: Sequence[GraphEdge],
) -> tuple[CategoryDraft, ...]:
    """Summarize local communities without resending full conversation representations."""

    analyses_by_key = {item.conversation_key: item for item in analyses}
    representations_by_key = {item.conversation_key: item for item in representations}
    members: defaultdict[str, list[str]] = defaultdict(list)
    degrees: defaultdict[str, float] = defaultdict(float)
    for key, category_id in memberships.items():
        members[category_id].append(key)
    for edge in edges:
        degrees[edge.source_key] += edge.similarity
        degrees[edge.target_key] += edge.similarity

    drafts: list[CategoryDraft] = []
    for category_id in sorted(members):
        keys = tuple(sorted(members[category_id]))
        labels: Counter[str] = Counter()
        dates: list[datetime] = []
        for key in keys:
            analysis = analyses_by_key[key]
            label_weight = 4 if analysis.confidence > 0.5 else 1
            for value in (
                analysis.primary_subject,
                *analysis.secondary_subjects,
                *analysis.projects,
                *analysis.recurring_themes,
            ):
                if value.strip():
                    labels[value.strip()] += label_weight
            representation = representations_by_key[key]
            date = representation.created_at or representation.updated_at
            if date is not None:
                dates.append(date)
        representatives = tuple(
            sorted(keys, key=lambda key: (-degrees[key], key))[: min(8, len(keys))]
        )
        drafts.append(
            CategoryDraft(
                category_id=category_id,
                conversation_keys=keys,
                representative_conversation_keys=representatives,
                candidate_labels=tuple(
                    label
                    for label, _ in sorted(labels.items(), key=lambda item: (-item[1], item[0]))[
                        :12
                    ]
                ),
                earliest_at=min(dates) if dates else None,
                latest_at=max(dates) if dates else None,
            )
        )
    return tuple(drafts)


def _select_refinement_keys(
    drafts: Sequence[CategoryDraft],
    representations: Sequence[ConversationRepresentation],
    *,
    per_category: int,
    maximum: int,
) -> tuple[str, ...]:
    """Select central and time-spanning representatives fairly across all categories."""

    if maximum <= 0:
        return ()
    representations_by_key = {item.conversation_key: item for item in representations}
    candidates_by_category: dict[str, tuple[str, ...]] = {}
    for draft in sorted(drafts, key=lambda item: item.category_id):
        chronological = sorted(
            draft.conversation_keys,
            key=lambda key: (
                (representations_by_key[key].created_at or representations_by_key[key].updated_at)
                is None,
                representations_by_key[key].created_at
                or representations_by_key[key].updated_at
                or datetime.max.replace(tzinfo=UTC),
                key,
            ),
        )
        preferred = [
            *draft.representative_conversation_keys[:1],
            *chronological[:1],
            *chronological[-1:],
            *draft.representative_conversation_keys[1:],
            *chronological,
        ]
        unique: list[str] = []
        seen: set[str] = set()
        for key in preferred:
            if key not in seen:
                unique.append(key)
                seen.add(key)
            if len(unique) == per_category:
                break
        candidates_by_category[draft.category_id] = tuple(unique)

    selected: list[str] = []
    for offset in range(per_category):
        for category_id in sorted(candidates_by_category):
            candidates = candidates_by_category[category_id]
            if offset < len(candidates):
                selected.append(candidates[offset])
                if len(selected) == maximum:
                    return tuple(selected)
    return tuple(selected)


def _parent_id(name: str) -> str:
    """Return a deterministic identifier for a provider-named broad domain."""

    normalized = " ".join(name.casefold().split())
    return f"domain-{hashlib.sha256(normalized.encode()).hexdigest()[:12]}"


def _build_taxonomy(
    drafts: Sequence[CategoryDraft],
    interpretation: TaxonomyInterpretation,
    analyses: Sequence[ConversationAnalysis],
    edges: Sequence[GraphEdge],
) -> Taxonomy:
    """Combine graph memberships and provider labels into a validated two-level hierarchy."""

    draft_ids = {draft.category_id for draft in drafts}
    interpretations = {item.category_id: item for item in interpretation.categories}
    if len(interpretations) != len(interpretation.categories) or set(interpretations) != draft_ids:
        raise SemanticAtlasError("Taxonomy interpretation did not name every discovered category.")
    draft_by_id = {draft.category_id: draft for draft in drafts}
    parent_key_by_category: dict[str, str] = {}
    parent_name_by_key: dict[str, str] = {}
    for category_id in sorted(interpretations):
        display_name = " ".join(interpretations[category_id].parent_name.split()) or "Other"
        canonical_key = display_name.casefold()
        parent_key_by_category[category_id] = canonical_key
        parent_name_by_key.setdefault(canonical_key, display_name)
    parent_keys = sorted(
        parent_name_by_key, key=lambda key: (parent_name_by_key[key].casefold(), key)
    )
    parent_ids = {key: _parent_id(key) for key in parent_keys}
    children_by_parent: defaultdict[str, list[str]] = defaultdict(list)
    for category_id, parent_key in parent_key_by_category.items():
        children_by_parent[parent_ids[parent_key]].append(category_id)

    categories: list[Category] = []
    for parent_key in parent_keys:
        parent_name = parent_name_by_key[parent_key]
        parent_id = parent_ids[parent_key]
        child_ids = tuple(sorted(children_by_parent[parent_id]))
        conversation_keys = tuple(
            sorted(
                {key for child_id in child_ids for key in draft_by_id[child_id].conversation_keys}
            )
        )
        categories.append(
            Category(
                category_id=parent_id,
                name=parent_name,
                description=f"Broad domain containing {len(child_ids)} discovered categories.",
                level=0,
                conversation_keys=conversation_keys,
                child_ids=child_ids,
                confidence=sum(interpretations[child_id].confidence for child_id in child_ids)
                / len(child_ids),
            )
        )
    for category_id in sorted(draft_ids):
        item = interpretations[category_id]
        related = tuple(
            related
            for related in item.related_category_ids
            if related in draft_ids and related != category_id
        )
        categories.append(
            Category(
                category_id=category_id,
                name=item.name,
                description=item.description,
                parent_id=parent_ids[parent_key_by_category[category_id]],
                level=1,
                conversation_keys=draft_by_id[category_id].conversation_keys,
                defining_concepts=item.defining_concepts,
                related_category_ids=tuple(sorted(set(related))),
                confidence=item.confidence,
            )
        )

    membership = {key: draft.category_id for draft in drafts for key in draft.conversation_keys}
    neighbor_scores: defaultdict[str, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    own_scores: defaultdict[str, float] = defaultdict(float)
    for edge in edges:
        source_category = membership[edge.source_key]
        target_category = membership[edge.target_key]
        if source_category == target_category:
            own_scores[edge.source_key] += edge.similarity
            own_scores[edge.target_key] += edge.similarity
        else:
            neighbor_scores[edge.source_key][target_category] += edge.similarity
            neighbor_scores[edge.target_key][source_category] += edge.similarity
    analysis_by_key = {item.conversation_key: item for item in analyses}
    assignments: list[CategoryAssignment] = []
    for key in sorted(membership):
        category_id = membership[key]
        alternatives = neighbor_scores[key]
        strongest_external = max(alternatives.values(), default=0.0)
        own = own_scores[key]
        total = own + sum(alternatives.values())
        structural_confidence = own / total if total else 0.5
        confidence = (
            0.45 * analysis_by_key[key].confidence
            + 0.35 * interpretations[category_id].confidence
            + 0.20 * structural_confidence
        )
        secondary = tuple(
            candidate
            for candidate, score in sorted(
                alternatives.items(), key=lambda item: (-item[1], item[0])
            )
            if score >= max(0.01, strongest_external * 0.60)
        )[:2]
        assignments.append(
            CategoryAssignment(
                conversation_key=key,
                primary_category_id=category_id,
                secondary_category_ids=secondary,
                confidence=max(0.0, min(1.0, confidence)),
                rationale="Combined profile confidence and weighted graph-community evidence.",
            )
        )
    category_ids = [category.category_id for category in categories]
    if len(category_ids) != len(set(category_ids)):
        raise SemanticAtlasError("Taxonomy construction produced duplicate category identifiers.")
    return Taxonomy(categories=tuple(categories), assignments=tuple(assignments))


def _review_queue(
    analyses: Sequence[ConversationAnalysis],
    taxonomy: Taxonomy,
    edges: Sequence[GraphEdge],
    *,
    threshold: float,
    source_community_counts: Mapping[str, int] | None = None,
) -> ReviewQueue:
    """Build content-free, keyed quality-control signals for human review."""

    items: list[ReviewItem] = []
    connected = {edge.source_key for edge in edges} | {edge.target_key for edge in edges}
    for analysis in analyses:
        if analysis.confidence < threshold:
            items.append(
                ReviewItem(
                    code=ReviewCode.LOW_ANALYSIS_CONFIDENCE,
                    priority=ReviewPriority.HIGH,
                    conversation_keys=(analysis.conversation_key,),
                    message="Conversation profile confidence is below the configured threshold.",
                    diagnostics={"confidence": analysis.confidence, "threshold": threshold},
                )
            )
        if analysis.conversation_key not in connected:
            items.append(
                ReviewItem(
                    code=ReviewCode.ISOLATED_CONVERSATION,
                    priority=ReviewPriority.MEDIUM,
                    conversation_keys=(analysis.conversation_key,),
                    message="Conversation has no retained semantic-graph edge.",
                )
            )
    for assignment in taxonomy.assignments:
        if assignment.confidence < threshold:
            items.append(
                ReviewItem(
                    code=ReviewCode.LOW_CATEGORY_CONFIDENCE,
                    priority=ReviewPriority.HIGH,
                    conversation_keys=(assignment.conversation_key,),
                    category_ids=(assignment.primary_category_id,),
                    message="Primary category confidence is below the configured threshold.",
                    diagnostics={"confidence": assignment.confidence, "threshold": threshold},
                )
            )
        if assignment.secondary_category_ids:
            items.append(
                ReviewItem(
                    code=ReviewCode.AMBIGUOUS_ASSIGNMENT,
                    priority=ReviewPriority.LOW,
                    conversation_keys=(assignment.conversation_key,),
                    category_ids=(
                        assignment.primary_category_id,
                        *assignment.secondary_category_ids,
                    ),
                    message="Conversation has meaningful graph links to secondary categories.",
                    diagnostics={
                        "secondary_category_count": len(assignment.secondary_category_ids)
                    },
                )
            )
    for category in taxonomy.categories:
        if category.level == 1 and len(category.conversation_keys) == 1:
            items.append(
                ReviewItem(
                    code=ReviewCode.SINGLETON_CATEGORY,
                    priority=ReviewPriority.MEDIUM,
                    conversation_keys=category.conversation_keys,
                    category_ids=(category.category_id,),
                    message="Discovered category contains only one conversation.",
                )
            )
    for category_id, source_count in sorted((source_community_counts or {}).items()):
        if source_count > 1:
            items.append(
                ReviewItem(
                    code=ReviewCode.CONSOLIDATED_CATEGORY,
                    priority=ReviewPriority.MEDIUM,
                    category_ids=(category_id,),
                    message=(
                        "Category combines multiple graph-discovered communities to satisfy "
                        "the configured leaf-category limit."
                    ),
                    diagnostics={"source_community_count": source_count},
                )
            )
    counts_by_code = Counter(item.code.value for item in items)
    counts_by_priority = Counter(item.priority.value for item in items)
    return ReviewQueue(
        items=tuple(items),
        counts_by_code=dict(sorted(counts_by_code.items())),
        counts_by_priority=dict(sorted(counts_by_priority.items())),
    )


def _graph_summary(
    conversation_keys: Sequence[str], edges: Sequence[GraphEdge], taxonomy: Taxonomy
) -> GraphSummary:
    """Return privacy-safe graph counts for synthesis and provenance."""

    connected = {edge.source_key for edge in edges} | {edge.target_key for edge in edges}
    leaf_count = sum(category.level == 1 for category in taxonomy.categories)
    return GraphSummary(
        node_count=len(conversation_keys),
        edge_count=len(edges),
        category_count=leaf_count,
        isolated_node_count=sum(key not in connected for key in conversation_keys),
    )


def _provider_stage(
    *,
    cache_path: Path,
    cache_key: str,
    model_type: type[_ModelT],
    enabled: bool,
    call: Callable[[], _ModelT],
    stage: str,
    progress_callback: SemanticProgressCallback | None,
) -> _ModelT:
    """Load or execute a whole-provider stage with a safe exception wrapper."""

    if enabled:
        cached = _load_stage_cache(cache_path, cache_key, model_type)
        if cached is not None:
            _report_progress(progress_callback, stage, 1, 1, 1)
            return cached
    _report_progress(progress_callback, stage, 0, 1)
    try:
        record = call()
    except ApiBudgetExceededError:
        raise
    except Exception as exc:
        raise SemanticAtlasError(
            f"Structured analysis provider failed safely ({type(exc).__name__})."
        ) from exc
    if enabled:
        _write_stage_cache(cache_path, cache_key, record)
    _report_progress(progress_callback, stage, 1, 1)
    return record


def _source_fingerprint(
    archive: Archive, representations: Sequence[ConversationRepresentation]
) -> str:
    """Fingerprint source provenance without exposing content, names, IDs, or paths."""

    declared = archive.source_manifest.archive_sha256
    if declared:
        return declared
    return hashlib.sha256(
        "\0".join(item.text_sha256 for item in representations).encode("ascii")
    ).hexdigest()


def _artifact_record(path: Path) -> ArtifactRecord:
    """Compute integrity metadata for a completed public/private output artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return ArtifactRecord(
        filename=path.name, sha256=digest.hexdigest(), size_bytes=path.stat().st_size
    )


def build_semantic_atlas(
    archive: Archive,
    output_directory: str | Path,
    *,
    embedding_provider: EmbeddingProvider,
    analysis_provider: StructuredAnalysisProvider,
    refinement_provider: StructuredAnalysisProvider | None = None,
    interpretation_provider: StructuredAnalysisProvider | None = None,
    options: SemanticAtlasOptions | None = None,
    progress_callback: SemanticProgressCallback | None = None,
) -> SemanticAtlasResult:
    """Build, checkpoint, validate, and export a complete semantic conversation atlas.

    ``analysis_provider`` profiles every conversation and is therefore normally local for large
    archives. ``refinement_provider`` optionally replaces only a bounded, graph-selected set of
    representative profiles. ``interpretation_provider`` names categories and performs the single
    archive synthesis; it defaults to the refinement provider and then to the baseline provider.
    """

    active_options = options or SemanticAtlasOptions()
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cache_directory = destination / ".semantic_cache"
    representations = prepare_conversation_representations(archive, active_options)
    _report_progress(
        progress_callback,
        "representations",
        len(representations),
        len(representations),
    )
    estimate = _estimate_representations(representations)
    if not representations:
        raise SemanticAtlasError("The selected archive contains no conversations to analyze.")

    embeddings = _get_embeddings(
        representations,
        embedding_provider,
        options=active_options,
        cache_directory=cache_directory,
        progress_callback=progress_callback,
    )
    analyses = _get_analyses(
        representations,
        analysis_provider,
        options=active_options,
        cache_directory=cache_directory,
        progress_callback=progress_callback,
    )
    _report_progress(progress_callback, "semantic_graph", 0, 1)
    analyses_by_key = {analysis.conversation_key: analysis for analysis in analyses}
    edges = build_similarity_graph(
        embeddings,
        options=active_options,
        analyses=analyses_by_key,
    )
    keys = tuple(item.conversation_key for item in representations)
    memberships = discover_communities(
        keys,
        edges,
        resolution=active_options.community_resolution,
    )
    consolidation = consolidate_communities(
        memberships,
        embeddings,
        max_categories=active_options.max_leaf_categories,
    )
    memberships = consolidation.memberships
    _report_progress(progress_callback, "semantic_graph", 1, 1)
    drafts = _category_drafts(memberships, analyses, representations, edges)
    if refinement_provider is not None and active_options.max_refined_conversations > 0:
        selected_keys = _select_refinement_keys(
            drafts,
            representations,
            per_category=active_options.refined_conversations_per_category,
            maximum=active_options.max_refined_conversations,
        )
        representations_by_key = {item.conversation_key: item for item in representations}
        refined = _get_analyses(
            tuple(representations_by_key[key] for key in selected_keys),
            refinement_provider,
            options=active_options,
            cache_directory=cache_directory,
            progress_callback=progress_callback,
            cache_filename="refined_analyses.jsonl",
            progress_stage="representative_profiles",
        )
        refined_by_key = {item.conversation_key: item for item in refined}
        analyses = tuple(refined_by_key.get(item.conversation_key, item) for item in analyses)
        drafts = _category_drafts(memberships, analyses, representations, edges)
    active_interpretation_provider = (
        interpretation_provider or refinement_provider or analysis_provider
    )
    taxonomy_request = TaxonomyInterpretationRequest(clusters=drafts, analyses=analyses)
    taxonomy_cache_key = _hash_value(
        {
            "stage": "taxonomy-v1",
            "provider": active_interpretation_provider.provider_name,
            "model": active_interpretation_provider.model_name,
            "request": taxonomy_request.model_dump(mode="json"),
        }
    )
    interpretation = _provider_stage(
        cache_path=cache_directory / "taxonomy.json",
        cache_key=taxonomy_cache_key,
        model_type=TaxonomyInterpretation,
        enabled=active_options.cache_enabled,
        call=lambda: active_interpretation_provider.interpret_taxonomy(taxonomy_request),
        stage="taxonomy",
        progress_callback=progress_callback,
    )
    taxonomy = _build_taxonomy(drafts, interpretation, analyses, edges)
    summary = _graph_summary(keys, edges, taxonomy)
    synthesis_request = ArchiveSynthesisRequest(
        analyses=analyses,
        references=tuple(
            ConversationReference(
                conversation_key=item.conversation_key,
                title=item.title,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
            for item in representations
        ),
        taxonomy=taxonomy,
        graph_summary=summary,
    )
    synthesis_cache_key = _hash_value(
        {
            "stage": "synthesis-v1",
            "provider": active_interpretation_provider.provider_name,
            "model": active_interpretation_provider.model_name,
            "request": synthesis_request.model_dump(mode="json"),
        }
    )
    bundle = _provider_stage(
        cache_path=cache_directory / "synthesis.json",
        cache_key=synthesis_cache_key,
        model_type=ArchiveSynthesisBundle,
        enabled=active_options.cache_enabled,
        call=lambda: active_interpretation_provider.synthesize_archive(synthesis_request),
        stage="archive_synthesis",
        progress_callback=progress_callback,
    )
    review_queue = _review_queue(
        analyses,
        taxonomy,
        edges,
        threshold=active_options.review_confidence_threshold,
        source_community_counts=consolidation.source_community_counts,
    )
    atlas = SemanticAtlas(
        representations=representations,
        analyses=analyses,
        embeddings=embeddings,
        graph_edges=edges,
        taxonomy=taxonomy,
        category_profiles=bundle.category_profiles,
        project_timelines=bundle.project_timelines,
        synthesis=bundle.synthesis,
        review_queue=review_queue,
    )
    _report_progress(progress_callback, "artifact_export", 0, 1)
    paths = write_semantic_artifacts(atlas, destination)
    artifact_paths = tuple(path for name, path in paths.items() if name != "manifest")
    options_json = cast(dict[str, JsonValue], active_options.model_dump(mode="json"))
    manifest = SemanticAtlasManifest(
        atlas_version=atlas.atlas_version,
        package_version=__version__,
        created_at=datetime.now(UTC),
        source_archive_version=archive.archive_version,
        source_fingerprint=_source_fingerprint(archive, representations),
        embedding_provider=embedding_provider.provider_name,
        embedding_model=embedding_provider.model_name,
        analysis_provider=(
            f"baseline={analysis_provider.provider_name};"
            f"refinement={refinement_provider.provider_name if refinement_provider else 'none'};"
            f"interpretation={active_interpretation_provider.provider_name}"
        ),
        analysis_model=(
            f"baseline={analysis_provider.model_name};"
            f"refinement={refinement_provider.model_name if refinement_provider else 'none'};"
            f"interpretation={active_interpretation_provider.model_name}"
        ),
        options=options_json,
        estimate=estimate,
        graph_summary=summary,
        review_counts_by_code=review_queue.counts_by_code,
        artifacts=tuple(_artifact_record(path) for path in artifact_paths),
    )
    manifest_path = _atomic_json(destination / "semantic_manifest.json", manifest)
    _report_progress(progress_callback, "artifact_export", 1, 1)
    return SemanticAtlasResult(
        output_directory=destination,
        catalog_path=paths["catalog"],
        embeddings_path=paths["embeddings"],
        graph_path=paths["graph"],
        taxonomy_path=paths["taxonomy"],
        category_profiles_path=paths["category_profiles"],
        project_timelines_path=paths["project_timelines"],
        synthesis_path=paths["synthesis"],
        review_queue_path=paths["review_queue"],
        atlas_html_path=paths["atlas_html"],
        manifest_path=manifest_path,
        estimate=estimate,
        atlas=atlas,
    )
