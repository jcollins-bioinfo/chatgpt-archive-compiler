"""Loss-aware normalization of ChatGPT conversation graph payloads."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any, cast

from pydantic import JsonValue

from chatgpt_archive_compiler.exceptions import ArchiveLimitError, UnsupportedSchemaError
from chatgpt_archive_compiler.models import (
    ArchiveWarning,
    ContentBlock,
    ContentBlockType,
    Conversation,
    IngestLimits,
    Message,
    MessageNode,
    NormalizationResult,
    Role,
    WarningSeverity,
)


@dataclass(slots=True)
class _WarningBudget:
    """Bound detailed warnings so malformed payloads cannot exhaust memory."""

    remaining: int
    suppressed: int = 0

    def add(self, target: list[ArchiveWarning], warning: ArchiveWarning) -> None:
        """Append a warning when capacity remains, otherwise count it as suppressed."""

        if self.remaining > 0:
            target.append(warning)
            self.remaining -= 1
        else:
            self.suppressed += 1


def _json_pointer_part(value: str) -> str:
    """Escape one JSON Pointer component according to RFC 6901."""

    return value.replace("~", "~0").replace("/", "~1")


def _as_json_value(value: Any) -> JsonValue:
    """Return JSON-compatible source data without using lossy Python ``repr`` output."""

    if value is None or isinstance(value, str | bool | int):
        return cast(JsonValue, value)
    if isinstance(value, float):
        return cast(JsonValue, value if math.isfinite(value) else None)
    if isinstance(value, list | tuple):
        return cast(JsonValue, [_as_json_value(item) for item in value])
    if isinstance(value, dict):
        return cast(
            JsonValue,
            {str(key): _as_json_value(item) for key, item in value.items() if isinstance(key, str)},
        )
    return cast(JsonValue, {"unsupported_python_type": type(value).__name__})


def _extras(record: dict[str, Any], known_keys: set[str]) -> dict[str, JsonValue]:
    """Copy unknown source fields into a JSON-compatible extras mapping."""

    return {key: _as_json_value(value) for key, value in record.items() if key not in known_keys}


def _warning(
    *,
    code: str,
    message: str,
    source_path: PurePosixPath,
    location: str,
    conversation_id: str | None = None,
    node_id: str | None = None,
    severity: WarningSeverity = WarningSeverity.WARNING,
    context: dict[str, JsonValue] | None = None,
) -> ArchiveWarning:
    """Construct a location-bearing warning without source text excerpts."""

    return ArchiveWarning(
        severity=severity,
        code=code,
        message=message,
        source_path=source_path,
        conversation_id=conversation_id,
        node_id=node_id,
        location=location,
        context=context or {},
    )


def _timestamp(
    value: Any,
    *,
    source_path: PurePosixPath,
    location: str,
    conversation_id: str | None,
    node_id: str | None,
    warnings: list[ArchiveWarning],
    budget: _WarningBudget,
) -> datetime | None:
    """Normalize a numeric epoch or ISO string to an aware UTC datetime."""

    if value is None:
        return None
    parsed: datetime | None = None
    if isinstance(value, bool):
        parsed = None
    elif isinstance(value, int | float) and math.isfinite(value):
        try:
            parsed = datetime.fromtimestamp(value, tz=UTC)
        except (OSError, OverflowError, ValueError):
            parsed = None
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                budget.add(
                    warnings,
                    _warning(
                        code="naive_timestamp",
                        message="Naive ISO timestamp was interpreted as UTC.",
                        source_path=source_path,
                        location=location,
                        conversation_id=conversation_id,
                        node_id=node_id,
                    ),
                )
                parsed = parsed.replace(tzinfo=UTC)
            parsed = parsed.astimezone(UTC)
        except ValueError:
            parsed = None
    if parsed is None:
        budget.add(
            warnings,
            _warning(
                code="invalid_timestamp",
                message="Timestamp could not be normalized and was set to null.",
                source_path=source_path,
                location=location,
                conversation_id=conversation_id,
                node_id=node_id,
                context={"source_type": type(value).__name__},
            ),
        )
    return parsed


def _role(
    value: Any,
    *,
    source_path: PurePosixPath,
    location: str,
    conversation_id: str | None,
    node_id: str,
    warnings: list[ArchiveWarning],
    budget: _WarningBudget,
) -> Role:
    """Normalize a role string while retaining unknown-role provenance elsewhere."""

    if isinstance(value, str):
        try:
            return Role(value.casefold())
        except ValueError:
            pass
    budget.add(
        warnings,
        _warning(
            code="unknown_author_role",
            message="Unknown or malformed author role was normalized to 'unknown'.",
            source_path=source_path,
            location=location,
            conversation_id=conversation_id,
            node_id=node_id,
            context={"source_type": type(value).__name__},
        ),
    )
    return Role.UNKNOWN


def _dict_part_block(part: dict[str, Any], source_content_type: str | None) -> ContentBlock:
    """Convert a structured multimodal part into a reference or unknown block."""

    pointer = part.get("asset_pointer")
    if not isinstance(pointer, str):
        pointer = part.get("url") if isinstance(part.get("url"), str) else None
    mime_type = part.get("mime_type") if isinstance(part.get("mime_type"), str) else ""
    part_type = part.get("content_type") if isinstance(part.get("content_type"), str) else ""
    classification_text = f"{mime_type} {part_type}".casefold()
    block_type = ContentBlockType.UNKNOWN
    if pointer is not None:
        block_type = ContentBlockType.FILE_REFERENCE
        if "image" in classification_text:
            block_type = ContentBlockType.IMAGE_REFERENCE
        elif "audio" in classification_text:
            block_type = ContentBlockType.AUDIO_REFERENCE
    return ContentBlock(
        type=block_type,
        reference=pointer,
        source_content_type=source_content_type,
        metadata={"raw_part": _as_json_value(part)},
    )


def _reasoning_text_fragments(value: Any, *, depth: int = 0) -> list[str]:
    """Extract displayable text from documented-as-opaque reasoning content shapes.

    ChatGPT's public export documentation does not define these internal payloads. Extraction is
    therefore deliberately conservative and the complete source object is retained on the block.
    """

    if depth > 8:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [
            fragment
            for item in value
            for fragment in _reasoning_text_fragments(item, depth=depth + 1)
        ]
    if not isinstance(value, dict):
        return []

    fragments: list[str] = []
    for key in ("parts", "thoughts", "summary", "recap", "text", "content"):
        if key in value:
            fragments.extend(_reasoning_text_fragments(value[key], depth=depth + 1))
    return fragments


def _reasoning_block(content: dict[str, Any], content_type: str) -> ContentBlock:
    """Classify a reasoning trace or recap while preserving its exact source object."""

    block_type = (
        ContentBlockType.THINKING_TRACE
        if content_type == "thoughts"
        else ContentBlockType.REASONING_SUMMARY
    )
    fragments = _reasoning_text_fragments(content)
    return ContentBlock(
        type=block_type,
        text="\n\n".join(fragments) if fragments else None,
        source_content_type=content_type,
        metadata={"raw_content": _as_json_value(content)},
    )


def _content_blocks(
    content: Any,
    *,
    source_path: PurePosixPath,
    location: str,
    conversation_id: str | None,
    node_id: str,
    warnings: list[ArchiveWarning],
    budget: _WarningBudget,
) -> list[ContentBlock]:
    """Normalize message content while preserving unsupported JSON structures."""

    if content is None:
        return []
    if not isinstance(content, dict):
        budget.add(
            warnings,
            _warning(
                code="invalid_content",
                message="Message content was not an object and was retained as unknown data.",
                source_path=source_path,
                location=location,
                conversation_id=conversation_id,
                node_id=node_id,
                context={"source_type": type(content).__name__},
            ),
        )
        return [
            ContentBlock(
                type=ContentBlockType.UNKNOWN,
                metadata={"raw_content": _as_json_value(content)},
            )
        ]

    raw_content_type = content.get("content_type")
    content_type = raw_content_type if isinstance(raw_content_type, str) else None
    if content_type in {"text", "multimodal_text"}:
        parts = content.get("parts")
        if not isinstance(parts, list):
            budget.add(
                warnings,
                _warning(
                    code="invalid_content_parts",
                    message="Text content parts were not a list and were retained as unknown data.",
                    source_path=source_path,
                    location=f"{location}/parts",
                    conversation_id=conversation_id,
                    node_id=node_id,
                    context={"source_type": type(parts).__name__},
                ),
            )
            return [
                ContentBlock(
                    type=ContentBlockType.UNKNOWN,
                    source_content_type=content_type,
                    metadata={"raw_content": _as_json_value(content)},
                )
            ]

        blocks: list[ContentBlock] = []
        for part in parts:
            if isinstance(part, str):
                blocks.append(
                    ContentBlock(
                        type=ContentBlockType.MARKDOWN,
                        text=part,
                        source_content_type=content_type,
                    )
                )
            elif isinstance(part, dict):
                blocks.append(_dict_part_block(part, content_type))
            else:
                blocks.append(
                    ContentBlock(
                        type=ContentBlockType.UNKNOWN,
                        source_content_type=content_type,
                        metadata={"raw_part": _as_json_value(part)},
                    )
                )
        return blocks

    if content_type == "code":
        code_text = content.get("text")
        language = content.get("language")
        if isinstance(code_text, str):
            return [
                ContentBlock(
                    type=ContentBlockType.CODE,
                    text=code_text,
                    language=language if isinstance(language, str) else None,
                    source_content_type=content_type,
                )
            ]

    if content_type in {"execution_output", "tool_result"}:
        result = content.get("text", content.get("result"))
        if isinstance(result, str):
            return [
                ContentBlock(
                    type=ContentBlockType.TOOL_RESULT,
                    text=result,
                    source_content_type=content_type,
                )
            ]

    if content_type in {"thoughts", "reasoning_recap"}:
        return [_reasoning_block(content, content_type)]

    budget.add(
        warnings,
        _warning(
            code="unknown_content_type",
            message="Unsupported content shape was retained as an unknown block.",
            source_path=source_path,
            location=location,
            conversation_id=conversation_id,
            node_id=node_id,
            context={"content_type": content_type},
        ),
    )
    return [
        ContentBlock(
            type=ContentBlockType.UNKNOWN,
            source_content_type=content_type,
            metadata={"raw_content": _as_json_value(content)},
        )
    ]


def _message(
    raw_message: dict[str, Any],
    *,
    source_path: PurePosixPath,
    location: str,
    conversation_id: str | None,
    node_id: str,
    warnings: list[ArchiveWarning],
    budget: _WarningBudget,
) -> Message:
    """Normalize one message object without assuming nested field types."""

    author_raw = raw_message.get("author")
    author = author_raw if isinstance(author_raw, dict) else {}
    if author_raw is not None and not isinstance(author_raw, dict):
        budget.add(
            warnings,
            _warning(
                code="invalid_author",
                message="Message author was not an object.",
                source_path=source_path,
                location=f"{location}/author",
                conversation_id=conversation_id,
                node_id=node_id,
            ),
        )

    metadata_raw = raw_message.get("metadata")
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
    if metadata_raw is not None and not isinstance(metadata_raw, dict):
        budget.add(
            warnings,
            _warning(
                code="invalid_message_metadata",
                message="Message metadata was not an object and was retained in source extras.",
                source_path=source_path,
                location=f"{location}/metadata",
                conversation_id=conversation_id,
                node_id=node_id,
            ),
        )

    raw_role = author.get("role")
    raw_end_turn = raw_message.get("end_turn")
    if raw_end_turn is not None and not isinstance(raw_end_turn, bool):
        budget.add(
            warnings,
            _warning(
                code="invalid_end_turn",
                message="Message end_turn was not boolean and was set to null.",
                source_path=source_path,
                location=f"{location}/end_turn",
                conversation_id=conversation_id,
                node_id=node_id,
            ),
        )

    model_value = metadata.get("model_slug", metadata.get("default_model_slug"))
    known_keys = {
        "id",
        "author",
        "create_time",
        "update_time",
        "content",
        "metadata",
        "recipient",
        "status",
        "end_turn",
    }
    source_extras = _extras(raw_message, known_keys)
    if author_raw is not None and not isinstance(author_raw, dict):
        source_extras["raw_author"] = _as_json_value(author_raw)
    if metadata_raw is not None and not isinstance(metadata_raw, dict):
        source_extras["raw_metadata"] = _as_json_value(metadata_raw)
    if not isinstance(raw_role, str) or raw_role.casefold() not in {role.value for role in Role}:
        source_extras["raw_author_role"] = _as_json_value(raw_role)

    return Message(
        message_id=raw_message.get("id") if isinstance(raw_message.get("id"), str) else None,
        role=_role(
            raw_role,
            source_path=source_path,
            location=f"{location}/author/role",
            conversation_id=conversation_id,
            node_id=node_id,
            warnings=warnings,
            budget=budget,
        ),
        author_name=author.get("name") if isinstance(author.get("name"), str) else None,
        created_at=_timestamp(
            raw_message.get("create_time"),
            source_path=source_path,
            location=f"{location}/create_time",
            conversation_id=conversation_id,
            node_id=node_id,
            warnings=warnings,
            budget=budget,
        ),
        updated_at=_timestamp(
            raw_message.get("update_time"),
            source_path=source_path,
            location=f"{location}/update_time",
            conversation_id=conversation_id,
            node_id=node_id,
            warnings=warnings,
            budget=budget,
        ),
        content=_content_blocks(
            raw_message.get("content"),
            source_path=source_path,
            location=f"{location}/content",
            conversation_id=conversation_id,
            node_id=node_id,
            warnings=warnings,
            budget=budget,
        ),
        model=model_value if isinstance(model_value, str) else None,
        recipient=(
            raw_message.get("recipient") if isinstance(raw_message.get("recipient"), str) else None
        ),
        status=raw_message.get("status") if isinstance(raw_message.get("status"), str) else None,
        end_turn=raw_end_turn if isinstance(raw_end_turn, bool) else None,
        metadata={str(key): _as_json_value(value) for key, value in metadata.items()},
        source_extras=source_extras,
    )


def _current_path(
    current_node: Any,
    nodes_by_id: dict[str, MessageNode],
    *,
    source_path: PurePosixPath,
    location: str,
    conversation_id: str | None,
    warnings: list[ArchiveWarning],
    budget: _WarningBudget,
) -> tuple[str | None, list[str]]:
    """Derive only the declared current node's ancestor path without branch guessing."""

    if current_node is None:
        budget.add(
            warnings,
            _warning(
                code="missing_current_node",
                message="Conversation has no declared current node; visible path is empty.",
                source_path=source_path,
                location=location,
                conversation_id=conversation_id,
            ),
        )
        return None, []
    if not isinstance(current_node, str) or current_node not in nodes_by_id:
        budget.add(
            warnings,
            _warning(
                code="unknown_current_node",
                message="Declared current node is absent from the mapping; visible path is empty.",
                source_path=source_path,
                location=location,
                conversation_id=conversation_id,
                context={"source_type": type(current_node).__name__},
            ),
        )
        return current_node if isinstance(current_node, str) else None, []

    reverse_path: list[str] = []
    seen: set[str] = set()
    node_id: str | None = current_node
    while node_id is not None:
        if node_id in seen:
            budget.add(
                warnings,
                _warning(
                    code="graph_cycle",
                    message="Cycle encountered while deriving the current path; path is empty.",
                    source_path=source_path,
                    location=location,
                    conversation_id=conversation_id,
                    node_id=node_id,
                ),
            )
            return current_node, []
        seen.add(node_id)
        node = nodes_by_id.get(node_id)
        if node is None:
            budget.add(
                warnings,
                _warning(
                    code="dangling_parent",
                    message="Current path ended at a missing parent node.",
                    source_path=source_path,
                    location=location,
                    conversation_id=conversation_id,
                    node_id=node_id,
                ),
            )
            break
        reverse_path.append(node_id)
        node_id = node.parent_node_id
    return current_node, list(reversed(reverse_path))


def _canonicalize_and_validate_graph(
    nodes_by_id: dict[str, MessageNode],
    *,
    source_path: PurePosixPath,
    location: str,
    conversation_id: str | None,
    warnings: list[ArchiveWarning],
    budget: _WarningBudget,
) -> None:
    """Build canonical child edges from parents and summarize source-edge disagreement."""

    canonical_children: dict[str, list[str]] = {node_id: [] for node_id in nodes_by_id}
    for node_id in sorted(nodes_by_id):
        node = nodes_by_id[node_id]
        if node.parent_node_id is not None:
            parent = nodes_by_id.get(node.parent_node_id)
            if parent is None:
                budget.add(
                    warnings,
                    _warning(
                        code="dangling_parent",
                        message="Node refers to a parent absent from the mapping.",
                        source_path=source_path,
                        location=f"{location}/{_json_pointer_part(node_id)}/parent",
                        conversation_id=conversation_id,
                        node_id=node_id,
                    ),
                )
            else:
                canonical_children[parent.node_id].append(node_id)

    for node_id, child_ids in canonical_children.items():
        nodes_by_id[node_id].children_node_ids = sorted(child_ids)

    nodes_with_source_differences = 0
    missing_source_edges = 0
    conflicting_source_edges = 0
    dangling_source_edges = 0
    duplicate_source_edges = 0
    for node_id in sorted(nodes_by_id):
        node = nodes_by_id[node_id]
        declared_children = node.source_children_node_ids
        if declared_children is None:
            continue

        declared_set = set(declared_children)
        canonical_set = set(node.children_node_ids)
        duplicate_count = len(declared_children) - len(declared_set)
        missing_count = len(canonical_set - declared_set)
        conflicting_count = 0
        dangling_count = 0
        for child_id in declared_set - canonical_set:
            child = nodes_by_id.get(child_id)
            if child is None:
                dangling_count += 1
            else:
                conflicting_count += 1
        if duplicate_count or missing_count or conflicting_count or dangling_count:
            nodes_with_source_differences += 1
            duplicate_source_edges += duplicate_count
            missing_source_edges += missing_count
            conflicting_source_edges += conflicting_count
            dangling_source_edges += dangling_count

    if nodes_with_source_differences:
        budget.add(
            warnings,
            _warning(
                code="source_child_edges_disagree",
                message=(
                    "Source child declarations differ from canonical edges reconstructed from "
                    "parent pointers."
                ),
                source_path=source_path,
                location=location,
                conversation_id=conversation_id,
                context={
                    "nodes_with_differences": nodes_with_source_differences,
                    "missing_source_edges": missing_source_edges,
                    "conflicting_source_edges": conflicting_source_edges,
                    "dangling_source_edges": dangling_source_edges,
                    "duplicate_source_edges": duplicate_source_edges,
                },
            ),
        )

    roots = [node.node_id for node in nodes_by_id.values() if node.parent_node_id is None]
    reachable: set[str] = set()
    frontier = sorted(roots, reverse=True)
    while frontier:
        node_id = frontier.pop()
        if node_id in reachable or node_id not in nodes_by_id:
            continue
        reachable.add(node_id)
        frontier.extend(sorted(nodes_by_id[node_id].children_node_ids, reverse=True))
    orphan_ids = sorted(set(nodes_by_id) - reachable)
    if orphan_ids:
        budget.add(
            warnings,
            _warning(
                code="orphan_component",
                message="Mapping contains nodes not reachable from a declared root.",
                source_path=source_path,
                location=location,
                conversation_id=conversation_id,
                context={
                    "count": len(orphan_ids),
                    "sample_node_ids": _as_json_value(orphan_ids[:20]),
                },
            ),
        )


def _conversation(
    raw: dict[str, Any],
    *,
    source_path: PurePosixPath,
    index: int,
    limits: IngestLimits,
    budget: _WarningBudget,
) -> Conversation:
    """Normalize one conversation and preserve its complete node graph."""

    base_location = f"/{index}"
    conversation_id = raw.get("id") if isinstance(raw.get("id"), str) else None
    warnings: list[ArchiveWarning] = []
    mapping_raw = raw.get("mapping")
    mapping = mapping_raw if isinstance(mapping_raw, dict) else {}
    if not isinstance(mapping_raw, dict):
        budget.add(
            warnings,
            _warning(
                code="missing_mapping",
                message="Conversation mapping is missing or malformed.",
                source_path=source_path,
                location=f"{base_location}/mapping",
                conversation_id=conversation_id,
            ),
        )
    if len(mapping) > limits.max_nodes_per_conversation:
        raise ArchiveLimitError(
            "conversation exceeds the configured maximum nodes per conversation"
        )

    nodes_by_id: dict[str, MessageNode] = {}
    for raw_node_id in sorted(mapping, key=lambda value: str(value)):
        node_id = str(raw_node_id)
        node_location = f"{base_location}/mapping/{_json_pointer_part(node_id)}"
        raw_node = mapping[raw_node_id]
        if not isinstance(raw_node, dict):
            budget.add(
                warnings,
                _warning(
                    code="invalid_mapping_node",
                    message="Mapping node was not an object and was retained as source data.",
                    source_path=source_path,
                    location=node_location,
                    conversation_id=conversation_id,
                    node_id=node_id,
                ),
            )
            nodes_by_id[node_id] = MessageNode(
                node_id=node_id,
                source_extras={"raw_node": _as_json_value(raw_node)},
            )
            continue

        embedded_id = raw_node.get("id")
        source_extras = _extras(raw_node, {"id", "parent", "children", "message"})
        if isinstance(embedded_id, str) and embedded_id != node_id:
            source_extras["embedded_id"] = embedded_id
            budget.add(
                warnings,
                _warning(
                    code="node_id_mismatch",
                    message="Embedded node id differs from its authoritative mapping key.",
                    source_path=source_path,
                    location=f"{node_location}/id",
                    conversation_id=conversation_id,
                    node_id=node_id,
                ),
            )

        parent_raw = raw_node.get("parent")
        parent_node_id = parent_raw if isinstance(parent_raw, str) else None
        if parent_raw is not None and not isinstance(parent_raw, str):
            source_extras["raw_parent"] = _as_json_value(parent_raw)
            budget.add(
                warnings,
                _warning(
                    code="invalid_parent_id",
                    message="Node parent identifier was not a string and was set to null.",
                    source_path=source_path,
                    location=f"{node_location}/parent",
                    conversation_id=conversation_id,
                    node_id=node_id,
                ),
            )

        children_raw = raw_node.get("children")
        source_children_node_ids: list[str] | None = None
        if isinstance(children_raw, list):
            source_children_node_ids = [child for child in children_raw if isinstance(child, str)]
            if len(source_children_node_ids) != len(children_raw):
                source_extras["raw_children"] = _as_json_value(children_raw)
                budget.add(
                    warnings,
                    _warning(
                        code="invalid_child_id",
                        message="Non-string child identifiers were omitted from normalized edges.",
                        source_path=source_path,
                        location=f"{node_location}/children",
                        conversation_id=conversation_id,
                        node_id=node_id,
                    ),
                )
        elif children_raw is not None:
            source_extras["raw_children"] = _as_json_value(children_raw)
            budget.add(
                warnings,
                _warning(
                    code="invalid_children",
                    message="Node children were not a list and normalized edges were left empty.",
                    source_path=source_path,
                    location=f"{node_location}/children",
                    conversation_id=conversation_id,
                    node_id=node_id,
                ),
            )

        message_raw = raw_node.get("message")
        normalized_message: Message | None = None
        if isinstance(message_raw, dict):
            normalized_message = _message(
                message_raw,
                source_path=source_path,
                location=f"{node_location}/message",
                conversation_id=conversation_id,
                node_id=node_id,
                warnings=warnings,
                budget=budget,
            )
        elif message_raw is not None:
            source_extras["raw_message"] = _as_json_value(message_raw)
            budget.add(
                warnings,
                _warning(
                    code="invalid_message",
                    message="Node message was not an object and was retained as source data.",
                    source_path=source_path,
                    location=f"{node_location}/message",
                    conversation_id=conversation_id,
                    node_id=node_id,
                ),
            )

        nodes_by_id[node_id] = MessageNode(
            node_id=node_id,
            parent_node_id=parent_node_id,
            source_children_node_ids=source_children_node_ids,
            message=normalized_message,
            source_extras=source_extras,
        )

    _canonicalize_and_validate_graph(
        nodes_by_id,
        source_path=source_path,
        location=f"{base_location}/mapping",
        conversation_id=conversation_id,
        warnings=warnings,
        budget=budget,
    )
    current_node_id, current_path_node_ids = _current_path(
        raw.get("current_node"),
        nodes_by_id,
        source_path=source_path,
        location=f"{base_location}/current_node",
        conversation_id=conversation_id,
        warnings=warnings,
        budget=budget,
    )
    current_path_set = set(current_path_node_ids)
    for node in nodes_by_id.values():
        node.is_on_current_path = node.node_id in current_path_set

    known_keys = {
        "id",
        "title",
        "create_time",
        "update_time",
        "current_node",
        "mapping",
    }
    return Conversation(
        conversation_id=conversation_id,
        title=raw.get("title") if isinstance(raw.get("title"), str) else None,
        created_at=_timestamp(
            raw.get("create_time"),
            source_path=source_path,
            location=f"{base_location}/create_time",
            conversation_id=conversation_id,
            node_id=None,
            warnings=warnings,
            budget=budget,
        ),
        updated_at=_timestamp(
            raw.get("update_time"),
            source_path=source_path,
            location=f"{base_location}/update_time",
            conversation_id=conversation_id,
            node_id=None,
            warnings=warnings,
            budget=budget,
        ),
        source_path=source_path,
        current_node_id=current_node_id,
        current_path_node_ids=current_path_node_ids,
        nodes=[nodes_by_id[node_id] for node_id in sorted(nodes_by_id)],
        source_extras=_extras(raw, known_keys),
        warnings=sorted(
            warnings,
            key=lambda item: (item.code, item.location or "", item.node_id or ""),
        ),
    )


def _conversation_records(
    payload: Any,
    *,
    source_path: PurePosixPath,
    result_warnings: list[ArchiveWarning],
    budget: _WarningBudget,
) -> list[Any]:
    """Recognize supported payload roots without truthiness-based fallthrough."""

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "conversations" in payload:
            records = payload["conversations"]
            wrapper_key = "conversations"
        elif "items" in payload:
            records = payload["items"]
            wrapper_key = "items"
        elif "mapping" in payload:
            budget.add(
                result_warnings,
                _warning(
                    code="single_conversation_payload",
                    message="Single conversation object was accepted as a one-item payload.",
                    source_path=source_path,
                    location="/",
                ),
            )
            return [payload]
        else:
            raise UnsupportedSchemaError("JSON root object has no recognized conversation field")
        if not isinstance(records, list):
            raise UnsupportedSchemaError(f"payload field '{wrapper_key}' is not a list")
        budget.add(
            result_warnings,
            _warning(
                code="wrapped_payload_root",
                message="Wrapped conversation-list payload was normalized.",
                source_path=source_path,
                location=f"/{wrapper_key}",
                context={"wrapper_key": wrapper_key},
            ),
        )
        return records
    raise UnsupportedSchemaError("JSON root must be a conversation list or supported object")


def normalize_conversation_payloads(
    payloads: Iterable[tuple[str | PurePosixPath, Any]],
    *,
    limits: IngestLimits | None = None,
) -> NormalizationResult:
    """Normalize one or more decoded JSON members with shared corpus limits.

    Parameters
    ----------
    payloads
        Ordered ``(source_path, decoded JSON value)`` pairs. Each source value normally contains
        a list of conversation objects. The iterable is consumed incrementally so multipart
        exports do not retain every decoded source payload at once.
    limits
        Aggregate semantic record, node, and warning limits shared by all members.

    Returns
    -------
    NormalizationResult
        Normalized conversations plus corpus-level warnings. Each conversation retains the source
        member path and its graph-specific warnings.

    Raises
    ------
    UnsupportedSchemaError
        If any JSON root has no recognized conversation-list shape.
    ArchiveLimitError
        If conversation or graph counts exceed configured limits.
    """

    active_limits = limits or IngestLimits()
    result_warnings: list[ArchiveWarning] = []
    budget = _WarningBudget(active_limits.max_warnings)
    conversations: list[Conversation] = []
    total_records = 0
    total_nodes = 0
    seen_ids: set[str] = set()
    last_member_path: PurePosixPath | None = None
    for source_path, payload in payloads:
        member_path = PurePosixPath(source_path)
        last_member_path = member_path
        records = _conversation_records(
            payload,
            source_path=member_path,
            result_warnings=result_warnings,
            budget=budget,
        )
        total_records += len(records)
        if total_records > active_limits.max_conversations:
            raise ArchiveLimitError("payloads exceed the configured maximum conversation count")

        for index, raw in enumerate(records):
            if not isinstance(raw, dict):
                budget.add(
                    result_warnings,
                    _warning(
                        code="invalid_conversation_record",
                        message="Non-object conversation record was skipped.",
                        source_path=member_path,
                        location=f"/{index}",
                        context={"source_type": type(raw).__name__},
                    ),
                )
                continue
            mapping = raw.get("mapping")
            if isinstance(mapping, dict):
                total_nodes += len(mapping)
                if total_nodes > active_limits.max_total_nodes:
                    raise ArchiveLimitError(
                        "payloads exceed the configured maximum total node count"
                    )
            conversation = _conversation(
                raw,
                source_path=member_path,
                index=index,
                limits=active_limits,
                budget=budget,
            )
            if conversation.conversation_id is not None:
                if conversation.conversation_id in seen_ids:
                    budget.add(
                        result_warnings,
                        _warning(
                            code="duplicate_conversation_id",
                            message=(
                                "Conversation identifier appears more than once; records were "
                                "preserved."
                            ),
                            source_path=member_path,
                            location=f"/{index}/id",
                            conversation_id=conversation.conversation_id,
                        ),
                    )
                seen_ids.add(conversation.conversation_id)
            conversations.append(conversation)
        del payload, records

    if budget.suppressed:
        assert last_member_path is not None
        result_warnings.append(
            _warning(
                code="warnings_suppressed",
                message="Additional normalization warnings were suppressed by the configured cap.",
                source_path=last_member_path,
                location="/",
                context={"suppressed_count": budget.suppressed},
            )
        )
    result_warnings.sort(
        key=lambda item: (str(item.source_path or ""), item.code, item.location or "")
    )
    return NormalizationResult(conversations=conversations, warnings=result_warnings)


def normalize_conversations_payload(
    payload: Any,
    *,
    source_path: str | PurePosixPath,
    limits: IngestLimits | None = None,
) -> NormalizationResult:
    """Normalize one decoded JSON member into a branch-preserving Archive IR fragment."""

    return normalize_conversation_payloads([(source_path, payload)], limits=limits)
