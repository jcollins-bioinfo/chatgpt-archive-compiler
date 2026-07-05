"""Initial normalization for conversation JSON payloads.

This module intentionally implements only a conservative first pass. It should accept
schema variation and return warnings rather than assuming a single permanent export
shape.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from chatgpt_archive_compiler.models import (
    ContentBlock,
    ContentBlockType,
    Conversation,
    Message,
    Role,
)


def _from_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        try:
            return datetime.fromtimestamp(value, tz=UTC)
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _normalize_role(value: Any) -> Role:
    if not isinstance(value, str):
        return Role.UNKNOWN
    try:
        return Role(value)
    except ValueError:
        return Role.UNKNOWN


def _content_blocks_from_message(message: dict[str, Any]) -> list[ContentBlock]:
    content = message.get("content") or {}
    content_type = content.get("content_type")

    if content_type in {"text", "multimodal_text"}:
        parts = content.get("parts") or []
        blocks: list[ContentBlock] = []
        for part in parts:
            if isinstance(part, str):
                blocks.append(ContentBlock(type=ContentBlockType.MARKDOWN, text=part))
            elif isinstance(part, dict):
                blocks.append(
                    ContentBlock(
                        type=ContentBlockType.UNKNOWN,
                        text=str(part),
                        metadata={"raw_part": part},
                    )
                )
        return blocks

    if isinstance(content, dict):
        return [
            ContentBlock(
                type=ContentBlockType.UNKNOWN,
                text=str(content),
                metadata={"content_type": content_type},
            )
        ]

    return []


def _message_from_node(node_id: str, node: dict[str, Any]) -> Message | None:
    raw_message = node.get("message")
    if not isinstance(raw_message, dict):
        return None

    author = raw_message.get("author") or {}
    role = _normalize_role(author.get("role"))

    metadata = raw_message.get("metadata") or {}
    model = metadata.get("model_slug") or metadata.get("default_model_slug")

    return Message(
        id=raw_message.get("id") or node_id,
        role=role,
        created_at=_from_timestamp(raw_message.get("create_time")),
        content=_content_blocks_from_message(raw_message),
        model=model if isinstance(model, str) else None,
        parent_id=node.get("parent"),
        children_ids=[str(child) for child in node.get("children") or []],
        raw_metadata={"node": node, "message_metadata": metadata},
    )


def _linearize_mapping(mapping: dict[str, Any], current_node: str | None) -> list[Message]:
    """Return a best-effort primary message path from a mapping graph.

    The first implementation follows parent pointers from the current node to root,
    then reverses the chain. Alternate branches are preserved in raw metadata but are
    not yet emitted as separate transcript branches.
    """

    if not current_node or current_node not in mapping:
        node_ids: Iterable[str] = mapping.keys()
        messages = []
        for node_id in node_ids:
            node = mapping.get(node_id)
            if isinstance(node, dict):
                parsed = _message_from_node(node_id, node)
                if parsed is not None:
                    messages.append(parsed)
        return messages

    chain: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    node_id: str | None = current_node

    while node_id and node_id in mapping and node_id not in seen:
        seen.add(node_id)
        node = mapping[node_id]
        if not isinstance(node, dict):
            break
        chain.append((node_id, node))
        parent = node.get("parent")
        node_id = parent if isinstance(parent, str) else None

    messages = []
    for node_id, node in reversed(chain):
        parsed = _message_from_node(node_id, node)
        if parsed is not None:
            messages.append(parsed)
    return messages


def normalize_conversations_payload(payload: Any) -> list[Conversation]:
    """Normalize a decoded conversations payload into Conversation objects."""

    if isinstance(payload, dict):
        raw_conversations = payload.get("conversations") or payload.get("items") or []
    else:
        raw_conversations = payload

    if not isinstance(raw_conversations, list):
        return []

    conversations: list[Conversation] = []
    for raw in raw_conversations:
        if not isinstance(raw, dict):
            continue

        mapping = raw.get("mapping") or {}
        messages: list[Message]
        if isinstance(mapping, dict):
            messages = _linearize_mapping(mapping, raw.get("current_node"))
        else:
            messages = []

        conversations.append(
            Conversation(
                id=raw.get("id") if isinstance(raw.get("id"), str) else None,
                title=raw.get("title") if isinstance(raw.get("title"), str) else None,
                created_at=_from_timestamp(raw.get("create_time")),
                updated_at=_from_timestamp(raw.get("update_time")),
                messages=messages,
                raw_metadata={"source_record": raw},
            )
        )

    return conversations
