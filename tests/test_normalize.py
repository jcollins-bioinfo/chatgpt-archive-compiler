"""Schema-drift and graph-preservation tests for conversation normalization."""

from __future__ import annotations

import copy
from pathlib import PurePosixPath
from typing import Any

import pytest

from chatgpt_archive_compiler.models import ContentBlockType, IngestLimits, Role
from chatgpt_archive_compiler.normalize import normalize_conversations_payload

SOURCE_PATH = PurePosixPath("conversations.json")


def test_all_branches_are_preserved_but_current_path_is_exact(
    branched_payload: list[dict[str, Any]],
) -> None:
    """Sibling regeneration remains in nodes but never enters the visible path."""

    conversation = normalize_conversations_payload(
        branched_payload, source_path=SOURCE_PATH
    ).conversations[0]
    nodes = {node.node_id: node for node in conversation.nodes}

    assert set(nodes) == {"root", "user-001", "assistant-current", "assistant-alternate"}
    assert conversation.current_path_node_ids == ["root", "user-001", "assistant-current"]
    assert nodes["assistant-current"].is_on_current_path
    assert not nodes["assistant-alternate"].is_on_current_path
    assert nodes["assistant-alternate"].message is not None
    assert nodes["assistant-alternate"].message.message_id == "message-assistant-alternate"


def test_mapping_order_does_not_change_normalized_graph(
    branched_payload: list[dict[str, Any]],
) -> None:
    """Node order and active-path semantics are independent of JSON object insertion order."""

    reordered = copy.deepcopy(branched_payload)
    mapping = reordered[0]["mapping"]
    reordered[0]["mapping"] = dict(reversed(list(mapping.items())))

    first = normalize_conversations_payload(branched_payload, source_path=SOURCE_PATH)
    second = normalize_conversations_payload(reordered, source_path=SOURCE_PATH)
    assert first == second


def test_parent_only_graph_reconstructs_canonical_children(
    branched_payload: list[dict[str, Any]],
) -> None:
    """Parent-only exports produce a complete graph without reciprocal-edge warnings."""

    payload = copy.deepcopy(branched_payload)
    for node in payload[0]["mapping"].values():
        node.pop("children")

    conversation = normalize_conversations_payload(payload, source_path=SOURCE_PATH).conversations[
        0
    ]
    nodes = {node.node_id: node for node in conversation.nodes}

    assert conversation.warnings == []
    assert nodes["root"].children_node_ids == ["user-001"]
    assert nodes["user-001"].children_node_ids == [
        "assistant-alternate",
        "assistant-current",
    ]
    assert nodes["assistant-current"].children_node_ids == []
    assert all(node.source_children_node_ids is None for node in nodes.values())
    assert conversation.current_path_node_ids == ["root", "user-001", "assistant-current"]


def test_source_child_disagreement_is_aggregated_and_preserved(
    branched_payload: list[dict[str, Any]],
) -> None:
    """Incomplete source child lists produce one summary while canonical edges remain complete."""

    payload = copy.deepcopy(branched_payload)
    for node in payload[0]["mapping"].values():
        node["children"] = []

    conversation = normalize_conversations_payload(payload, source_path=SOURCE_PATH).conversations[
        0
    ]
    nodes = {node.node_id: node for node in conversation.nodes}
    warnings = [
        warning
        for warning in conversation.warnings
        if warning.code == "source_child_edges_disagree"
    ]

    assert len(warnings) == 1
    assert warnings[0].context == {
        "nodes_with_differences": 2,
        "missing_source_edges": 3,
        "conflicting_source_edges": 0,
        "dangling_source_edges": 0,
        "duplicate_source_edges": 0,
    }
    assert nodes["root"].children_node_ids == ["user-001"]
    assert nodes["user-001"].children_node_ids == [
        "assistant-alternate",
        "assistant-current",
    ]
    assert all(node.source_children_node_ids == [] for node in nodes.values())


def test_missing_current_node_never_concatenates_siblings(
    branched_payload: list[dict[str, Any]],
) -> None:
    """Absent current-node provenance produces an empty path rather than a guessed transcript."""

    payload = copy.deepcopy(branched_payload)
    payload[0].pop("current_node")
    conversation = normalize_conversations_payload(payload, source_path=SOURCE_PATH).conversations[
        0
    ]

    assert conversation.current_path_node_ids == []
    assert "missing_current_node" in {warning.code for warning in conversation.warnings}
    assert len(conversation.nodes) == 4


def test_cycle_terminates_and_empties_visible_path(
    branched_payload: list[dict[str, Any]],
) -> None:
    """A parent cycle cannot cause nontermination or fabricate a partial visible transcript."""

    payload = copy.deepcopy(branched_payload)
    payload[0]["mapping"]["root"]["parent"] = "assistant-current"
    conversation = normalize_conversations_payload(payload, source_path=SOURCE_PATH).conversations[
        0
    ]

    assert conversation.current_path_node_ids == []
    assert "graph_cycle" in {warning.code for warning in conversation.warnings}


def test_string_parts_are_not_split_into_characters(
    branched_payload: list[dict[str, Any]],
) -> None:
    """Malformed scalar parts become one unknown block with a warning."""

    payload = copy.deepcopy(branched_payload)
    message = payload[0]["mapping"]["user-001"]["message"]
    message["content"]["parts"] = "abc"
    conversation = normalize_conversations_payload(payload, source_path=SOURCE_PATH).conversations[
        0
    ]
    node = next(node for node in conversation.nodes if node.node_id == "user-001")

    assert node.message is not None
    assert len(node.message.content) == 1
    assert node.message.content[0].type is ContentBlockType.UNKNOWN
    assert "invalid_content_parts" in {warning.code for warning in conversation.warnings}


@pytest.mark.parametrize("bad_value", ["scalar", 7, True, ["not", "an", "object"]])
def test_nested_schema_scalars_do_not_crash(
    branched_payload: list[dict[str, Any]], bad_value: Any
) -> None:
    """Malformed author, metadata, and content values remain total over JSON types."""

    payload = copy.deepcopy(branched_payload)
    message = payload[0]["mapping"]["user-001"]["message"]
    message["author"] = bad_value
    message["metadata"] = bad_value
    message["content"] = bad_value
    result = normalize_conversations_payload(payload, source_path=SOURCE_PATH)

    node = next(node for node in result.conversations[0].nodes if node.node_id == "user-001")
    assert node.message is not None
    assert node.message.role is Role.UNKNOWN
    assert node.message.content[0].type is ContentBlockType.UNKNOWN


def test_empty_conversations_wrapper_does_not_fall_through_to_items() -> None:
    """An explicitly empty canonical wrapper takes precedence over a populated fallback."""

    result = normalize_conversations_payload(
        {"conversations": [], "items": [{"id": "should-not-appear", "mapping": {}}]},
        source_path=SOURCE_PATH,
    )
    assert result.conversations == []


def test_node_limit_is_enforced(branched_payload: list[dict[str, Any]]) -> None:
    """Decoded graph cardinality is bounded independently of ZIP byte size."""

    from chatgpt_archive_compiler.exceptions import ArchiveLimitError

    with pytest.raises(ArchiveLimitError):
        normalize_conversations_payload(
            branched_payload,
            source_path=SOURCE_PATH,
            limits=IngestLimits(max_nodes_per_conversation=3),
        )
