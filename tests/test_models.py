"""Unit tests for core Archive IR model invariants."""

from chatgpt_archive_compiler.models import (
    Archive,
    ContentBlock,
    ContentBlockType,
    Message,
    MessageNode,
    SourceManifest,
)


def test_minimal_archive_model() -> None:
    """A manifest-backed empty archive has the canonical IR version."""

    manifest = SourceManifest(archive_name="synthetic.zip", archive_size_bytes=0)
    archive = Archive(source_manifest=manifest)
    assert archive.archive_version == "1.0"
    assert archive.conversations == []


def test_content_block_model() -> None:
    """A Markdown block preserves text under the closed model schema."""

    block = ContentBlock(type=ContentBlockType.MARKDOWN, text="hello")
    assert block.text == "hello"


def test_node_and_message_identifiers_are_distinct() -> None:
    """Graph node IDs are not conflated with semantic message IDs."""

    node = MessageNode(node_id="node-001", message=Message(message_id="message-001"))
    assert node.node_id == "node-001"
    assert node.message is not None
    assert node.message.message_id == "message-001"
