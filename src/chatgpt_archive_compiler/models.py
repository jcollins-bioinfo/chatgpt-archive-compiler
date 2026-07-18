"""Versioned intermediate representation for ChatGPT conversation archives."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class Role(StrEnum):
    """Normalized message-author roles observed in export data."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    DEVELOPER = "developer"
    TOOL = "tool"
    UNKNOWN = "unknown"


class ContentBlockType(StrEnum):
    """Normalized content-block categories independent of the source schema."""

    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FILE_REFERENCE = "file_reference"
    IMAGE_REFERENCE = "image_reference"
    AUDIO_REFERENCE = "audio_reference"
    UNKNOWN = "unknown"


class WarningSeverity(StrEnum):
    """Severity of a non-secret diagnostic emitted during ingestion."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class SchemaMode(StrEnum):
    """Policy for recoverable source-schema variation.

    ``TOLERANT`` preserves interpretable records and emits warnings. ``STRICT`` rejects a
    payload as soon as normalization produces any warning.
    """

    TOLERANT = "tolerant"
    STRICT = "strict"


class SourceFileKind(StrEnum):
    """Kinds assigned to files from their archive-relative paths."""

    CONVERSATIONS_JSON = "conversations_json"
    CONVERSATIONS_JSON_CANDIDATE = "conversations_json_candidate"
    JSON = "json"
    OTHER = "other"


class IngestLimits(BaseModel):
    """Hard resource and compression limits for untrusted ZIP input.

    Attributes
    ----------
    max_files
        Maximum number of central-directory entries, including directories.
    max_archive_bytes
        Maximum size of the compressed ZIP file itself.
    max_total_uncompressed_bytes
        Maximum sum of declared uncompressed sizes across all members.
    max_member_uncompressed_bytes
        Maximum declared uncompressed size for any individual member.
    max_json_bytes
        Maximum number of bytes read from one conversation JSON member.
    max_compression_ratio
        Maximum allowed ratio of uncompressed to compressed bytes for a non-empty member.
    min_ratio_check_bytes
        Minimum uncompressed size at which compression-ratio rejection is applied.
    max_conversations
        Maximum number of conversation records accepted from the payload.
    max_nodes_per_conversation
        Maximum number of graph nodes accepted in one conversation.
    max_total_nodes
        Maximum graph nodes accepted across the entire payload.
    max_warnings
        Maximum detailed normalization warnings retained before a suppression summary.
    read_chunk_bytes
        Maximum chunk requested from a decompression stream at one time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    max_archive_bytes: int = Field(default=8 * 1024**3, gt=0)
    max_files: int = Field(default=100_000, gt=0)
    max_total_uncompressed_bytes: int = Field(default=20 * 1024**3, gt=0)
    max_member_uncompressed_bytes: int = Field(default=2 * 1024**3, gt=0)
    max_json_bytes: int = Field(default=512 * 1024**2, gt=0)
    max_compression_ratio: float = Field(default=250.0, gt=0)
    min_ratio_check_bytes: int = Field(default=1024**2, ge=0)
    max_conversations: int = Field(default=200_000, gt=0)
    max_nodes_per_conversation: int = Field(default=100_000, gt=0)
    max_total_nodes: int = Field(default=5_000_000, gt=0)
    max_warnings: int = Field(default=10_000, gt=0)
    read_chunk_bytes: int = Field(default=1024**2, gt=0, le=16 * 1024**2)


class ArchiveWarning(BaseModel):
    """A stable, non-secret diagnostic about source or normalization quality."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    severity: WarningSeverity = WarningSeverity.WARNING
    code: str
    message: str
    source_path: PurePosixPath | None = None
    conversation_id: str | None = None
    node_id: str | None = None
    location: str | None = None
    context: dict[str, JsonValue] = Field(default_factory=dict)


class SourceFile(BaseModel):
    """Metadata for one safe archive member discovered in the source ZIP."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    path: PurePosixPath
    size_bytes: int = Field(ge=0)
    compressed_size_bytes: int = Field(ge=0)
    crc32: int = Field(ge=0)
    compression_method: int = Field(ge=0)
    sha256: str | None = None
    detected_kind: SourceFileKind = SourceFileKind.OTHER
    is_readable: bool = True


class SourceManifest(BaseModel):
    """Deterministic provenance and safety metadata for a source ZIP."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    archive_name: str
    archive_size_bytes: int = Field(ge=0)
    archive_sha256: str | None = None
    files: list[SourceFile] = Field(default_factory=list)
    total_uncompressed_bytes: int = Field(default=0, ge=0)
    total_compressed_bytes: int = Field(default=0, ge=0)
    selected_conversation_path: PurePosixPath | None = None
    selected_conversation_sha256: str | None = None
    warnings: list[ArchiveWarning] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Return whether inspection found any fatal-severity diagnostics."""

        return any(warning.severity is WarningSeverity.ERROR for warning in self.warnings)


class ContentBlock(BaseModel):
    """A typed content unit within a normalized message."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    type: ContentBlockType
    text: str | None = None
    language: str | None = None
    reference: str | None = None
    source_content_type: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class Message(BaseModel):
    """Semantic message data independent of its graph position."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    message_id: str | None = None
    role: Role = Role.UNKNOWN
    author_name: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    content: list[ContentBlock] = Field(default_factory=list)
    model: str | None = None
    recipient: str | None = None
    status: str | None = None
    end_turn: bool | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    source_extras: dict[str, JsonValue] = Field(default_factory=dict)


class MessageNode(BaseModel):
    """One node in the exported conversation graph.

    ``message`` is ``None`` for structural roots. Node identifiers and message identifiers are
    intentionally distinct because ChatGPT exports use both namespaces.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    node_id: str
    parent_node_id: str | None = None
    children_node_ids: list[str] = Field(default_factory=list)
    message: Message | None = None
    is_on_current_path: bool = False
    source_extras: dict[str, JsonValue] = Field(default_factory=dict)


class Conversation(BaseModel):
    """A normalized conversation with its complete branch graph preserved."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    conversation_id: str | None = None
    title: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    source_path: PurePosixPath
    current_node_id: str | None = None
    current_path_node_ids: list[str] = Field(default_factory=list)
    nodes: list[MessageNode] = Field(default_factory=list)
    source_extras: dict[str, JsonValue] = Field(default_factory=dict)
    warnings: list[ArchiveWarning] = Field(default_factory=list)

    @property
    def current_path_messages(self) -> list[Message]:
        """Return messages on the selected visible path in root-to-leaf order."""

        nodes_by_id = {node.node_id: node for node in self.nodes}
        messages: list[Message] = []
        for node_id in self.current_path_node_ids:
            node = nodes_by_id.get(node_id)
            if node is not None and node.message is not None:
                messages.append(node.message)
        return messages


class Archive(BaseModel):
    """Top-level Archive IR produced by deterministic export ingestion."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    archive_version: str = "1.0"
    source_manifest: SourceManifest
    conversations: list[Conversation] = Field(default_factory=list)
    warnings: list[ArchiveWarning] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def all_warnings(self) -> list[ArchiveWarning]:
        """Return manifest, archive, and conversation warnings as one ordered list."""

        conversation_warnings = [
            warning for conversation in self.conversations for warning in conversation.warnings
        ]
        return [*self.source_manifest.warnings, *self.warnings, *conversation_warnings]


class NormalizationResult(BaseModel):
    """Conversations and corpus-level warnings produced from one JSON payload."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    conversations: list[Conversation] = Field(default_factory=list)
    warnings: list[ArchiveWarning] = Field(default_factory=list)


class ArchiveSummary(BaseModel):
    """Non-content counts suitable for logs and notebook validation."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    conversation_count: int = Field(ge=0)
    node_count: int = Field(ge=0)
    message_count: int = Field(ge=0)
    current_path_message_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    earliest_conversation_at: datetime | None = None
    latest_conversation_at: datetime | None = None
