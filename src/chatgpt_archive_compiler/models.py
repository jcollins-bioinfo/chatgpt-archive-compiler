"""Canonical intermediate representation for parsed conversation archives."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Role(StrEnum):
    """Normalized message author roles."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    UNKNOWN = "unknown"


class ContentBlockType(StrEnum):
    """Normalized content block types."""

    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FILE_REFERENCE = "file_reference"
    IMAGE_REFERENCE = "image_reference"
    UNKNOWN = "unknown"


class WarningSeverity(StrEnum):
    """Severity levels for ingestion and normalization warnings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ArchiveWarning(BaseModel):
    """Non-fatal issue encountered while ingesting or normalizing an archive."""

    model_config = ConfigDict(extra="forbid")

    severity: WarningSeverity = WarningSeverity.WARNING
    code: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)


class SourceFile(BaseModel):
    """A file discovered inside the source archive."""

    model_config = ConfigDict(extra="forbid")

    path: PurePosixPath
    size_bytes: int
    sha256: str | None = None
    detected_kind: str | None = None


class SourceManifest(BaseModel):
    """Manifest describing the source archive before semantic parsing."""

    model_config = ConfigDict(extra="forbid")

    archive_name: str | None = None
    files: list[SourceFile] = Field(default_factory=list)
    warnings: list[ArchiveWarning] = Field(default_factory=list)


class ContentBlock(BaseModel):
    """A typed content block within a message."""

    model_config = ConfigDict(extra="allow")

    type: ContentBlockType
    text: str | None = None
    language: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A normalized message."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    role: Role = Role.UNKNOWN
    created_at: datetime | None = None
    content: list[ContentBlock] = Field(default_factory=list)
    model: str | None = None
    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class Conversation(BaseModel):
    """A normalized conversation."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    title: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    messages: list[Message] = Field(default_factory=list)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ArchiveWarning] = Field(default_factory=list)


class Archive(BaseModel):
    """Top-level normalized archive object."""

    model_config = ConfigDict(extra="allow")

    archive_version: str = "1.0"
    source_manifest: SourceManifest
    conversations: list[Conversation] = Field(default_factory=list)
    warnings: list[ArchiveWarning] = Field(default_factory=list)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
