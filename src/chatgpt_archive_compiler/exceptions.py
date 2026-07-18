"""Typed failures raised by archive ingestion and serialization."""

from __future__ import annotations

from pathlib import Path

from chatgpt_archive_compiler.models import SourceManifest


class ArchiveCompilerError(Exception):
    """Base class for expected ChatGPT Archive Compiler failures."""


class ArchiveIngestError(ArchiveCompilerError):
    """Base class for failures encountered while reading an export archive."""


class InvalidArchiveError(ArchiveIngestError):
    """Raised when an input is missing, unreadable, or not a valid ZIP archive."""


class ArchiveSafetyError(ArchiveIngestError):
    """Raised when a ZIP violates a security or resource boundary.

    Parameters
    ----------
    message
        Human-readable failure description that does not include private message content.
    manifest
        Partial source manifest containing the stable warning codes that caused rejection.
    """

    def __init__(self, message: str, manifest: SourceManifest) -> None:
        super().__init__(message)
        self.manifest = manifest


class ArchiveLimitError(ArchiveIngestError):
    """Raised when decoded payload structure exceeds a configured semantic limit."""


class PayloadNotFoundError(ArchiveIngestError):
    """Raised when no supported conversation JSON member exists in an export."""


class AmbiguousPayloadError(ArchiveIngestError):
    """Raised when multiple equally ranked conversation payloads prevent safe selection."""


class InvalidPayloadError(ArchiveIngestError):
    """Raised when the selected conversation payload is not valid strict UTF-8 JSON."""


class ArchiveSchemaError(ArchiveIngestError):
    """Raised when source JSON cannot be interpreted under strict schema handling."""


class UnsupportedSchemaError(ArchiveSchemaError):
    """Raised when decoded JSON has no recognized conversation-payload root shape."""


class ArchiveSerializationError(ArchiveCompilerError):
    """Raised when Archive IR cannot be read from or written to a local path.

    Parameters
    ----------
    path
        Destination or source path involved in the failed serialization operation.
    message
        Human-readable failure description.
    """

    def __init__(self, path: str | Path, message: str) -> None:
        self.path = Path(path)
        super().__init__(f"{message}: {self.path}")
