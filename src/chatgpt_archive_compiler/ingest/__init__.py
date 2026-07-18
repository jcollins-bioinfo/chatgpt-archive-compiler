"""Public, local-only archive ingestion interfaces."""

from chatgpt_archive_compiler.exceptions import (
    AmbiguousPayloadError,
    ArchiveIngestError,
    ArchiveLimitError,
    ArchiveSafetyError,
    ArchiveSchemaError,
    InvalidArchiveError,
    InvalidPayloadError,
    PayloadNotFoundError,
    UnsupportedSchemaError,
)
from chatgpt_archive_compiler.ingest.pipeline import ingest_export_zip, summarize_archive
from chatgpt_archive_compiler.ingest.zip_reader import inspect_zip, read_zip_member_bytes
from chatgpt_archive_compiler.models import IngestLimits, SchemaMode

__all__ = [
    "AmbiguousPayloadError",
    "ArchiveIngestError",
    "ArchiveLimitError",
    "ArchiveSafetyError",
    "ArchiveSchemaError",
    "IngestLimits",
    "InvalidArchiveError",
    "InvalidPayloadError",
    "PayloadNotFoundError",
    "SchemaMode",
    "UnsupportedSchemaError",
    "ingest_export_zip",
    "inspect_zip",
    "read_zip_member_bytes",
    "summarize_archive",
]
