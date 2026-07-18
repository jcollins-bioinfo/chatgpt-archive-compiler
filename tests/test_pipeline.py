"""End-to-end ingestion and serialization tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chatgpt_archive_compiler.exceptions import (
    AmbiguousPayloadError,
    ArchiveSchemaError,
    InvalidPayloadError,
    PayloadNotFoundError,
)
from chatgpt_archive_compiler.ingest import SchemaMode, ingest_export_zip, summarize_archive
from chatgpt_archive_compiler.serialization import read_archive_ir, write_archive_ir


def test_ingest_export_zip_end_to_end(
    branched_payload: list[dict[str, Any]], write_payload_zip  # type: ignore[no-untyped-def]
) -> None:
    """The public API produces a complete, branch-aware Archive IR."""

    archive_path = write_payload_zip(branched_payload)
    archive = ingest_export_zip(archive_path)
    summary = summarize_archive(archive)

    assert archive.archive_version == "1.0"
    assert archive.source_manifest.selected_conversation_path is not None
    assert summary.conversation_count == 1
    assert summary.node_count == 4
    assert summary.message_count == 3
    assert summary.current_path_message_count == 2
    assert summary.warning_count == 0


def test_utf8_bom_is_supported(branched_payload: list[dict[str, Any]], write_zip) -> None:  # type: ignore[no-untyped-def]
    """UTF-8 BOM input is accepted with a stable provenance warning."""

    raw = b"\xef\xbb\xbf" + json.dumps(branched_payload).encode("utf-8")
    archive = ingest_export_zip(write_zip({"conversations.json": raw}))
    assert "utf8_bom" in {warning.code for warning in archive.warnings}


def test_strict_mode_rejects_recoverable_schema_drift(
    branched_payload: list[dict[str, Any]], write_payload_zip  # type: ignore[no-untyped-def]
) -> None:
    """Strict mode escalates normalization warnings after location-aware collection."""

    payload = {"conversations": branched_payload}
    archive_path = write_payload_zip(payload)
    with pytest.raises(ArchiveSchemaError):
        ingest_export_zip(archive_path, schema_mode=SchemaMode.STRICT)


def test_invalid_json_is_typed_failure(write_zip) -> None:  # type: ignore[no-untyped-def]
    """Malformed JSON cannot leak a raw decoder traceback from the public API."""

    archive_path = write_zip({"conversations.json": "{not json"})
    with pytest.raises(InvalidPayloadError):
        ingest_export_zip(archive_path)


def test_missing_payload_is_typed_failure(write_zip) -> None:  # type: ignore[no-untyped-def]
    """An unrelated JSON file is not guessed to contain conversations."""

    archive_path = write_zip({"user.json": "{}"})
    with pytest.raises(PayloadNotFoundError):
        ingest_export_zip(archive_path)


def test_ambiguous_payload_is_typed_failure(write_zip) -> None:  # type: ignore[no-untyped-def]
    """Multiple equally ranked non-root payloads cause a fail-closed selection error."""

    archive_path = write_zip({"one/conversations.json": "[]", "two/conversations.json": "[]"})
    with pytest.raises(AmbiguousPayloadError):
        ingest_export_zip(archive_path)


def test_archive_ir_round_trip_is_canonical(
    branched_payload: list[dict[str, Any]], write_payload_zip, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    """Repeated canonical writes are byte-identical and validate back to the same IR."""

    archive = ingest_export_zip(write_payload_zip(branched_payload))
    first = write_archive_ir(archive, tmp_path / "first.ir.json")
    second = write_archive_ir(archive, tmp_path / "second.ir.json")

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().endswith(b"\n")
    assert read_archive_ir(first) == archive
    assert json.loads(first.read_text(encoding="utf-8"))["archive_version"] == "1.0"
