"""End-to-end ingestion and serialization tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from chatgpt_archive_compiler.exceptions import (
    AmbiguousPayloadError,
    ArchiveLimitError,
    ArchiveSchemaError,
    InvalidPayloadError,
    PayloadNotFoundError,
)
from chatgpt_archive_compiler.ingest import (
    IngestLimits,
    SchemaMode,
    ingest_export_zip,
    summarize_archive,
)
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
    assert archive.source_manifest.selected_conversation_paths == [Path("conversations.json")]
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


def test_numbered_multipart_payloads_are_ingested_in_numeric_order(
    branched_payload: list[dict[str, Any]], write_zip  # type: ignore[no-untyped-def]
) -> None:
    """Contiguous root-level parts are normalized incrementally in numeric filename order."""

    members: dict[str, str] = {}
    for index in range(10, -1, -1):
        payload = deepcopy(branched_payload)
        payload[0]["id"] = f"conversation-{index}"
        members[f"conversations-{index}.json"] = json.dumps(payload)

    archive = ingest_export_zip(write_zip(members), schema_mode=SchemaMode.STRICT)

    expected_paths = [Path(f"conversations-{index}.json") for index in range(11)]
    assert archive.source_manifest.selected_conversation_paths == expected_paths
    assert archive.source_manifest.selected_conversation_path is None
    assert archive.source_manifest.selected_conversation_sha256 is None
    assert [conversation.conversation_id for conversation in archive.conversations] == [
        f"conversation-{index}" for index in range(11)
    ]
    assert [conversation.source_path for conversation in archive.conversations] == expected_paths
    assert archive.metadata["conversation_payload_count"] == 11
    selected_files = [
        source_file
        for source_file in archive.source_manifest.files
        if source_file.path in expected_paths
    ]
    assert all(source_file.sha256 is not None for source_file in selected_files)


def test_numbered_multipart_sequence_must_be_contiguous(write_zip) -> None:  # type: ignore[no-untyped-def]
    """A missing numbered part is fatal rather than silently producing an incomplete corpus."""

    archive_path = write_zip({"conversations-000.json": "[]", "conversations-002.json": "[]"})
    with pytest.raises(AmbiguousPayloadError):
        ingest_export_zip(archive_path)


def test_multipart_total_json_limit_is_aggregate(write_zip) -> None:  # type: ignore[no-untyped-def]
    """Individually small parts cannot bypass the combined decoded-byte boundary."""

    archive_path = write_zip({"conversations-000.json": "[]", "conversations-001.json": "[]"})
    limits = IngestLimits(max_json_bytes=3, max_total_json_bytes=3)
    with pytest.raises(ArchiveLimitError):
        ingest_export_zip(archive_path, limits=limits)


def test_multipart_conversation_limit_is_aggregate(
    branched_payload: list[dict[str, Any]], write_zip  # type: ignore[no-untyped-def]
) -> None:
    """Conversation cardinality is enforced across parts instead of once per member."""

    archive_path = write_zip(
        {
            "conversations-000.json": json.dumps(branched_payload),
            "conversations-001.json": json.dumps(branched_payload),
        }
    )
    with pytest.raises(ArchiveLimitError):
        ingest_export_zip(archive_path, limits=IngestLimits(max_conversations=1))


def test_multipart_node_limit_is_aggregate(
    branched_payload: list[dict[str, Any]], write_zip  # type: ignore[no-untyped-def]
) -> None:
    """Graph-node cardinality is enforced across all numbered members."""

    archive_path = write_zip(
        {
            "conversations-000.json": json.dumps(branched_payload),
            "conversations-001.json": json.dumps(branched_payload),
        }
    )
    with pytest.raises(ArchiveLimitError):
        ingest_export_zip(archive_path, limits=IngestLimits(max_total_nodes=7))


def test_multipart_warning_budget_is_shared(write_zip) -> None:  # type: ignore[no-untyped-def]
    """Malformed records across parts share one bounded diagnostic budget."""

    archive = ingest_export_zip(
        write_zip(
            {
                "conversations-000.json": "[42]",
                "conversations-001.json": "[43]",
            }
        ),
        limits=IngestLimits(max_warnings=1),
    )

    assert [warning.code for warning in archive.warnings] == [
        "invalid_conversation_record",
        "warnings_suppressed",
    ]
    assert archive.warnings[-1].context == {"suppressed_count": 1}


def test_duplicate_conversation_ids_across_parts_are_reported(
    branched_payload: list[dict[str, Any]], write_zip  # type: ignore[no-untyped-def]
) -> None:
    """The shared normalizer detects identifiers repeated across different members."""

    encoded = json.dumps(branched_payload)
    archive = ingest_export_zip(
        write_zip(
            {
                "conversations-000.json": encoded,
                "conversations-001.json": encoded,
            }
        )
    )

    duplicate_warnings = [
        warning for warning in archive.warnings if warning.code == "duplicate_conversation_id"
    ]
    assert len(duplicate_warnings) == 1
    assert duplicate_warnings[0].source_path == Path("conversations-001.json")


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
