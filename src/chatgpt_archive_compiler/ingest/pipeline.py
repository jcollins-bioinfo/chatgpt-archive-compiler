"""End-to-end ZIP-to-Archive-IR ingestion orchestration."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from chatgpt_archive_compiler.exceptions import (
    AmbiguousPayloadError,
    ArchiveLimitError,
    ArchiveSafetyError,
    ArchiveSchemaError,
    InvalidPayloadError,
    PayloadNotFoundError,
)
from chatgpt_archive_compiler.ingest.zip_reader import inspect_zip, read_zip_member_bytes
from chatgpt_archive_compiler.models import (
    Archive,
    ArchiveSummary,
    ArchiveWarning,
    IngestLimits,
    SchemaMode,
    SourceFile,
    SourceFileKind,
    SourceManifest,
    WarningSeverity,
)
from chatgpt_archive_compiler.normalize import normalize_conversation_payloads
from chatgpt_archive_compiler.version import __version__

_NUMBERED_CONVERSATION_FILE = re.compile(r"^conversations-(\d+)\.json$", re.IGNORECASE)


def _select_conversation_files(
    manifest: SourceManifest,
) -> tuple[list[SourceFile], list[ArchiveWarning]]:
    """Select one canonical member or a complete numbered multipart sequence."""

    readable = [source_file for source_file in manifest.files if source_file.is_readable]
    exact = [
        source_file
        for source_file in readable
        if source_file.detected_kind is SourceFileKind.CONVERSATIONS_JSON
    ]
    if len(exact) == 1:
        return exact, []
    if len(exact) > 1:
        raise AmbiguousPayloadError("multiple canonical conversations.json members found")

    basename_matches = [
        source_file
        for source_file in readable
        if source_file.path.name.casefold() == "conversations.json"
    ]
    if len(basename_matches) == 1:
        selected = basename_matches[0]
        return [selected], [
            ArchiveWarning(
                severity=WarningSeverity.WARNING,
                code="noncanonical_conversations_path",
                message="Conversation payload was selected from a non-root archive path.",
                source_path=selected.path,
                location="/",
            )
        ]
    if len(basename_matches) > 1:
        raise AmbiguousPayloadError(
            "multiple non-root conversations.json members prevent safe payload selection"
        )

    numbered_candidates = [
        source_file
        for source_file in readable
        if source_file.detected_kind is SourceFileKind.CONVERSATIONS_JSON_CANDIDATE
    ]
    if len(numbered_candidates) == 1:
        selected = numbered_candidates[0]
        return [selected], [
            ArchiveWarning(
                severity=WarningSeverity.WARNING,
                code="noncanonical_conversations_filename",
                message="A single conversation-like JSON filename was selected as schema drift.",
                source_path=selected.path,
                location="/",
            )
        ]
    if len(numbered_candidates) > 1:
        indexed_candidates: list[tuple[int, SourceFile]] = []
        for source_file in numbered_candidates:
            match = _NUMBERED_CONVERSATION_FILE.fullmatch(source_file.path.name)
            if len(source_file.path.parts) != 1 or match is None:
                raise AmbiguousPayloadError(
                    "multiple conversation-like JSON members do not form one root-level "
                    "numbered sequence"
                )
            indexed_candidates.append((int(match.group(1)), source_file))

        indices = [index for index, _ in indexed_candidates]
        if len(indices) != len(set(indices)):
            raise AmbiguousPayloadError(
                "numbered conversation JSON members contain duplicate numeric indices"
            )
        ordered = sorted(indexed_candidates, key=lambda item: item[0])
        ordered_indices = [index for index, _ in ordered]
        if ordered_indices != list(range(len(ordered))):
            raise AmbiguousPayloadError(
                "numbered conversation JSON members must be contiguous and start at zero"
            )
        return [source_file for _, source_file in ordered], []
    raise PayloadNotFoundError("archive contains no supported conversation JSON payload")


def _decode_json(raw_bytes: bytes) -> tuple[object, bool]:
    """Decode strict UTF-8 JSON while rejecting non-standard non-finite constants."""

    had_bom = raw_bytes.startswith(b"\xef\xbb\xbf")
    try:
        text = raw_bytes.decode("utf-8-sig")

        def reject_constant(value: str) -> None:
            """Reject NaN and Infinity extensions accepted by Python's JSON decoder."""

            raise ValueError(f"non-finite JSON constant is unsupported: {value}")

        return json.loads(text, parse_constant=reject_constant), had_bom
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise InvalidPayloadError(
            "conversation payload is not supported strict UTF-8 JSON"
        ) from exc


def ingest_export_zip(
    archive_path: str | Path,
    *,
    limits: IngestLimits | None = None,
    schema_mode: SchemaMode = SchemaMode.TOLERANT,
    compute_member_hashes: bool = False,
    compute_archive_hash: bool = True,
) -> Archive:
    """Ingest a ChatGPT export ZIP into the versioned, branch-preserving Archive IR.

    The function never extracts the archive, makes no network calls, and does not log private
    conversation fields. ZIP security/resource failures are fatal; recoverable source-schema
    variation becomes structured warnings in tolerant mode.

    Parameters
    ----------
    archive_path
        Local path to a ChatGPT data-export ZIP.
    limits
        Hard ZIP and semantic resource limits. Defaults to :class:`IngestLimits`.
    schema_mode
        ``tolerant`` retains interpretable schema drift; ``strict`` rejects any normalization
        warning.
    compute_member_hashes
        Whether to decompress and SHA-256 hash every safe archive member after preflight.
    compute_archive_hash
        Whether to SHA-256 hash the compressed source ZIP for provenance.

    Returns
    -------
    Archive
        Canonical source manifest, complete conversation graphs, visible paths, and warnings.

    Raises
    ------
    InvalidArchiveError
        If the local input is missing, unreadable, or not a valid ZIP.
    ArchiveSafetyError
        If ZIP paths, compression properties, encryption, or sizes violate configured limits.
    PayloadNotFoundError
        If no supported conversation JSON member exists.
    AmbiguousPayloadError
        If multiple equally ranked payload members exist.
    InvalidPayloadError
        If the selected member is not supported strict UTF-8 JSON.
    ArchiveSchemaError
        If strict mode encounters recoverable schema variation.
    ArchiveLimitError
        If decoded conversation or graph counts exceed configured semantic limits.
    """

    active_limits = limits or IngestLimits()
    manifest = inspect_zip(
        archive_path,
        limits=active_limits,
        compute_hashes=compute_member_hashes,
        compute_archive_hash=compute_archive_hash,
    )
    if manifest.has_errors:
        codes = sorted({warning.code for warning in manifest.warnings})
        raise ArchiveSafetyError(
            f"archive failed ZIP safety preflight ({', '.join(codes)})",
            manifest,
        )

    selected_files, selection_warnings = _select_conversation_files(manifest)
    declared_json_bytes = sum(source_file.size_bytes for source_file in selected_files)
    if declared_json_bytes > active_limits.max_total_json_bytes:
        raise ArchiveLimitError(
            "selected conversation payloads exceed the configured maximum total JSON bytes"
        )
    for selected_file in selected_files:
        if selected_file.size_bytes > active_limits.max_json_bytes:
            raise ArchiveLimitError(
                "a selected conversation payload exceeds the configured per-member JSON bytes"
            )

    selected_paths = [source_file.path for source_file in selected_files]
    manifest.selected_conversation_paths = selected_paths
    if len(selected_files) == 1:
        manifest.selected_conversation_path = selected_files[0].path

    payload_warnings = list(selection_warnings)
    observed_json_bytes = 0
    bom_member_count = 0
    first_bom_path: PurePosixPath | None = None

    def decoded_payloads() -> Iterator[tuple[PurePosixPath, object]]:
        """Read and decode selected parts incrementally in deterministic numeric order."""

        nonlocal bom_member_count, first_bom_path, observed_json_bytes
        for selected_file in selected_files:
            raw_payload = read_zip_member_bytes(
                archive_path,
                selected_file.path,
                max_bytes=active_limits.max_json_bytes,
                chunk_bytes=active_limits.read_chunk_bytes,
                max_compression_ratio=active_limits.max_compression_ratio,
                min_ratio_check_bytes=active_limits.min_ratio_check_bytes,
            )
            observed_json_bytes += len(raw_payload)
            if observed_json_bytes > active_limits.max_total_json_bytes:
                raise ArchiveLimitError(
                    "selected conversation payloads exceeded the streaming total JSON limit"
                )
            selected_hash = hashlib.sha256(raw_payload).hexdigest()
            selected_file.sha256 = selected_hash
            if len(selected_files) == 1:
                manifest.selected_conversation_sha256 = selected_hash

            payload, had_bom = _decode_json(raw_payload)
            if had_bom:
                bom_member_count += 1
                first_bom_path = first_bom_path or selected_file.path
            yield selected_file.path, payload
            del payload, raw_payload

    normalized = normalize_conversation_payloads(decoded_payloads(), limits=active_limits)
    if bom_member_count:
        payload_warnings.append(
            ArchiveWarning(
                severity=WarningSeverity.INFO,
                code="utf8_bom",
                message="UTF-8 byte-order marks were accepted and removed.",
                source_path=first_bom_path if bom_member_count == 1 else None,
                location="/",
                context={"member_count": bom_member_count},
            )
        )
    archive_warnings = [*payload_warnings, *normalized.warnings]
    conversation_warnings = [
        warning for conversation in normalized.conversations for warning in conversation.warnings
    ]
    if schema_mode is SchemaMode.STRICT and (archive_warnings or conversation_warnings):
        raise ArchiveSchemaError(
            "strict schema mode rejected one or more recoverable normalization warnings"
        )

    return Archive(
        source_manifest=manifest,
        conversations=normalized.conversations,
        warnings=sorted(
            archive_warnings,
            key=lambda warning: (warning.code, warning.location or ""),
        ),
        metadata={
            "compiler_version": __version__,
            "schema_mode": schema_mode.value,
            "compute_member_hashes": compute_member_hashes,
            "compute_archive_hash": compute_archive_hash,
            "conversation_payload_count": len(selected_files),
            "ingest_limits": active_limits.model_dump(mode="json"),
        },
    )


def summarize_archive(archive: Archive) -> ArchiveSummary:
    """Calculate content-free counts and temporal bounds for validation logs.

    Parameters
    ----------
    archive
        Normalized Archive IR.

    Returns
    -------
    ArchiveSummary
        Counts of conversations, graph nodes, messages, visible-path messages, and warnings.
    """

    node_count = sum(len(conversation.nodes) for conversation in archive.conversations)
    message_count = sum(
        node.message is not None
        for conversation in archive.conversations
        for node in conversation.nodes
    )
    current_path_message_count = sum(
        len(conversation.current_path_messages) for conversation in archive.conversations
    )
    timestamps = sorted(
        conversation.created_at
        for conversation in archive.conversations
        if conversation.created_at is not None
    )
    return ArchiveSummary(
        conversation_count=len(archive.conversations),
        node_count=node_count,
        message_count=message_count,
        current_path_message_count=current_path_message_count,
        warning_count=len(archive.all_warnings),
        earliest_conversation_at=timestamps[0] if timestamps else None,
        latest_conversation_at=timestamps[-1] if timestamps else None,
    )
