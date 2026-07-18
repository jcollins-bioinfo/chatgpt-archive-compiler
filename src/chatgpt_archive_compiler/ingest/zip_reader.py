"""Fail-closed inspection and bounded reading of untrusted ZIP archives."""

from __future__ import annotations

import hashlib
import stat
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import cast

from pydantic import JsonValue

from chatgpt_archive_compiler.exceptions import InvalidArchiveError
from chatgpt_archive_compiler.models import (
    ArchiveWarning,
    IngestLimits,
    SourceFile,
    SourceFileKind,
    SourceManifest,
    WarningSeverity,
)


def _canonical_member_path(raw_path: str) -> PurePosixPath:
    """Return a normalized POSIX path after rejecting traversal syntax.

    The check deliberately rejects backslashes and Windows drive prefixes rather than trying to
    reinterpret them. This keeps the same archive unambiguous on POSIX and Windows hosts.
    """

    if not raw_path or "\x00" in raw_path or "\\" in raw_path:
        raise ValueError("unsafe archive-member path")
    path = PurePosixPath(raw_path)
    first_part = path.parts[0] if path.parts else ""
    if (
        path.is_absolute()
        or raw_path.startswith("/")
        or ".." in path.parts
        or first_part.endswith(":")
    ):
        raise ValueError("unsafe archive-member path")
    return path


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    """Return whether a ZIP entry declares a Unix symbolic-link file mode."""

    unix_mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(unix_mode)


def _detect_kind(path: PurePosixPath) -> SourceFileKind:
    """Classify an archive path without inspecting private file contents."""

    name = path.name.casefold()
    if str(path) == "conversations.json":
        return SourceFileKind.CONVERSATIONS_JSON
    if name == "conversations.json" or (
        name.startswith("conversations") and name.endswith(".json")
    ):
        return SourceFileKind.CONVERSATIONS_JSON_CANDIDATE
    if name.endswith(".json"):
        return SourceFileKind.JSON
    return SourceFileKind.OTHER


def _compression_ratio(info: zipfile.ZipInfo) -> float:
    """Return a member's declared uncompressed-to-compressed size ratio."""

    if info.file_size == 0:
        return 0.0
    if info.compress_size == 0:
        return float("inf")
    return info.file_size / info.compress_size


def _hash_file(path: Path, chunk_bytes: int) -> str:
    """Calculate a SHA-256 digest for a local file using bounded reads."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    max_bytes: int,
    chunk_bytes: int,
) -> str:
    """Hash one decompressed member while enforcing an independent byte counter."""

    digest = hashlib.sha256()
    observed_bytes = 0
    with archive.open(info, "r") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            observed_bytes += len(chunk)
            if observed_bytes > max_bytes:
                raise InvalidArchiveError(
                    "archive member exceeded its configured streaming byte limit"
                )
            digest.update(chunk)
    if observed_bytes != info.file_size:
        raise InvalidArchiveError("archive member size did not match its central-directory entry")
    return digest.hexdigest()


def inspect_zip(
    archive_path: str | Path,
    *,
    limits: IngestLimits | None = None,
    compute_hashes: bool = False,
    compute_archive_hash: bool = True,
) -> SourceManifest:
    """Inspect a ZIP without extracting it or parsing semantic contents.

    Inspection is split into a metadata-only preflight and an optional hashing pass. If any
    security or resource boundary fails, no archive member is opened, even when
    ``compute_hashes`` is true.

    Parameters
    ----------
    archive_path
        Local path to the ChatGPT export ZIP.
    limits
        Hard resource boundaries. Defaults to :class:`IngestLimits`.
    compute_hashes
        Whether to compute decompressed SHA-256 for every safe member.
    compute_archive_hash
        Whether to compute SHA-256 for the compressed ZIP itself. This does not decompress data.

    Returns
    -------
    SourceManifest
        Deterministically ordered metadata and structured preflight warnings.

    Raises
    ------
    InvalidArchiveError
        If the path is missing, is not a regular file, cannot be read, or is not a valid ZIP.
    """

    path = Path(archive_path)
    active_limits = limits or IngestLimits()
    if not path.is_file():
        raise InvalidArchiveError(f"archive path is not a readable file: {path}")

    try:
        archive_size = path.stat().st_size
    except OSError as exc:
        raise InvalidArchiveError(f"could not stat archive: {path}") from exc

    warnings: list[ArchiveWarning] = []
    if archive_size > active_limits.max_archive_bytes:
        warnings.append(
            ArchiveWarning(
                severity=WarningSeverity.ERROR,
                code="archive_too_large",
                message="Compressed archive exceeds the configured byte limit.",
                context={
                    "archive_size_bytes": archive_size,
                    "max_archive_bytes": active_limits.max_archive_bytes,
                },
            )
        )

    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            if len(infos) > active_limits.max_files:
                warnings.append(
                    ArchiveWarning(
                        severity=WarningSeverity.ERROR,
                        code="too_many_files",
                        message="Archive exceeds the configured member-count limit.",
                        context={
                            "file_count": len(infos),
                            "max_files": active_limits.max_files,
                        },
                    )
                )

            total_uncompressed = sum(info.file_size for info in infos)
            total_compressed = sum(info.compress_size for info in infos)
            if total_uncompressed > active_limits.max_total_uncompressed_bytes:
                warnings.append(
                    ArchiveWarning(
                        severity=WarningSeverity.ERROR,
                        code="too_large_uncompressed",
                        message="Archive exceeds the configured total uncompressed-byte limit.",
                        context={
                            "total_uncompressed_bytes": total_uncompressed,
                            "max_total_uncompressed_bytes": (
                                active_limits.max_total_uncompressed_bytes
                            ),
                        },
                    )
                )

            canonical_by_index: dict[int, PurePosixPath] = {}
            indices_by_casefolded_path: dict[str, list[int]] = defaultdict(list)
            for index, info in enumerate(infos):
                try:
                    canonical_path = _canonical_member_path(info.filename)
                except ValueError:
                    warnings.append(
                        ArchiveWarning(
                            severity=WarningSeverity.ERROR,
                            code="unsafe_zip_path",
                            message="Archive contains an unsafe member path.",
                            context={"member_index": index},
                        )
                    )
                    continue
                canonical_by_index[index] = canonical_path
                indices_by_casefolded_path[str(canonical_path).casefold()].append(index)

            duplicate_indices: set[int] = set()
            for duplicate_group in indices_by_casefolded_path.values():
                if len(duplicate_group) > 1:
                    duplicate_indices.update(duplicate_group)
                    warnings.append(
                        ArchiveWarning(
                            severity=WarningSeverity.ERROR,
                            code="duplicate_member_path",
                            message="Archive contains duplicate or case-colliding member paths.",
                            context={"member_indices": cast(JsonValue, duplicate_group)},
                        )
                    )

            files_with_info: list[tuple[SourceFile, zipfile.ZipInfo]] = []
            for index, info in enumerate(infos):
                member_path = canonical_by_index.get(index)
                if member_path is None or info.is_dir() or index in duplicate_indices:
                    continue

                readable = True
                if info.flag_bits & 0x1:
                    readable = False
                    warnings.append(
                        ArchiveWarning(
                            severity=WarningSeverity.ERROR,
                            code="encrypted_member",
                            message="Encrypted ZIP members are not supported.",
                            source_path=member_path,
                        )
                    )
                if _is_symlink(info):
                    readable = False
                    warnings.append(
                        ArchiveWarning(
                            severity=WarningSeverity.ERROR,
                            code="symlink_member",
                            message="Symbolic-link ZIP members are not supported.",
                            source_path=member_path,
                        )
                    )
                if info.file_size > active_limits.max_member_uncompressed_bytes:
                    readable = False
                    warnings.append(
                        ArchiveWarning(
                            severity=WarningSeverity.ERROR,
                            code="member_too_large",
                            message="Archive member exceeds the configured byte limit.",
                            source_path=member_path,
                            context={
                                "size_bytes": info.file_size,
                                "max_member_uncompressed_bytes": (
                                    active_limits.max_member_uncompressed_bytes
                                ),
                            },
                        )
                    )
                ratio = _compression_ratio(info)
                if (
                    info.file_size >= active_limits.min_ratio_check_bytes
                    and ratio > active_limits.max_compression_ratio
                ):
                    readable = False
                    warnings.append(
                        ArchiveWarning(
                            severity=WarningSeverity.ERROR,
                            code="suspicious_compression_ratio",
                            message="Archive member exceeds the configured compression ratio.",
                            source_path=member_path,
                            context={
                                "compression_ratio": ratio,
                                "max_compression_ratio": active_limits.max_compression_ratio,
                            },
                        )
                    )

                files_with_info.append(
                    (
                        SourceFile(
                            path=member_path,
                            size_bytes=info.file_size,
                            compressed_size_bytes=info.compress_size,
                            crc32=info.CRC,
                            compression_method=info.compress_type,
                            detected_kind=_detect_kind(member_path),
                            is_readable=readable,
                        ),
                        info,
                    )
                )

            files_with_info.sort(key=lambda pair: (str(pair[0].path).casefold(), str(pair[0].path)))
            preflight_has_errors = any(
                warning.severity is WarningSeverity.ERROR for warning in warnings
            )
            archive_sha256: str | None = None
            if compute_archive_hash and not preflight_has_errors:
                archive_sha256 = _hash_file(path, active_limits.read_chunk_bytes)
            if compute_hashes and not preflight_has_errors:
                for source_file, info in files_with_info:
                    try:
                        source_file.sha256 = _hash_member(
                            archive,
                            info,
                            max_bytes=min(
                                active_limits.max_member_uncompressed_bytes,
                                info.file_size,
                            ),
                            chunk_bytes=active_limits.read_chunk_bytes,
                        )
                    except (InvalidArchiveError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
                        raise InvalidArchiveError(
                            f"failed to read archive member: {source_file.path}"
                        ) from exc

    except zipfile.BadZipFile as exc:
        raise InvalidArchiveError(f"input is not a valid ZIP archive: {path}") from exc
    except OSError as exc:
        raise InvalidArchiveError(f"could not read ZIP archive: {path}") from exc

    return SourceManifest(
        archive_name=path.name,
        archive_size_bytes=archive_size,
        archive_sha256=archive_sha256,
        files=[source_file for source_file, _ in files_with_info],
        total_uncompressed_bytes=total_uncompressed,
        total_compressed_bytes=total_compressed,
        warnings=warnings,
    )


def read_zip_member_bytes(
    archive_path: str | Path,
    member_path: str | PurePosixPath,
    *,
    max_bytes: int,
    chunk_bytes: int = 1024**2,
    max_compression_ratio: float = 250.0,
    min_ratio_check_bytes: int = 1024**2,
) -> bytes:
    """Read exactly one safe ZIP member with metadata and streaming size checks.

    Parameters
    ----------
    archive_path
        Local ZIP path.
    member_path
        Exact canonical POSIX member path selected from a trusted :class:`SourceManifest`.
    max_bytes
        Maximum declared and observed decompressed size.
    chunk_bytes
        Bounded read size for the decompression stream.
    max_compression_ratio
        Maximum allowed declared uncompressed-to-compressed ratio.
    min_ratio_check_bytes
        Minimum uncompressed member size at which the ratio limit is applied.

    Returns
    -------
    bytes
        Complete decompressed member bytes.

    Raises
    ------
    InvalidArchiveError
        If the path is unsafe, missing, duplicated, encrypted, oversized, corrupt, or changes size
        during decompression.
    ValueError
        If a numeric read or compression limit is not positive.
    """

    path = Path(archive_path)
    if (
        max_bytes <= 0
        or chunk_bytes <= 0
        or max_compression_ratio <= 0
        or min_ratio_check_bytes < 0
    ):
        raise ValueError("member read limits must be positive")
    try:
        canonical = _canonical_member_path(str(member_path))
    except ValueError as exc:
        raise InvalidArchiveError("requested ZIP member path is unsafe") from exc

    try:
        with zipfile.ZipFile(path, "r") as archive:
            matches = [
                info
                for info in archive.infolist()
                if not info.is_dir() and _safe_canonical_equals(info.filename, canonical)
            ]
            if len(matches) != 1:
                raise InvalidArchiveError(
                    "requested ZIP member is missing or has an ambiguous duplicate path"
                )
            info = matches[0]
            if info.flag_bits & 0x1 or _is_symlink(info):
                raise InvalidArchiveError("requested ZIP member is not safely readable")
            if info.file_size > max_bytes:
                raise InvalidArchiveError("requested ZIP member exceeds its configured byte limit")
            if (
                info.file_size >= min_ratio_check_bytes
                and _compression_ratio(info) > max_compression_ratio
            ):
                raise InvalidArchiveError(
                    "requested ZIP member exceeds its configured compression ratio"
                )

            chunks: list[bytes] = []
            observed_bytes = 0
            with archive.open(info, "r") as handle:
                while True:
                    chunk = handle.read(chunk_bytes)
                    if not chunk:
                        break
                    observed_bytes += len(chunk)
                    if observed_bytes > max_bytes:
                        raise InvalidArchiveError(
                            "ZIP member exceeded its streaming decompressed-byte limit"
                        )
                    chunks.append(chunk)
            if observed_bytes != info.file_size:
                raise InvalidArchiveError(
                    "ZIP member size did not match its central-directory declaration"
                )
            return b"".join(chunks)
    except zipfile.BadZipFile as exc:
        raise InvalidArchiveError(f"input is not a valid ZIP archive: {path}") from exc
    except OSError as exc:
        raise InvalidArchiveError(f"could not read ZIP archive: {path}") from exc


def _safe_canonical_equals(raw_path: str, expected: PurePosixPath) -> bool:
    """Compare a raw member name with a canonical path without propagating path errors."""

    try:
        return _canonical_member_path(raw_path) == expected
    except ValueError:
        return False
