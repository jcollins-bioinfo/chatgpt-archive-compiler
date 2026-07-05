"""Safe ZIP inspection for source export archives."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path, PurePosixPath

from chatgpt_archive_compiler.models import (
    ArchiveWarning,
    SourceFile,
    SourceManifest,
    WarningSeverity,
)

DEFAULT_MAX_FILES = 20_000
DEFAULT_MAX_TOTAL_UNCOMPRESSED_BYTES = 5 * 1024**3


def _is_unsafe_zip_path(path: str) -> bool:
    posix_path = PurePosixPath(path)
    has_parent_ref = ".." in posix_path.parts
    has_backslash = "\\" in path
    return posix_path.is_absolute() or has_parent_ref or path.startswith("/") or has_backslash


def _detect_kind(path: PurePosixPath) -> str | None:
    name = path.name.lower()
    if name == "conversations.json":
        return "conversations_json"
    if name.startswith("conversations") and name.endswith(".json"):
        return "conversation_json_candidate"
    if "shared" in name and name.endswith(".json"):
        return "shared_json_candidate"
    if name.endswith(".json"):
        return "json"
    return None


def inspect_zip(
    archive_path: str | Path,
    *,
    compute_hashes: bool = False,
    max_files: int = DEFAULT_MAX_FILES,
    max_total_uncompressed_bytes: int = DEFAULT_MAX_TOTAL_UNCOMPRESSED_BYTES,
) -> SourceManifest:
    """Inspect a ZIP archive without extracting it.

    The function returns a manifest plus warnings. It does not parse semantic contents.
    """

    archive_path = Path(archive_path)
    warnings: list[ArchiveWarning] = []
    files: list[SourceFile] = []
    total_size = 0

    with zipfile.ZipFile(archive_path) as zf:
        infos = zf.infolist()

        if len(infos) > max_files:
            warnings.append(
                ArchiveWarning(
                    severity=WarningSeverity.ERROR,
                    code="too_many_files",
                    message="Archive exceeds configured file-count limit.",
                    context={"file_count": len(infos), "max_files": max_files},
                )
            )

        for info in infos:
            total_size += info.file_size
            path = PurePosixPath(info.filename)

            if _is_unsafe_zip_path(info.filename):
                warnings.append(
                    ArchiveWarning(
                        severity=WarningSeverity.ERROR,
                        code="unsafe_zip_path",
                        message=(
                            "Archive member has an unsafe path and must not be "
                            "extracted blindly."
                        ),
                        context={"path": info.filename},
                    )
                )
                continue

            sha256 = None
            if compute_hashes and not info.is_dir():
                digest = hashlib.sha256()
                with zf.open(info, "r") as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        digest.update(chunk)
                sha256 = digest.hexdigest()

            if not info.is_dir():
                files.append(
                    SourceFile(
                        path=path,
                        size_bytes=info.file_size,
                        sha256=sha256,
                        detected_kind=_detect_kind(path),
                    )
                )

    if total_size > max_total_uncompressed_bytes:
        warnings.append(
            ArchiveWarning(
                severity=WarningSeverity.ERROR,
                code="too_large_uncompressed",
                message="Archive exceeds configured uncompressed-size limit.",
                context={
                    "total_uncompressed_bytes": total_size,
                    "max_total_uncompressed_bytes": max_total_uncompressed_bytes,
                },
            )
        )

    return SourceManifest(
        archive_name=archive_path.name,
        files=files,
        warnings=warnings,
    )
