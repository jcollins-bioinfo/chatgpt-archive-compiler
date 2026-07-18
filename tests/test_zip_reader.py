"""Security and resource-boundary tests for ZIP inspection."""

from __future__ import annotations

import hashlib
import stat
import zipfile
from pathlib import Path

import pytest

from chatgpt_archive_compiler.exceptions import InvalidArchiveError
from chatgpt_archive_compiler.ingest import IngestLimits, inspect_zip, read_zip_member_bytes
from chatgpt_archive_compiler.models import WarningSeverity


def test_safe_member_is_manifested_and_hashed(write_zip, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """A safe member is classified and hashed without extraction."""

    content = b"[]"
    archive_path = write_zip({"conversations.json": content})
    manifest = inspect_zip(archive_path, compute_hashes=True)

    assert manifest.archive_sha256 == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert len(manifest.files) == 1
    assert manifest.files[0].sha256 == hashlib.sha256(content).hexdigest()
    assert not (tmp_path / "conversations.json").exists()


def test_total_size_limit_blocks_member_opening(
    write_zip, monkeypatch: pytest.MonkeyPatch  # type: ignore[no-untyped-def]
) -> None:
    """Preflight size failure prevents every decompression and hashing call."""

    archive_path = write_zip({"conversations.json": b"x" * 1_025})

    def forbidden_open(*args: object, **kwargs: object) -> None:
        raise AssertionError("member data was opened after preflight failure")

    monkeypatch.setattr(zipfile.ZipFile, "open", forbidden_open)
    limits = IngestLimits(max_total_uncompressed_bytes=1_024)
    manifest = inspect_zip(archive_path, limits=limits, compute_hashes=True)

    assert [warning.code for warning in manifest.warnings] == ["too_large_uncompressed"]
    assert manifest.warnings[0].severity is WarningSeverity.ERROR
    assert manifest.archive_sha256 is None


@pytest.mark.parametrize(
    "member_name",
    ["../x.json", "/x.json", "a/../../x.json", "..\\x.json", "C:\\x.json"],
)
def test_unsafe_paths_are_rejected(write_zip, member_name: str) -> None:  # type: ignore[no-untyped-def]
    """Cross-platform traversal and absolute path spellings fail preflight."""

    archive_path = write_zip({member_name: "[]"})
    manifest = inspect_zip(archive_path)

    assert "unsafe_zip_path" in {warning.code for warning in manifest.warnings}
    assert manifest.files == []


def test_parent_like_substring_is_safe(write_zip) -> None:  # type: ignore[no-untyped-def]
    """A filename containing two dots as a substring is not a traversal false positive."""

    archive_path = write_zip({"folder/a..b.json": "[]"})
    manifest = inspect_zip(archive_path)

    assert not manifest.has_errors
    assert str(manifest.files[0].path) == "folder/a..b.json"


def test_duplicate_paths_are_fatal_and_excluded(tmp_path: Path) -> None:
    """Duplicate member names never remain eligible for semantic payload selection."""

    archive_path = tmp_path / "duplicates.zip"
    with (
        pytest.warns(UserWarning, match="Duplicate name"),
        zipfile.ZipFile(archive_path, "w") as archive,
    ):
        archive.writestr("conversations.json", "[]")
        archive.writestr("conversations.json", "[]")
    manifest = inspect_zip(archive_path)

    assert "duplicate_member_path" in {warning.code for warning in manifest.warnings}
    assert manifest.files == []


def test_symlink_member_is_rejected(tmp_path: Path) -> None:
    """Unix symlink metadata is a fatal preflight condition."""

    archive_path = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("conversations.json")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(info, "target")
    manifest = inspect_zip(archive_path)

    assert "symlink_member" in {warning.code for warning in manifest.warnings}


def test_bounded_reader_checks_declared_limit_before_opening(write_zip) -> None:  # type: ignore[no-untyped-def]
    """The selected-member reader rejects declared oversize before decompression."""

    archive_path = write_zip({"conversations.json": "0123456789"})
    with pytest.raises(InvalidArchiveError):
        read_zip_member_bytes(archive_path, "conversations.json", max_bytes=5)


def test_invalid_zip_is_typed_failure(tmp_path: Path) -> None:
    """Non-ZIP input raises the documented exception instead of leaking BadZipFile."""

    path = tmp_path / "not-a-zip.zip"
    path.write_bytes(b"not a zip")
    with pytest.raises(InvalidArchiveError):
        inspect_zip(path)
