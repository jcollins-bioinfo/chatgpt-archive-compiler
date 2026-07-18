"""Canonical, atomic serialization of Archive IR files."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from chatgpt_archive_compiler.exceptions import ArchiveSerializationError
from chatgpt_archive_compiler.models import Archive


def write_archive_ir(archive: Archive, path: str | Path) -> Path:
    """Write canonical UTF-8 Archive IR JSON using an atomic sibling replacement.

    Parameters
    ----------
    archive
        Validated Archive IR object to serialize.
    path
        Destination filename, conventionally ``archive.ir.json``. Parent directories are created.

    Returns
    -------
    Path
        Resolved destination path after a successful atomic replacement.

    Raises
    ------
    ArchiveSerializationError
        If JSON conversion, temporary-file creation, flushing, or replacement fails.
    """

    destination = Path(path).expanduser().resolve()
    temporary_path: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = archive.model_dump(mode="json", round_trip=True)
        canonical = (
            json.dumps(
                data,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(canonical)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    except (OSError, TypeError, ValueError) as exc:
        raise ArchiveSerializationError(destination, "could not write Archive IR") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def read_archive_ir(path: str | Path) -> Archive:
    """Read and validate a canonical Archive IR JSON file.

    Parameters
    ----------
    path
        Local Archive IR JSON path.

    Returns
    -------
    Archive
        Fully validated Archive IR.

    Raises
    ------
    ArchiveSerializationError
        If the file cannot be read or does not validate as Archive IR v1.
    """

    source = Path(path).expanduser().resolve()
    try:
        return Archive.model_validate_json(source.read_bytes())
    except (OSError, ValidationError, ValueError) as exc:
        raise ArchiveSerializationError(source, "could not read valid Archive IR") from exc
