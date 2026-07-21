"""Isolated application sessions and transparent correction persistence."""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime
from pathlib import Path

from chatgpt_archive_compiler.semantic.domain import UserCorrection


class SessionStore:
    """Allocate opaque, non-guessable disk sessions below one configured root."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self) -> str:
        """Create and return an opaque session identifier."""

        identifier = secrets.token_urlsafe(18)
        (self.root / identifier).mkdir(mode=0o700)
        return identifier

    def path(self, identifier: str) -> Path:
        """Resolve a known safe session or reject traversal/unknown identifiers."""

        if not identifier or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in identifier
        ):
            raise ValueError("Invalid session identifier.")
        path = (self.root / identifier).resolve()
        if path.parent != self.root or not path.is_dir():
            raise ValueError("Unknown session identifier.")
        return path


def append_correction(path: Path, correction: UserCorrection) -> None:
    """Append a validated correction using a readable versioned JSON document."""

    records: list[object] = []
    if path.is_file():
        records = json.loads(path.read_text(encoding="utf-8"))["corrections"]
    records.append(correction.model_dump(mode="json"))
    path.write_text(
        json.dumps({"schema_version": "1.0", "corrections": records}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def make_correction(
    correction_type: str, object_id: str, old_value: object, new_value: object
) -> UserCorrection:
    """Create an explicit user-authored correction."""

    return UserCorrection(
        correction_type=correction_type,
        object_id=object_id,
        old_value=old_value,
        new_value=new_value,
        timestamp=datetime.now(UTC),
    )
