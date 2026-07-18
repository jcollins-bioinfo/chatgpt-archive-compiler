"""Synthetic-only pytest fixtures for archive ingestion."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def branched_payload() -> list[dict[str, Any]]:
    """Return one synthetic conversation with a current and alternate assistant branch."""

    return [
        {
            "id": "synthetic-conversation-001",
            "title": "Synthetic branched conversation",
            "create_time": 1_735_689_600,
            "update_time": 1_735_689_700,
            "current_node": "assistant-current",
            "mapping": {
                "root": {
                    "id": "root",
                    "parent": None,
                    "children": ["user-001"],
                    "message": None,
                },
                "user-001": {
                    "id": "user-001",
                    "parent": "root",
                    "children": ["assistant-current", "assistant-alternate"],
                    "message": {
                        "id": "message-user-001",
                        "author": {"role": "user"},
                        "create_time": 1_735_689_600,
                        "content": {"content_type": "text", "parts": ["Synthetic question"]},
                        "metadata": {},
                    },
                },
                "assistant-current": {
                    "id": "assistant-current",
                    "parent": "user-001",
                    "children": [],
                    "message": {
                        "id": "message-assistant-current",
                        "author": {"role": "assistant"},
                        "create_time": 1_735_689_700,
                        "content": {"content_type": "text", "parts": ["Current answer"]},
                        "metadata": {"model_slug": "synthetic-model"},
                    },
                },
                "assistant-alternate": {
                    "id": "assistant-alternate",
                    "parent": "user-001",
                    "children": [],
                    "message": {
                        "id": "message-assistant-alternate",
                        "author": {"role": "assistant"},
                        "create_time": 1_735_689_650,
                        "content": {"content_type": "text", "parts": ["Alternate answer"]},
                        "metadata": {},
                    },
                },
            },
        }
    ]


@pytest.fixture
def write_zip(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Return a helper that writes explicit synthetic member bytes to a new ZIP."""

    def _write_zip(
        members: Mapping[str, bytes | str],
        *,
        name: str = "synthetic-export.zip",
    ) -> Path:
        path = tmp_path / name
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for member_name, value in members.items():
                archive.writestr(member_name, value)
        return path

    return _write_zip


@pytest.fixture
def write_payload_zip(write_zip):  # type: ignore[no-untyped-def]
    """Return a helper that JSON-encodes a synthetic conversation payload."""

    def _write_payload_zip(payload: Any, *, member_name: str = "conversations.json") -> Path:
        return write_zip({member_name: json.dumps(payload, allow_nan=False)})

    return _write_payload_zip
