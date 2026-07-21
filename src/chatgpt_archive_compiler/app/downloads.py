"""Typed boundary around Dash's dynamically exported download helper."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from dash import dcc

_SendFile = Callable[[str], dict[str, Any]]
_send_file = cast(_SendFile, getattr(dcc, "send_file"))  # noqa: B009


def make_download_payload(path: Path) -> dict[str, Any]:
    """Create Dash download data without leaking dynamic typing into callbacks."""

    return _send_file(str(path))
