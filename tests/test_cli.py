"""Command-line interface smoke tests."""

from __future__ import annotations

from typer.testing import CliRunner

from chatgpt_archive_compiler import __version__
from chatgpt_archive_compiler.cli import app


def test_cli_reports_public_version() -> None:
    """The installed console entry point reports the canonical package version."""

    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_cli_exposes_reproducible_workflow_commands() -> None:
    """Help advertises normalization, local execution, and verification commands."""

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ingest", "compile", "semantic-local", "run-local", "verify"):
        assert command in result.stdout
