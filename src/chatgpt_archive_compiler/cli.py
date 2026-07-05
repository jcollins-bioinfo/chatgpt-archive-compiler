"""Command-line interface for ChatGPT Archive Compiler."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from chatgpt_archive_compiler import __version__
from chatgpt_archive_compiler.ingest import inspect_zip

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", help="Print version and exit."),
    ] = False,
) -> None:
    """Compile exported conversation archives into analyzed local artifacts."""
    if version:
        console.print(__version__)
        raise typer.Exit


@app.command()
def inspect(
    archive: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False),
    ],
    hashes: Annotated[
        bool,
        typer.Option("--hashes", help="Compute SHA-256 for archive members."),
    ] = False,
) -> None:
    """Inspect an export ZIP without extracting it."""
    manifest = inspect_zip(archive, compute_hashes=hashes)

    console.print(f"[bold]Archive:[/bold] {manifest.archive_name}")
    console.print(f"[bold]Files:[/bold] {len(manifest.files)}")
    console.print(f"[bold]Warnings:[/bold] {len(manifest.warnings)}")

    table = Table(title="Detected files")
    table.add_column("Path")
    table.add_column("Size bytes", justify="right")
    table.add_column("Kind")

    for file in manifest.files[:50]:
        table.add_row(str(file.path), str(file.size_bytes), file.detected_kind or "")

    console.print(table)

    if manifest.warnings:
        warning_table = Table(title="Warnings")
        warning_table.add_column("Severity")
        warning_table.add_column("Code")
        warning_table.add_column("Message")
        for warning in manifest.warnings:
            warning_table.add_row(warning.severity, warning.code, warning.message)
        console.print(warning_table)
