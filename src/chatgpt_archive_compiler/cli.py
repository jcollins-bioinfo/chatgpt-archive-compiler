"""Command-line interface for safe, reproducible archive workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from chatgpt_archive_compiler import __version__
from chatgpt_archive_compiler.compiler import CompileOptions, VolumeMode, compile_archive
from chatgpt_archive_compiler.ingest import SchemaMode, ingest_export_zip, inspect_zip
from chatgpt_archive_compiler.semantic import (
    LocalHashingEmbeddingProvider,
    LocalHeuristicAnalysisProvider,
    SemanticAtlasOptions,
    build_semantic_atlas,
)
from chatgpt_archive_compiler.serialization import read_archive_ir, write_archive_ir
from chatgpt_archive_compiler.workflows import (
    LocalWorkflowOptions,
    run_local_workflow,
    verify_local_workflow,
)

app = typer.Typer(
    no_args_is_help=True,
    invoke_without_command=True,
    pretty_exceptions_show_locals=False,
)
console = Console()


def _abort_safely(stage: str, exception: BaseException) -> NoReturn:
    """Exit with a content-free failure diagnostic suitable for private archives."""

    console.print(f"[red]{stage} failed ({type(exception).__name__}).[/red]")
    raise typer.Exit(code=1)


def _progress_printer(stage: str, completed: int, total: int, cached: int) -> None:
    """Print content-free semantic stage progress."""

    console.print(f"[{stage}] {completed:,}/{total:,} (cached: {cached:,})", markup=False)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", help="Print version and exit."),
    ] = False,
) -> None:
    """Inspect, normalize, compile, and organize ChatGPT data exports locally."""

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

    try:
        manifest = inspect_zip(archive, compute_hashes=hashes)
    except Exception as exception:
        _abort_safely("Archive inspection", exception)

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


@app.command()
def ingest(
    archive: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False),
    ],
    output_ir: Annotated[Path, typer.Argument(file_okay=True, dir_okay=False)],
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Reject every recoverable source-schema warning."),
    ] = False,
) -> None:
    """Normalize an export ZIP into canonical, loss-aware Archive IR JSON."""

    try:
        normalized = ingest_export_zip(
            archive,
            schema_mode=SchemaMode.STRICT if strict else SchemaMode.TOLERANT,
            compute_archive_hash=True,
        )
        destination = write_archive_ir(normalized, output_ir)
    except Exception as exception:
        _abort_safely("Archive ingestion", exception)
    console.print(f"Wrote Archive IR: {destination}")
    console.print(f"Conversations: {len(normalized.conversations):,}")
    console.print(f"Warnings: {len(normalized.all_warnings):,}")


@app.command("compile")
def compile_command(
    archive_ir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False),
    ],
    output_directory: Annotated[Path, typer.Argument(file_okay=False, dir_okay=True)],
    pdf: Annotated[
        bool,
        typer.Option("--pdf", help="Render PDF in addition to self-contained HTML."),
    ] = False,
    volume_mode: Annotated[
        VolumeMode,
        typer.Option(help="Split the chronological archive by month, year, or not at all."),
    ] = VolumeMode.MONTH,
) -> None:
    """Compile canonical Archive IR into chronological HTML and optional PDF volumes."""

    try:
        archive = read_archive_ir(archive_ir)
        result = compile_archive(
            archive,
            output_directory,
            options=CompileOptions(render_pdf=pdf, volume_mode=volume_mode),
        )
    except Exception as exception:
        _abort_safely("Archive compilation", exception)
    console.print(f"Compilation manifest: {result.manifest_path}")
    console.print(f"Volumes: {len(result.volumes):,}")


@app.command("semantic-local")
def semantic_local(
    archive_ir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False),
    ],
    output_directory: Annotated[Path, typer.Argument(file_okay=False, dir_okay=True)],
) -> None:
    """Build a deterministic, network-free semantic baseline from Archive IR."""

    try:
        archive = read_archive_ir(archive_ir)
        analysis = LocalHeuristicAnalysisProvider()
        result = build_semantic_atlas(
            archive,
            output_directory,
            embedding_provider=LocalHashingEmbeddingProvider(),
            analysis_provider=analysis,
            interpretation_provider=analysis,
            options=SemanticAtlasOptions(),
            progress_callback=_progress_printer,
        )
    except Exception as exception:
        _abort_safely("Local semantic analysis", exception)
    console.print(f"Semantic manifest: {result.manifest_path}")
    console.print(f"Conversations represented: {result.estimate.conversation_count:,}")


@app.command("run-local")
def run_local(
    archive: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False),
    ],
    output_directory: Annotated[Path, typer.Argument(file_okay=False, dir_okay=True)],
    pdf: Annotated[
        bool,
        typer.Option("--pdf", help="Render chronological PDFs; requires the pdf extra."),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Reject every recoverable source-schema warning."),
    ] = False,
) -> None:
    """Run the complete local pipeline and write a verifiable SHA-256 manifest."""

    options = LocalWorkflowOptions(
        schema_mode=SchemaMode.STRICT if strict else SchemaMode.TOLERANT,
        compile_options=CompileOptions(render_pdf=pdf),
    )
    try:
        result = run_local_workflow(
            archive,
            output_directory,
            options=options,
            progress_callback=_progress_printer,
        )
    except Exception as exception:
        _abort_safely("Local workflow", exception)
    console.print(f"Run manifest: {result.manifest_path}")
    console.print(f"Verified artifacts recorded: {len(result.manifest.artifacts):,}")


@app.command("verify")
def verify_command(
    output_directory: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True),
    ],
) -> None:
    """Verify every artifact in a completed local workflow against its SHA-256 manifest."""

    try:
        verification = verify_local_workflow(output_directory)
    except Exception as exception:
        _abort_safely("Workflow verification", exception)
    if not verification.valid:
        for problem in verification.problems:
            console.print(f"[red]{problem}[/red]")
        raise typer.Exit(code=1)
    console.print(f"Verified {verification.checked_artifact_count:,} artifacts.")


@app.command("app")
def app_command(
    host: Annotated[str, typer.Option(help="Local interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8050,
    output_directory: Annotated[
        Path, typer.Option(help="Private writable session and artifact directory.")
    ] = Path("archive-output"),
) -> None:
    """Launch the local-first interactive semantic atlas."""

    try:
        from chatgpt_archive_compiler.app import create_app

        dash_app = create_app(data_root=output_directory)
        console.print(f"Private local application: http://{host}:{port}")
        dash_app.run(host=host, port=port, debug=False)
    except Exception as exception:
        _abort_safely("Application launch", exception)
