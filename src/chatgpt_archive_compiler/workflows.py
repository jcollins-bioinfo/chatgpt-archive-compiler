"""Reproducible, local-only workflow orchestration and artifact verification."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chatgpt_archive_compiler.compiler import (
    CompilationResult,
    CompileOptions,
    compile_archive,
)
from chatgpt_archive_compiler.ingest import SchemaMode, ingest_export_zip
from chatgpt_archive_compiler.semantic import (
    LocalHashingEmbeddingProvider,
    LocalHeuristicAnalysisProvider,
    SemanticAtlasOptions,
    SemanticAtlasResult,
    SemanticProgressCallback,
    build_semantic_atlas,
)
from chatgpt_archive_compiler.serialization import write_archive_ir
from chatgpt_archive_compiler.version import __version__


class LocalWorkflowError(RuntimeError):
    """Raised when a reproducible local workflow cannot safely run or be verified."""


class LocalWorkflowOptions(BaseModel):
    """Validated configuration for one network-free end-to-end workflow.

    Parameters
    ----------
    schema_mode
        Whether recoverable source-schema drift is retained or rejected.
    compile_options
        Deterministic chronological compiler configuration.
    semantic_options
        Deterministic local semantic-atlas configuration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_mode: SchemaMode = SchemaMode.TOLERANT
    compile_options: CompileOptions = Field(default_factory=CompileOptions)
    semantic_options: SemanticAtlasOptions = Field(default_factory=SemanticAtlasOptions)


class WorkflowArtifact(BaseModel):
    """One workflow artifact and its content-integrity metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class RuntimeProvenance(BaseModel):
    """Content-free runtime and dependency identity for one local workflow."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    python_version: str
    python_implementation: str
    operating_system: str
    machine: str
    source_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    lockfile_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    dependency_versions: dict[str, str]


class LocalWorkflowManifest(BaseModel):
    """Content-free provenance manifest for a complete local workflow run."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    workflow_version: str = "1.0"
    package_version: str
    runtime: RuntimeProvenance
    archive_version: str
    source_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_archive_size_bytes: int = Field(ge=0)
    options: LocalWorkflowOptions
    artifacts: tuple[WorkflowArtifact, ...]


class LocalWorkflowResult(BaseModel):
    """Paths and integrity metadata returned by :func:`run_local_workflow`."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    output_directory: Path
    archive_ir_path: Path
    compiled_directory: Path
    semantic_directory: Path
    manifest_path: Path
    manifest: LocalWorkflowManifest


class WorkflowVerification(BaseModel):
    """Result of checking every artifact declared by a local workflow manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    manifest_path: Path
    checked_artifact_count: int = Field(ge=0)
    valid: bool
    problems: tuple[str, ...] = ()


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one local file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_root() -> Path | None:
    """Return the editable source root when this package is running from a checkout."""

    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "pyproject.toml").is_file() else None


def _source_commit(project_root: Path | None) -> str | None:
    """Read the current checkout commit without exposing remote or path metadata."""

    configured = os.environ.get("CAC_SOURCE_COMMIT", "").strip().casefold()
    if re.fullmatch(r"[0-9a-f]{40}", configured):
        return configured
    if project_root is None or not (project_root / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=False,
        text=True,
        capture_output=True,
    )
    commit = result.stdout.strip().casefold()
    return commit if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", commit) else None


def _distribution_versions() -> dict[str, str]:
    """Return exact installed versions for dependencies that affect durable artifacts."""

    names = (
        "beautifulsoup4",
        "markdown-it-py",
        "polars",
        "pydantic",
        "scikit-learn",
        "weasyprint",
    )
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def _runtime_provenance() -> RuntimeProvenance:
    """Capture content-free runtime, source, lock, and dependency identity."""

    project_root = _project_root()
    lockfile = project_root / "uv.lock" if project_root is not None else None
    return RuntimeProvenance(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        operating_system=platform.system(),
        machine=platform.machine(),
        source_commit=_source_commit(project_root),
        lockfile_sha256=_sha256(lockfile) if lockfile is not None and lockfile.is_file() else None,
        dependency_versions=_distribution_versions(),
    )


def _atomic_json(path: Path, value: BaseModel) -> Path:
    """Write one validated model as canonical JSON using a sibling atomic replacement."""

    destination = path.expanduser().resolve()
    temporary_path: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(
                value.model_dump(mode="json"),
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
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    except (OSError, TypeError, ValueError) as exc:
        raise LocalWorkflowError("Could not write the local workflow manifest.") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _result_artifact_paths(
    archive_ir_path: Path,
    compilation: CompilationResult,
    semantic: SemanticAtlasResult,
) -> tuple[Path, ...]:
    """Return every durable, user-facing artifact produced by the local workflow."""

    paths = {
        archive_ir_path,
        compilation.analysis_path,
        compilation.index_path,
        compilation.manifest_path,
        semantic.catalog_path,
        semantic.embeddings_path,
        semantic.graph_path,
        semantic.taxonomy_path,
        semantic.category_profiles_path,
        semantic.project_timelines_path,
        semantic.synthesis_path,
        semantic.review_queue_path,
        semantic.atlas_html_path,
        semantic.manifest_path,
    }
    for volume in compilation.volumes:
        paths.add(volume.html_path)
        if volume.pdf_path is not None:
            paths.add(volume.pdf_path)
    return tuple(sorted(paths))


def _artifact_record(path: Path, root: Path) -> WorkflowArtifact:
    """Build an integrity record for one artifact beneath the workflow root."""

    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise LocalWorkflowError("A workflow artifact escaped the output directory.") from exc
    return WorkflowArtifact(
        relative_path=PurePosixPath(relative).as_posix(),
        sha256=_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


def run_local_workflow(
    archive_path: str | Path,
    output_directory: str | Path,
    *,
    options: LocalWorkflowOptions | None = None,
    progress_callback: SemanticProgressCallback | None = None,
) -> LocalWorkflowResult:
    """Run ingestion, compilation, and semantic organization without network access.

    The destination must be empty so artifacts from different inputs or configurations cannot be
    silently mixed. The workflow hashes its source ZIP, writes canonical Archive IR, runs the
    chronological compiler, builds a deterministic local semantic baseline, and records every
    durable artifact in a content-free SHA-256 manifest.

    Parameters
    ----------
    archive_path
        ChatGPT data-export ZIP to process without extraction.
    output_directory
        New or empty directory that will receive all durable products.
    options
        Validated workflow configuration. Defaults to safe local-only settings.
    progress_callback
        Optional content-free semantic stage callback receiving ``(stage, completed, total)``.

    Returns
    -------
    LocalWorkflowResult
        Output paths plus the validated run manifest.

    Raises
    ------
    LocalWorkflowError
        If the destination is non-empty or provenance/artifact recording fails.
    """

    active_options = options or LocalWorkflowOptions()
    source = Path(archive_path).expanduser().resolve()
    destination = Path(output_directory).expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise LocalWorkflowError("The local workflow output path must be a directory.")
    if destination.exists() and any(destination.iterdir()):
        raise LocalWorkflowError("The local workflow output directory must be empty.")
    destination.mkdir(parents=True, exist_ok=True)

    archive = ingest_export_zip(
        source,
        schema_mode=active_options.schema_mode,
        compute_archive_hash=True,
    )
    source_hash = archive.source_manifest.archive_sha256
    if source_hash is None:
        raise LocalWorkflowError("Source archive provenance hash is unavailable.")

    archive_ir_path = write_archive_ir(archive, destination / "archive.ir.json")
    compiled_directory = destination / "compiled"
    compilation = compile_archive(
        archive,
        compiled_directory,
        options=active_options.compile_options,
    )
    semantic_directory = destination / "semantic"
    local_analysis = LocalHeuristicAnalysisProvider()
    semantic = build_semantic_atlas(
        archive,
        semantic_directory,
        embedding_provider=LocalHashingEmbeddingProvider(),
        analysis_provider=local_analysis,
        interpretation_provider=local_analysis,
        options=active_options.semantic_options,
        progress_callback=progress_callback,
    )
    artifacts = tuple(
        _artifact_record(path, destination)
        for path in _result_artifact_paths(archive_ir_path, compilation, semantic)
    )
    manifest = LocalWorkflowManifest(
        package_version=__version__,
        runtime=_runtime_provenance(),
        archive_version=archive.archive_version,
        source_archive_sha256=source_hash,
        source_archive_size_bytes=source.stat().st_size,
        options=active_options,
        artifacts=artifacts,
    )
    manifest_path = _atomic_json(destination / "run_manifest.json", manifest)
    return LocalWorkflowResult(
        output_directory=destination,
        archive_ir_path=archive_ir_path,
        compiled_directory=compiled_directory,
        semantic_directory=semantic_directory,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _resolve_declared_artifact(root: Path, relative_path: str) -> Path:
    """Resolve one manifest path while rejecting absolute paths and traversal."""

    declared = PurePosixPath(relative_path)
    if declared.is_absolute() or ".." in declared.parts:
        raise LocalWorkflowError("The workflow manifest contains an unsafe artifact path.")
    candidate = root.joinpath(*declared.parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise LocalWorkflowError("The workflow manifest contains an unsafe artifact path.") from exc
    return candidate


def verify_local_workflow(output_directory: str | Path) -> WorkflowVerification:
    """Verify all files declared by a local workflow's SHA-256 manifest.

    Parameters
    ----------
    output_directory
        Workflow root containing ``run_manifest.json`` and its declared artifacts.

    Returns
    -------
    WorkflowVerification
        A content-free list of missing, size-mismatched, or digest-mismatched artifact paths.

    Raises
    ------
    LocalWorkflowError
        If the manifest is unreadable, invalid, or contains an unsafe path.
    """

    root = Path(output_directory).expanduser().resolve()
    manifest_path = root / "run_manifest.json"
    try:
        manifest = LocalWorkflowManifest.model_validate_json(manifest_path.read_bytes())
    except (OSError, ValueError, ValidationError) as exc:
        raise LocalWorkflowError("Could not read a valid local workflow manifest.") from exc

    problems: list[str] = []
    for artifact in manifest.artifacts:
        path = _resolve_declared_artifact(root, artifact.relative_path)
        if not path.is_file():
            problems.append(f"missing:{artifact.relative_path}")
            continue
        if path.stat().st_size != artifact.size_bytes:
            problems.append(f"size:{artifact.relative_path}")
            continue
        if _sha256(path) != artifact.sha256:
            problems.append(f"sha256:{artifact.relative_path}")
    return WorkflowVerification(
        manifest_path=manifest_path,
        checked_artifact_count=len(manifest.artifacts),
        valid=not problems,
        problems=tuple(problems),
    )
