"""Framework-neutral application service used by Dash and end-to-end tests."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from chatgpt_archive_compiler.compiler import CompileOptions, compile_archive
from chatgpt_archive_compiler.ingest import ingest_export_zip, inspect_zip
from chatgpt_archive_compiler.professional_safety import (
    SafetyDecision,
    classify_archive,
)
from chatgpt_archive_compiler.semantic import (
    LocalHashingEmbeddingProvider,
    LocalHeuristicAnalysisProvider,
    SemanticAtlasOptions,
    build_semantic_atlas,
)
from chatgpt_archive_compiler.serialization import write_archive_ir
from chatgpt_archive_compiler.version import __version__

StageCallback = Callable[[str], None]
STAGES = (
    "Validating archive",
    "Normalizing conversation graph",
    "Applying Professional-safe boundary",
    "Compiling chronological archive",
    "Extracting semantic representations",
    "Constructing similarity graph",
    "Organizing topics and projects",
    "Synthesizing timelines and insights",
    "Building review queue",
    "Rendering outputs",
    "Verifying manifests",
)


@dataclass(frozen=True)
class PreflightSummary:
    """Content-minimal preflight data safe for application display."""

    filename: str
    size_bytes: int
    member_count: int
    conversation_member_found: bool
    warning_codes: tuple[str, ...]


@dataclass(frozen=True)
class AppCompilationResult:
    """Completed app products and safe overview metadata."""

    output_directory: Path
    bundle_path: Path
    manifest_path: Path
    category_count: int
    project_count: int
    included_count: int
    excluded_count: int
    review_count: int


def preflight_archive(path: Path) -> PreflightSummary:
    """Inspect ZIP metadata without extraction or source-derived exceptions."""

    manifest = inspect_zip(path)
    return PreflightSummary(
        filename=path.name,
        size_bytes=path.stat().st_size,
        member_count=len(manifest.files),
        conversation_member_found=any(
            item.path.name.casefold().startswith("conversations") for item in manifest.files
        ),
        warning_codes=tuple(item.code for item in manifest.warnings),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compile_for_app(
    archive_path: Path,
    output_directory: Path,
    *,
    professional_safe: bool = True,
    render_pdf: bool = True,
    stage_callback: StageCallback | None = None,
) -> AppCompilationResult:
    """Run the local vertical slice with filtering before every derived output."""

    notify = stage_callback or (lambda _stage: None)
    notify(STAGES[0])
    preflight_archive(archive_path)
    notify(STAGES[1])
    archive = ingest_export_zip(archive_path, compute_archive_hash=True)
    decisions = classify_archive(tuple(archive.conversations))
    notify(STAGES[2])
    allowed_indexes = {
        index
        for index, decision in enumerate(decisions)
        if not professional_safe or decision.decision is SafetyDecision.INCLUDE
    }
    filtered = archive.model_copy(
        update={
            "conversations": [
                conversation
                for index, conversation in enumerate(archive.conversations)
                if index in allowed_indexes
            ],
            "metadata": {
                **archive.metadata,
                "artifact_scope": "professional-safe" if professional_safe else "private-full",
            },
        }
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    write_archive_ir(filtered, output_directory / "archive.ir.json")
    safety_path = output_directory / "professional_safety_manifest.json"
    safety_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "artifact_scope": "professional-safe" if professional_safe else "private-full",
                "review_items_excluded_by_default": professional_safe,
                "decisions": [item.model_dump(mode="json") for item in decisions],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_directory / "user_corrections.json").write_text(
        json.dumps({"schema_version": "1.0", "corrections": []}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    notify(STAGES[3])
    compile_archive(
        filtered,
        output_directory / "compiled",
        options=CompileOptions(
            render_pdf=render_pdf,
            title=(
                "Professional-safe Conversation Archive"
                if professional_safe
                else "Private Full Conversation Archive"
            ),
        ),
    )
    notify(STAGES[4])
    local = LocalHeuristicAnalysisProvider()

    def semantic_progress(stage: str, _completed: int, _total: int, _cached: int) -> None:
        mapping = {
            "embedding": STAGES[4],
            "graph": STAGES[5],
            "taxonomy": STAGES[6],
            "synthesis": STAGES[7],
            "review": STAGES[8],
        }
        notify(mapping.get(stage, STAGES[6]))

    semantic = build_semantic_atlas(
        filtered,
        output_directory / "semantic",
        embedding_provider=LocalHashingEmbeddingProvider(),
        analysis_provider=local,
        interpretation_provider=local,
        options=SemanticAtlasOptions(),
        progress_callback=semantic_progress,
    )
    notify(STAGES[9])
    manifest_path = output_directory / "app_manifest.json"
    files = sorted(
        path for path in output_directory.rglob("*") if path.is_file() and path != manifest_path
    )
    manifest = {
        "schema_version": "1.0",
        "package_version": __version__,
        "scope": "professional-safe" if professional_safe else "private-full",
        "source_fingerprint": archive.source_manifest.archive_sha256,
        "artifacts": [
            {
                "path": path.relative_to(output_directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    notify(STAGES[10])
    bundle_path = output_directory.parent / f"{output_directory.name}-verified.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in [*files, manifest_path]:
            bundle.write(path, path.relative_to(output_directory).as_posix())
    counts = {
        value: sum(item.decision.value == value for item in decisions)
        for value in ("include", "exclude", "review")
    }
    return AppCompilationResult(
        output_directory=output_directory,
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        category_count=len(semantic.atlas.taxonomy.categories),
        project_count=len(semantic.atlas.project_timelines),
        included_count=counts["include"],
        excluded_count=counts["exclude"],
        review_count=counts["review"],
    )
