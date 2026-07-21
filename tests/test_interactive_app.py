"""Synthetic tests for the interactive local application vertical slice."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import zipfile
from pathlib import Path
from types import ModuleType

from chatgpt_archive_compiler.app import create_app
from chatgpt_archive_compiler.app.downloads import make_download_payload
from chatgpt_archive_compiler.app.jobs import JobRegistry, JobStatus
from chatgpt_archive_compiler.app.service import compile_for_app, preflight_archive
from chatgpt_archive_compiler.app.state import SessionStore, append_correction, make_correction
from chatgpt_archive_compiler.app.visualizations import build_figures
from chatgpt_archive_compiler.ingest import ingest_export_zip
from chatgpt_archive_compiler.professional_safety import SafetyDecision, classify_archive
from chatgpt_archive_compiler.semantic.domain import EvidenceKind, EvidenceReference, Insight
from chatgpt_archive_compiler.semantic.evaluation import evaluate_contest_atlas
from chatgpt_archive_compiler.semantic.local_providers import (
    LocalHashingEmbeddingProvider,
    LocalHeuristicAnalysisProvider,
)
from chatgpt_archive_compiler.semantic.models import SemanticAtlasOptions
from chatgpt_archive_compiler.semantic.pipeline import build_semantic_atlas


def _generator_module() -> ModuleType:
    path = Path(__file__).parents[1] / "examples" / "make_contest_demo_export.py"
    spec = importlib.util.spec_from_file_location("contest_generator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_app_factory_state_isolation(tmp_path: Path) -> None:
    first = create_app(data_root=tmp_path / "one")
    second = create_app(data_root=tmp_path / "two")
    assert first is not second
    assert first.title == "Private Semantic Atlas"
    assert first.server.config["MAX_CONTENT_LENGTH"] == 512 * 1024 * 1024


def test_session_corrections_and_traversal(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    first, second = store.create(), store.create()
    assert first != second
    correction_path = store.path(first) / "corrections.json"
    append_correction(correction_path, make_correction("rename", "project-1", "Old", "New"))
    payload = json.loads(correction_path.read_text())
    assert payload["corrections"][0]["new_value"] == "New"
    try:
        store.path("../escape")
    except ValueError:
        pass
    else:
        raise AssertionError("Traversal should be rejected")


def test_jobs_are_background_and_fail_safely() -> None:
    registry = JobRegistry()
    successful = registry.submit(lambda stage: (stage("working"), 42)[1])
    failed = registry.submit(lambda _stage: (_ for _ in ()).throw(RuntimeError("private text")))
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        success_job, failed_job = registry.get(successful), registry.get(failed)
        if success_job and failed_job and success_job.finished_at and failed_job.finished_at:
            break
        time.sleep(0.01)
    assert success_job is not None and success_job.status is JobStatus.SUCCEEDED
    assert any(entry.status.value == "complete" for entry in success_job.log)
    assert failed_job is not None and failed_job.status is JobStatus.FAILED
    assert failed_job.error == "Compilation failed safely (RuntimeError)."
    assert "private text" not in failed_job.error


def test_synthetic_generator_preflight_safety_and_stability(tmp_path: Path) -> None:
    generator = _generator_module()
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    truth = tmp_path / "truth.json"
    fingerprint = generator.write_demo(first, truth, 20260721)
    assert generator.write_demo(second, tmp_path / "truth-two.json", 20260721) == fingerprint
    assert first.read_bytes() == second.read_bytes()
    summary = preflight_archive(first)
    assert summary.conversation_member_found
    archive = ingest_export_zip(first)
    assert 100 <= len(archive.conversations) <= 140
    assert sum(len(item.current_path_messages) for item in archive.conversations) >= 600
    decisions = classify_archive(tuple(archive.conversations))
    assert sum(item.decision is SafetyDecision.EXCLUDE for item in decisions) >= 4
    assert sum(item.decision is SafetyDecision.REVIEW for item in decisions) >= 1
    include_trap = next(
        decision
        for conversation, decision in zip(archive.conversations, decisions, strict=True)
        if conversation.title == "Medication table migration"
    )
    assert include_trap.decision is SafetyDecision.INCLUDE
    assert len(json.loads(truth.read_text())["projects"]) == 6


def test_evidence_schema_requires_grounding() -> None:
    evidence = EvidenceReference(
        conversation_id="synthetic-1",
        title="Synthetic evidence",
        excerpt="A bounded fictional excerpt.",
        relationship="Directly states the candidate milestone.",
        kind=EvidenceKind.HEURISTIC,
    )
    insight = Insight(
        insight_id="insight-1", statement="Candidate conclusion", evidence=(evidence,)
    )
    assert insight.schema_version == "1.0"


def test_typed_download_adapter(tmp_path: Path) -> None:
    artifact = tmp_path / "synthetic.txt"
    artifact.write_text("fictional", encoding="utf-8")
    payload = make_download_payload(artifact)
    assert payload["filename"] == "synthetic.txt"
    assert payload["base64"] is True


def test_anonymous_figures_suppress_titles() -> None:
    data = {
        "conversations": [
            {
                "key": "key-1",
                "title": "Private title",
                "date": "2026-01-01",
                "projects": ["private-repository"],
                "theme": "Theme",
                "role": "milestone",
                "message_count": 6,
                "evidence_count": 1,
            }
        ],
        "edges": [],
        "timelines": [],
    }
    figures = build_figures(data, anonymous=True)
    serialized = json.dumps(figures)
    assert "Private title" not in serialized
    assert "private-repository" not in serialized
    assert "Conversation 001" in serialized


def test_end_to_end_professional_bundle_has_no_excluded_text(tmp_path: Path) -> None:
    generator = _generator_module()
    source = tmp_path / "demo.zip"
    generator.write_demo(source, tmp_path / "truth.json", 20260721)
    stages: list[str] = []
    result = compile_for_app(
        source,
        tmp_path / "run",
        professional_safe=True,
        render_pdf=False,
        stage_callback=stages.append,
    )
    assert result.excluded_count >= 4 and result.review_count >= 1
    assert result.bundle_path.is_file() and result.manifest_path.is_file()
    with zipfile.ZipFile(result.bundle_path) as bundle:
        combined = b"\n".join(bundle.read(name) for name in bundle.namelist())
    assert b"fictional_demo_secret_12345" not in combined
    assert b"cannot pay rent" not in combined
    assert (
        b"app_manifest.json" in "\n".join(zipfile.ZipFile(result.bundle_path).namelist()).encode()
    )
    taxonomy = json.loads((result.output_directory / "semantic" / "taxonomy.json").read_text())
    known_keys = {
        row["conversation_key"]
        for row in json.loads(
            (result.output_directory / "semantic" / "review_queue.json").read_text()
        )["items"]
        if row.get("conversation_key")
    }
    assert all(isinstance(item, str) for item in known_keys)
    assert taxonomy["categories"]


def test_contest_semantic_acceptance(tmp_path: Path) -> None:
    generator = _generator_module()
    source, truth_path = tmp_path / "contest.zip", tmp_path / "truth.json"
    generator.write_demo(source, truth_path, 20260721)
    archive = ingest_export_zip(source)
    local = LocalHeuristicAnalysisProvider()
    result = build_semantic_atlas(
        archive,
        tmp_path / "semantic",
        embedding_provider=LocalHashingEmbeddingProvider(),
        analysis_provider=local,
        interpretation_provider=local,
        options=SemanticAtlasOptions(max_leaf_categories=12),
    )
    validation = evaluate_contest_atlas(
        result.atlas,
        json.loads(truth_path.read_text()),
    )
    assert validation.passed, validation.as_dict()
