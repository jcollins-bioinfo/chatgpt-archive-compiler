"""End-to-end tests for reproducible local workflow orchestration."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from chatgpt_archive_compiler import __version__
from chatgpt_archive_compiler.workflows import run_local_workflow, verify_local_workflow


def _synthetic_export(path: Path) -> Path:
    """Write one deterministic synthetic conversation export."""

    payload = [
        {
            "id": "synthetic-conversation-001",
            "title": "Reproducible software design",
            "create_time": 1_735_689_600,
            "update_time": 1_735_689_700,
            "current_node": "assistant-001",
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
                    "children": ["assistant-001"],
                    "message": {
                        "id": "message-user-001",
                        "author": {"role": "user"},
                        "create_time": 1_735_689_600,
                        "content": {
                            "content_type": "text",
                            "parts": ["How should this software record provenance?"],
                        },
                        "metadata": {},
                    },
                },
                "assistant-001": {
                    "id": "assistant-001",
                    "parent": "user-001",
                    "children": [],
                    "message": {
                        "id": "message-assistant-001",
                        "author": {"role": "assistant"},
                        "create_time": 1_735_689_700,
                        "content": {
                            "content_type": "text",
                            "parts": [
                                "Record versions, options, source hashes, and artifact hashes."
                            ],
                        },
                        "metadata": {"model_slug": "synthetic-model"},
                    },
                },
            },
        }
    ]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("conversations.json", json.dumps(payload, allow_nan=False))
    return path


def test_local_workflow_is_manifested_and_verifiable(tmp_path: Path) -> None:
    """A clean local run records versioned hashes for all durable products."""

    source = _synthetic_export(tmp_path / "synthetic.zip")
    result = run_local_workflow(source, tmp_path / "output")

    assert result.manifest.package_version == __version__
    assert result.manifest.runtime.python_version
    assert result.manifest.runtime.dependency_versions["pydantic"]
    assert result.manifest.source_archive_sha256
    assert len(result.manifest.artifacts) >= 10
    verification = verify_local_workflow(result.output_directory)
    assert verification.valid
    assert verification.checked_artifact_count == len(result.manifest.artifacts)


def test_local_workflow_verification_detects_tampering(tmp_path: Path) -> None:
    """Verification reports a content-free digest failure after artifact mutation."""

    source = _synthetic_export(tmp_path / "synthetic.zip")
    result = run_local_workflow(source, tmp_path / "output")
    artifact = result.manifest.artifacts[0]
    changed = result.output_directory / artifact.relative_path
    changed.write_bytes(changed.read_bytes() + b"\n")

    verification = verify_local_workflow(result.output_directory)
    assert not verification.valid
    assert verification.problems == (f"size:{artifact.relative_path}",)
