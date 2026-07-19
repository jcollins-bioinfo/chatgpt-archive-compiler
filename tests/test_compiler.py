"""Synthetic-only analysis and document compilation tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from chatgpt_archive_compiler.compiler import (
    CompileOptions,
    RedactionRule,
    VolumeMode,
    analyze_archive,
    compile_archive,
)
from chatgpt_archive_compiler.ingest import ingest_export_zip


def test_analysis_reports_structural_counts_only(
    branched_payload: list[dict[str, Any]], write_payload_zip  # type: ignore[no-untyped-def]
) -> None:
    """Corpus analysis separates graph and current-path counts."""

    archive = ingest_export_zip(write_payload_zip(branched_payload))
    analysis = analyze_archive(archive)

    assert analysis.conversation_count == 1
    assert analysis.graph_node_count == 4
    assert analysis.graph_message_count == 3
    assert analysis.current_path_message_count == 2
    assert analysis.alternate_path_message_count == 1
    assert analysis.messages_by_role == {"assistant": 2, "user": 1}
    encoded = json.dumps(analysis.model_dump(mode="json"))
    assert "Synthetic question" not in encoded
    assert "Synthetic branched conversation" not in encoded


def test_compile_archive_writes_year_volume_and_manifest(
    branched_payload: list[dict[str, Any]], write_payload_zip, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    """A current-path HTML volume, index, analysis, and integrity manifest are produced."""

    archive = ingest_export_zip(write_payload_zip(branched_payload))
    result = compile_archive(
        archive,
        tmp_path / "compiled",
        options=CompileOptions(volume_mode=VolumeMode.YEAR),
    )

    assert result.selected_conversation_count == 1
    assert len(result.volumes) == 1
    assert result.volumes[0].label == "2025"
    assert result.volumes[0].conversation_count == 1
    assert result.volumes[0].message_count == 2
    assert result.volumes[0].pdf_path is None
    html_text = result.volumes[0].html_path.read_text(encoding="utf-8")
    assert "Synthetic question" in html_text
    assert "Current answer" in html_text
    assert "Alternate answer" not in html_text
    assert result.index_path.is_file()
    assert result.analysis_path.is_file()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["volumes"][0]["html_sha256"] == result.volumes[0].html_sha256


def test_compile_archive_defaults_to_monthly_volumes(
    branched_payload: list[dict[str, Any]], write_payload_zip, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    """The safer default bounds each PDF layout job to one calendar month."""

    archive = ingest_export_zip(write_payload_zip(branched_payload))
    result = compile_archive(archive, tmp_path / "monthly")

    assert [volume.label for volume in result.volumes] == ["2025-01"]


def test_reasoning_is_omitted_by_default_and_can_be_included(
    branched_payload: list[dict[str, Any]], write_payload_zip, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    """Reasoning UI records never silently become ordinary assistant answer text."""

    payload = copy.deepcopy(branched_payload)
    reasoning_node = payload[0]["mapping"]["assistant-current"]
    reasoning_node["message"]["content"] = {
        "content_type": "reasoning_recap",
        "content": "Synthetic reasoning recap",
    }
    archive = ingest_export_zip(write_payload_zip(payload))

    omitted = compile_archive(
        archive,
        tmp_path / "omitted",
        options=CompileOptions(volume_mode=VolumeMode.SINGLE),
    )
    included = compile_archive(
        archive,
        tmp_path / "included",
        options=CompileOptions(
            volume_mode=VolumeMode.SINGLE,
            include_reasoning_summaries=True,
        ),
    )

    assert "Synthetic reasoning recap" not in omitted.volumes[0].html_path.read_text(
        encoding="utf-8"
    )
    included_text = included.volumes[0].html_path.read_text(encoding="utf-8")
    assert "Reasoning summary" in included_text
    assert "Synthetic reasoning recap" in included_text


def test_explicit_redaction_rules_apply_to_titles_and_messages(
    branched_payload: list[dict[str, Any]], write_payload_zip, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    """Caller-supplied patterns are applied before private text enters output artifacts."""

    archive = ingest_export_zip(write_payload_zip(branched_payload))
    result = compile_archive(
        archive,
        tmp_path / "redacted",
        options=CompileOptions(
            volume_mode=VolumeMode.SINGLE,
            redaction_rules=(RedactionRule(pattern="Synthetic", replacement="[PRIVATE]"),),
        ),
    )
    html_text = result.volumes[0].html_path.read_text(encoding="utf-8")

    assert "Synthetic" not in html_text
    assert "[PRIVATE]" in html_text


def test_unicode_line_and_paragraph_separators_are_normalized(
    branched_payload: list[dict[str, Any]], write_payload_zip, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    """Unicode separators that trigger PDF-layout assertions become ordinary line breaks."""

    payload = copy.deepcopy(branched_payload)
    payload[0]["mapping"]["user-001"]["message"]["content"]["parts"] = [
        "First line \u2028 second line \u2029 next paragraph"
    ]
    archive = ingest_export_zip(write_payload_zip(payload))
    result = compile_archive(
        archive,
        tmp_path / "unicode-separators",
        options=CompileOptions(volume_mode=VolumeMode.SINGLE),
    )
    html_text = result.volumes[0].html_path.read_text(encoding="utf-8")

    assert "\u2028" not in html_text
    assert "\u2029" not in html_text
    assert "First line" in html_text
    assert "next paragraph" in html_text
