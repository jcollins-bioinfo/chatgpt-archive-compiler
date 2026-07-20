"""Synthetic-only tests for semantic-atlas book rendering."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from chatgpt_archive_compiler.ingest import ingest_export_zip
from chatgpt_archive_compiler.semantic.models import (
    ArchiveSynthesis,
    Category,
    CategoryAssignment,
    CategoryProfile,
    ConversationAnalysis,
    ConversationRepresentation,
    ProjectTimeline,
    ReviewQueue,
    SemanticAtlas,
    Taxonomy,
    TimelineEvent,
)
from chatgpt_archive_compiler.semantic_book import (
    SemanticBookOptions,
    compile_semantic_book,
)


def _synthetic_atlas() -> dict[str, object]:
    """Return a nested, multi-label semantic atlas for the synthetic fixture."""

    return {
        "taxonomy": {
            "categories": [
                {
                    "category_id": "research",
                    "name": "Scientific and Technical Work",
                    "summary": "A broad domain connecting **research** and implementation.",
                    "themes": ["evidence", "reproducibility"],
                    "children": [
                        {
                            "category_id": "semantic-systems",
                            "name": "Semantic Systems",
                            "summary": "Methods for finding structure across many conversations.",
                            "conversation_ids": ["synthetic-conversation-001"],
                        }
                    ],
                },
                {
                    "category_id": "creative-work",
                    "name": "Creative Work",
                    "summary": "Related work outside the primary technical placement.",
                    "related_category_ids": ["semantic-systems"],
                },
            ]
        },
        "conversation_catalog": [
            {
                "conversation_id": "synthetic-conversation-001",
                "synopsis": "The user tests a semantic archive workflow.\u2029It remains private.",
                "primary_category_id": "semantic-systems",
                "category_ids": ["semantic-systems", "creative-work"],
                "projects": ["Archive compiler"],
                "themes": ["classification"],
                "entities": ["HTML", "PDF"],
                "status": "implementation",
                "related_conversations": [
                    {
                        "target_conversation_id": "synthetic-conversation-001",
                        "reason": "self reference is ignored",
                    }
                ],
                "confidence": 0.93,
            }
        ],
        "cross_archive_synthesis": {
            "overview": "The archive develops from questions into durable projects.",
            "sections": [
                {
                    "title": "Recurring pattern",
                    "body": "Ideas are repeatedly refined through implementation.",
                }
            ],
        },
    }


def test_semantic_book_renders_hierarchy_synopsis_and_cross_references(
    branched_payload: list[dict[str, Any]], write_payload_zip, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    """The HTML edition is organized by taxonomy and preserves secondary membership."""

    archive = ingest_export_zip(write_payload_zip(branched_payload))
    result = compile_semantic_book(
        archive,
        _synthetic_atlas(),
        tmp_path / "semantic-book",
        options=SemanticBookOptions(author="Synthetic Author"),
    )

    document = result.html_path.read_text(encoding="utf-8")
    assert result.category_count == 3
    assert result.conversation_count == 1
    assert result.unclassified_conversation_count == 0
    assert "Scientific and Technical Work" in document
    assert "Semantic Systems" in document
    assert "Synthetic branched conversation" in document
    assert "The user tests a semantic archive workflow." in document
    assert "Also appears in" in document
    assert "Creative Work" in document
    assert "Synthetic question" not in document
    assert "\u2029" not in document
    assert "target-counter(attr(href), page)" in document
    assert "string(chapter-title)" in document
    assert "http://" not in document
    assert "https://" not in document
    assert result.manifest_path.is_file()
    soup = BeautifulSoup(document, "html.parser")
    parent_title = soup.find("span", class_="toc-title", string="Scientific and Technical Work")
    assert parent_title is not None
    parent_link = parent_title.find_parent("a")
    assert parent_link is not None
    assert parent_link.find("span", class_="toc-count").get_text(strip=True) == "1 chat"
    parent_opener = soup.select_one(str(parent_link["href"]))
    assert parent_opener is not None
    assert "1 conversation" in parent_opener.get_text(" ", strip=True)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["html_sha256"] == result.html_sha256


def test_semantic_book_can_include_only_current_path_transcript(
    branched_payload: list[dict[str, Any]], write_payload_zip, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    """Transcript mode includes current user/assistant messages but no alternate branch."""

    payload = copy.deepcopy(branched_payload)
    payload[0]["mapping"]["user-001"]["message"]["content"]["parts"] = [
        "Question before separator \u2028 after separator"
    ]
    archive = ingest_export_zip(write_payload_zip(payload))
    result = compile_semantic_book(
        archive,
        _synthetic_atlas(),
        tmp_path / "with-transcript",
        options=SemanticBookOptions(include_transcripts=True),
    )

    document = result.html_path.read_text(encoding="utf-8")
    assert "Question before separator" in document
    assert "Current answer" in document
    assert "Alternate answer" not in document
    assert "\u2028" not in document


def test_semantic_book_keeps_unclassified_conversations_visible(
    branched_payload: list[dict[str, Any]], write_payload_zip, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    """Missing assignments produce a visible review category rather than data loss."""

    archive = ingest_export_zip(write_payload_zip(branched_payload))
    result = compile_semantic_book(
        archive,
        {"categories": [], "conversation_catalog": []},
        tmp_path / "unclassified",
    )

    document = result.html_path.read_text(encoding="utf-8")
    assert result.category_count == 1
    assert result.unclassified_conversation_count == 1
    assert "Unclassified conversations" in document
    assert "No semantic synopsis was available" in document


def test_semantic_book_bounds_expanded_profiles_but_keeps_complete_directory(
    branched_payload: list[dict[str, Any]], write_payload_zip, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    """A bounded print edition lists all conversations while expanding only the best profile."""

    payload: list[dict[str, Any]] = []
    for index, title in enumerate(
        ("Alpha conversation", "Beta conversation", "Gamma conversation")
    ):
        record = copy.deepcopy(branched_payload[0])
        record["id"] = f"conversation-{index}"
        record["title"] = title
        record["create_time"] += index * 100
        payload.append(record)
    archive = ingest_export_zip(write_payload_zip(payload))
    atlas: dict[str, object] = {
        "categories": [
            {
                "category_id": "bounded",
                "name": "Bounded category",
                "conversation_ids": [f"conversation-{index}" for index in range(3)],
            }
        ],
        "conversation_catalog": [
            {
                "conversation_id": f"conversation-{index}",
                "synopsis": f"Unique synopsis number {index}.",
                "primary_category_id": "bounded",
                "confidence": confidence,
            }
            for index, confidence in enumerate((0.2, 0.99, 0.5))
        ],
    }
    result = compile_semantic_book(
        archive,
        atlas,
        tmp_path / "bounded",
        options=SemanticBookOptions(max_synopsis_entries_per_category=1),
    )

    document = result.html_path.read_text(encoding="utf-8")
    assert result.conversation_count == 3
    assert result.expanded_conversation_count == 1
    assert all(
        title in document
        for title in ("Alpha conversation", "Beta conversation", "Gamma conversation")
    )
    assert "Unique synopsis number 1." in document
    assert "Unique synopsis number 0." not in document
    assert "Unique synopsis number 2." not in document


def test_semantic_book_applies_a_global_profile_budget(
    branched_payload: list[dict[str, Any]], write_payload_zip, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    """A global budget bounds expansion even when every category is below its local cap."""

    payload: list[dict[str, Any]] = []
    categories: list[dict[str, object]] = []
    catalog: list[dict[str, object]] = []
    for index in range(4):
        record = copy.deepcopy(branched_payload[0])
        record["id"] = f"budget-conversation-{index}"
        record["title"] = f"Budget conversation {index}"
        record["create_time"] += index * 100
        payload.append(record)
        category_id = f"budget-category-{index}"
        categories.append(
            {
                "category_id": category_id,
                "name": f"Budget category {index}",
                "conversation_ids": [record["id"]],
            }
        )
        catalog.append(
            {
                "conversation_id": record["id"],
                "synopsis": f"Budget synopsis {index}.",
                "primary_category_id": category_id,
                "confidence": 0.9,
            }
        )
    archive = ingest_export_zip(write_payload_zip(payload))
    result = compile_semantic_book(
        archive,
        {"categories": categories, "conversation_catalog": catalog},
        tmp_path / "global-budget",
        options=SemanticBookOptions(
            max_synopsis_entries_per_category=12,
            max_expanded_conversations=2,
        ),
    )

    document = result.html_path.read_text(encoding="utf-8")
    assert result.conversation_count == 4
    assert result.expanded_conversation_count == 2
    assert sum(f"Budget synopsis {index}." in document for index in range(4)) == 2
    assert all(f"Budget conversation {index}" in document for index in range(4))


def test_semantic_book_pdf_path_uses_optional_renderer(
    branched_payload: list[dict[str, Any]],
    write_payload_zip,  # type: ignore[no-untyped-def]
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    """PDF mode records a mocked local WeasyPrint artifact and checksum."""

    class FakeHTML:
        """Minimal stand-in for the optional WeasyPrint HTML class."""

        def __init__(self, **kwargs: object) -> None:
            assert "filename" in kwargs
            assert "url_fetcher" in kwargs

        def write_pdf(self, destination: str) -> None:
            Path(destination).write_bytes(b"%PDF-synthetic")

    fake_module = types.ModuleType("weasyprint")
    fake_module.HTML = FakeHTML  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "weasyprint", fake_module)
    archive = ingest_export_zip(write_payload_zip(branched_payload))
    result = compile_semantic_book(
        archive,
        _synthetic_atlas(),
        tmp_path / "with-pdf",
        options=SemanticBookOptions(render_pdf=True),
    )

    assert result.pdf_path is not None
    assert result.pdf_path.read_bytes() == b"%PDF-synthetic"
    assert result.pdf_sha256 is not None


def test_semantic_book_accepts_typed_atlas_and_joins_opaque_key(
    branched_payload: list[dict[str, Any]], write_payload_zip, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    """A real SemanticAtlas joins its versioned opaque key back to Archive IR."""

    archive = ingest_export_zip(write_payload_zip(branched_payload))
    conversation = archive.conversations[0]
    preimage = "\0".join(
        (
            "semantic-conversation-v1",
            "0",
            str(conversation.source_path),
            conversation.conversation_id or "",
        )
    )
    conversation_key = hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:24]
    atlas = SemanticAtlas(
        representations=(
            ConversationRepresentation(
                conversation_key=conversation_key,
                source_index=0,
                source_conversation_id=conversation.conversation_id,
                title=conversation.title or "Untitled conversation",
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
                text="Synthetic representation",
                message_count=2,
                character_count=24,
                original_character_count=24,
                text_sha256="0" * 64,
            ),
        ),
        analyses=(
            ConversationAnalysis(
                conversation_key=conversation_key,
                synopsis="A typed semantic analysis joined through an opaque key.",
                primary_subject="Archive design",
                secondary_subjects=("Typography",),
                projects=("Semantic atlas",),
                conversation_type="implementation",
                entities=("WeasyPrint",),
                goals=("Create a navigable book",),
                decisions=("Use a category-first organization",),
                unresolved_questions=("How should later editions be revised?",),
                recurring_themes=("Durable knowledge",),
                temporal_role="refinement",
                confidence=0.96,
            ),
        ),
        embeddings=(),
        graph_edges=(),
        taxonomy=Taxonomy(
            categories=(
                Category(
                    category_id="archive-design",
                    name="Archive Design",
                    description="Classification, navigation, and publication design.",
                    level=0,
                    conversation_keys=(conversation_key,),
                    defining_concepts=("semantic structure",),
                    confidence=0.95,
                ),
            ),
            assignments=(
                CategoryAssignment(
                    conversation_key=conversation_key,
                    primary_category_id="archive-design",
                    confidence=0.94,
                ),
            ),
        ),
        category_profiles=(
            CategoryProfile(
                category_id="archive-design",
                overview="The work moves from raw transcripts toward a durable knowledge atlas.",
                characteristic_questions=("How can related conversations be rediscovered?",),
                major_conclusions=("Chronology alone is not an adequate organizing principle.",),
                unresolved_threads=("Human review of ambiguous assignments",),
                temporal_evolution="Early ingestion work gives way to semantic organization.",
                representative_conversation_keys=(conversation_key,),
            ),
        ),
        project_timelines=(
            ProjectTimeline(
                project_name="Semantic atlas",
                overview="A project to transform an export into navigable knowledge.",
                events=(
                    TimelineEvent(
                        label="Category-first edition",
                        summary="The archive acquires a semantic hierarchy.",
                        conversation_keys=(conversation_key,),
                    ),
                ),
                current_state="The first analytical edition is ready for review.",
                unresolved_work=("Revise low-confidence assignments",),
            ),
        ),
        synthesis=ArchiveSynthesis(
            executive_summary="The archive records a progression from questions to systems.",
            dominant_domains=("Scientific and technical work",),
            cross_domain_connections=("Publication design supports knowledge retrieval.",),
            recurring_patterns=("Questions become durable projects.",),
            opportunities=("Build future revised editions.",),
        ),
        review_queue=ReviewQueue(items=(), counts_by_code={}, counts_by_priority={}),
    )

    result = compile_semantic_book(archive, atlas, tmp_path / "typed-atlas")
    document = result.html_path.read_text(encoding="utf-8")

    assert result.unclassified_conversation_count == 0
    assert "A typed semantic analysis joined through an opaque key." in document
    assert "The archive records a progression from questions to systems." in document
    assert "Project timelines" in document
    assert "Category-first edition" in document
    assert "Chronology alone is not an adequate organizing principle." in document
