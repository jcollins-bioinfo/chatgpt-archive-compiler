"""Synthetic-only tests for semantic representations, graph logic, caching, and export."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pytest

import chatgpt_archive_compiler.semantic.graph as semantic_graph
from chatgpt_archive_compiler.models import (
    Archive,
    ContentBlock,
    ContentBlockType,
    Conversation,
    Message,
    MessageNode,
    Role,
    SourceManifest,
)
from chatgpt_archive_compiler.semantic import (
    LocalHashingEmbeddingProvider,
    LocalHeuristicAnalysisProvider,
    SemanticAtlasOptions,
    build_semantic_atlas,
    build_similarity_graph,
    conversation_key_for,
    discover_communities,
    estimate_semantic_run,
    prepare_conversation_representations,
)
from chatgpt_archive_compiler.semantic.export import _write_synthesis_markdown
from chatgpt_archive_compiler.semantic.models import (
    ArchiveSynthesis,
    CategoryDraft,
    CategoryInterpretation,
    ConversationAnalysis,
    EmbeddingRecord,
    TaxonomyInterpretation,
)
from chatgpt_archive_compiler.semantic.pipeline import _build_taxonomy


def _synthetic_archive() -> Archive:
    """Return a private-free two-conversation Archive IR with graph edge cases."""

    first = Conversation(
        conversation_id="conversation-one",
        title="Genomics notebook design",
        source_path=PurePosixPath("conversations.json"),
        current_node_id="assistant-one",
        current_path_node_ids=["root", "user-one", "system-one", "assistant-one"],
        nodes=[
            MessageNode(node_id="root"),
            MessageNode(
                node_id="user-one",
                parent_node_id="root",
                message=Message(
                    role=Role.USER,
                    content=[
                        ContentBlock(
                            type=ContentBlockType.TEXT,
                            text="Design a reproducible genomics notebook.",
                        )
                    ],
                ),
                is_on_current_path=True,
            ),
            MessageNode(
                node_id="system-one",
                parent_node_id="user-one",
                message=Message(
                    role=Role.SYSTEM,
                    content=[
                        ContentBlock(type=ContentBlockType.TEXT, text="Hidden system material")
                    ],
                ),
                is_on_current_path=True,
            ),
            MessageNode(
                node_id="assistant-one",
                parent_node_id="system-one",
                message=Message(
                    role=Role.ASSISTANT,
                    content=[
                        ContentBlock(
                            type=ContentBlockType.THINKING_TRACE,
                            text="Hidden reasoning material",
                        ),
                        ContentBlock(
                            type=ContentBlockType.MARKDOWN,
                            text="Use a versioned workflow and synthetic tests.",
                        ),
                        ContentBlock(
                            type=ContentBlockType.FILE_REFERENCE,
                            reference="private/source-name.csv",
                        ),
                    ],
                ),
                is_on_current_path=True,
            ),
            MessageNode(
                node_id="alternate-one",
                parent_node_id="user-one",
                message=Message(
                    role=Role.ASSISTANT,
                    content=[
                        ContentBlock(type=ContentBlockType.TEXT, text="Discarded branch material")
                    ],
                ),
            ),
        ],
    )
    second = Conversation(
        conversation_id="conversation-two",
        title="Genome workflow testing",
        source_path=PurePosixPath("conversations.json"),
        current_node_id="assistant-two",
        current_path_node_ids=["user-two", "assistant-two"],
        nodes=[
            MessageNode(
                node_id="user-two",
                message=Message(
                    role=Role.USER,
                    content=[
                        ContentBlock(
                            type=ContentBlockType.TEXT,
                            text="How should the genomics pipeline be tested?",
                        )
                    ],
                ),
                is_on_current_path=True,
            ),
            MessageNode(
                node_id="assistant-two",
                parent_node_id="user-two",
                message=Message(
                    role=Role.ASSISTANT,
                    content=[
                        ContentBlock(
                            type=ContentBlockType.TEXT,
                            text="Use fixtures, checksums, and deterministic assertions.",
                        )
                    ],
                ),
                is_on_current_path=True,
            ),
        ],
    )
    return Archive(
        source_manifest=SourceManifest(archive_name="synthetic.zip", archive_size_bytes=100),
        conversations=[first, second],
    )


def test_representations_use_only_visible_current_path_content() -> None:
    """Default provider material omits reasoning, system records, and alternate branches."""

    archive = _synthetic_archive()
    representations = prepare_conversation_representations(archive)
    first = representations[0]

    assert "Design a reproducible" in first.text
    assert "versioned workflow" in first.text
    assert "Hidden reasoning" not in first.text
    assert "Hidden system" not in first.text
    assert "Discarded branch" not in first.text
    assert "private/source-name.csv" not in first.text
    assert "[file reference omitted]" in first.text
    assert first.message_count == 2
    assert first.conversation_key == conversation_key_for(archive.conversations[0], 0)
    assert prepare_conversation_representations(archive) == representations


def test_estimate_contains_counts_but_no_source_text() -> None:
    """The preflight estimate is useful while remaining safe for logs and display."""

    estimate = estimate_semantic_run(_synthetic_archive())
    encoded = json.dumps(estimate.model_dump(mode="json"))

    assert estimate.conversation_count == 2
    assert estimate.message_count == 4
    assert estimate.approximate_input_tokens > 0
    assert "Genomics notebook design" not in encoded
    assert "reproducible" not in encoded


def test_synthesis_markdown_escapes_active_model_markup(tmp_path: Path) -> None:
    """Generated Markdown cannot activate model-supplied remote images or raw HTML."""

    destination = _write_synthesis_markdown(
        tmp_path / "synthesis.md",
        ArchiveSynthesis(
            executive_summary='![remote](https://example.invalid/x.png) <img src="remote">',
            opportunities=("[external](https://example.invalid)",),
        ),
    )

    rendered = destination.read_text(encoding="utf-8")
    assert "![remote]" not in rendered
    assert " <img" not in rendered
    assert "\\!\\[remote\\]" in rendered
    assert "\\<img" in rendered


def test_similarity_graph_and_communities_are_deterministic() -> None:
    """Close synthetic vectors cluster together and distant vectors remain separated."""

    embeddings = (
        EmbeddingRecord(conversation_key="a", vector=(1.0, 0.0, 0.0)),
        EmbeddingRecord(conversation_key="b", vector=(0.99, 0.01, 0.0)),
        EmbeddingRecord(conversation_key="c", vector=(0.0, 1.0, 0.0)),
        EmbeddingRecord(conversation_key="d", vector=(0.0, 0.99, 0.01)),
    )
    options = SemanticAtlasOptions(min_similarity=0.9, max_neighbors=2)
    edges = build_similarity_graph(embeddings, options=options)
    communities = discover_communities(tuple("abcd"), edges)

    assert {(edge.source_key, edge.target_key) for edge in edges} == {("a", "b"), ("c", "d")}
    assert communities["a"] == communities["b"]
    assert communities["c"] == communities["d"]
    assert communities["a"] != communities["c"]
    assert discover_communities(tuple("abcd"), edges) == communities


def test_similarity_graph_uses_exact_sklearn_neighbors_when_installed(monkeypatch) -> None:
    """The Colab-scale graph path uses sklearn rather than the approximate fallback."""

    pytest.importorskip("sklearn")
    embeddings = (
        EmbeddingRecord(conversation_key="a", vector=(1.0, 0.0, 0.0)),
        EmbeddingRecord(conversation_key="b", vector=(0.99, 0.01, 0.0)),
        EmbeddingRecord(conversation_key="c", vector=(0.0, 1.0, 0.0)),
        EmbeddingRecord(conversation_key="d", vector=(0.01, 0.99, 0.0)),
        EmbeddingRecord(conversation_key="e", vector=(0.0, 0.0, 1.0)),
        EmbeddingRecord(conversation_key="f", vector=(0.0, 0.01, 0.99)),
    )

    def fallback_must_not_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("approximate candidate fallback was used")

    monkeypatch.setattr(semantic_graph, "_candidate_pairs", fallback_must_not_run)
    edges = build_similarity_graph(
        embeddings,
        options=SemanticAtlasOptions(
            graph_brute_force_threshold=4,
            max_neighbors=1,
            min_similarity=0.9,
        ),
    )

    assert {(edge.source_key, edge.target_key) for edge in edges} == {
        ("a", "b"),
        ("c", "d"),
        ("e", "f"),
    }


def test_taxonomy_canonicalizes_case_variant_parent_names() -> None:
    """Provider capitalization variants cannot create duplicate broad-domain identifiers."""

    drafts = (
        CategoryDraft(
            category_id="category-0001",
            conversation_keys=("conversation-a",),
            representative_conversation_keys=("conversation-a",),
            candidate_labels=("Genomics",),
        ),
        CategoryDraft(
            category_id="category-0002",
            conversation_keys=("conversation-b",),
            representative_conversation_keys=("conversation-b",),
            candidate_labels=("Proteomics",),
        ),
    )
    analyses = tuple(
        ConversationAnalysis(
            conversation_key=key,
            synopsis="Synthetic analysis.",
            primary_subject=subject,
            conversation_type="research",
            confidence=0.9,
        )
        for key, subject in (
            ("conversation-a", "Genomics"),
            ("conversation-b", "Proteomics"),
        )
    )
    interpretation = TaxonomyInterpretation(
        categories=(
            CategoryInterpretation(
                category_id="category-0001",
                name="Genomics",
                description="Synthetic genomics category.",
                parent_name="Science",
                confidence=0.9,
            ),
            CategoryInterpretation(
                category_id="category-0002",
                name="Proteomics",
                description="Synthetic proteomics category.",
                parent_name=" science ",
                confidence=0.9,
            ),
        )
    )

    taxonomy = _build_taxonomy(drafts, interpretation, analyses, ())

    parents = [category for category in taxonomy.categories if category.level == 0]
    assert len(parents) == 1
    assert parents[0].child_ids == ("category-0001", "category-0002")
    assert len({category.category_id for category in taxonomy.categories}) == 3


def test_category_limit_has_sensible_validated_bounds() -> None:
    """The default is bounded while callers can intentionally request one leaf."""

    assert SemanticAtlasOptions().max_leaf_categories == 64
    assert SemanticAtlasOptions(max_leaf_categories=1).max_leaf_categories == 1
    with pytest.raises(ValueError):
        SemanticAtlasOptions(max_leaf_categories=0)
    with pytest.raises(ValueError):
        SemanticAtlasOptions(max_leaf_categories=513)


def test_community_consolidation_caps_categories_deterministically(monkeypatch) -> None:
    """Centroid clustering produces stable IDs and records merge provenance."""

    memberships = {
        "a": "raw-a",
        "b": "raw-b",
        "c": "raw-c",
        "d": "raw-d",
    }
    embeddings = (
        EmbeddingRecord(conversation_key="a", vector=(1.0, 0.0)),
        EmbeddingRecord(conversation_key="b", vector=(0.99, 0.01)),
        EmbeddingRecord(conversation_key="c", vector=(0.0, 1.0)),
        EmbeddingRecord(conversation_key="d", vector=(0.01, 0.99)),
    )
    expected = {
        "a": "category-0001",
        "b": "category-0001",
        "c": "category-0002",
        "d": "category-0002",
    }

    first = semantic_graph.consolidate_communities(
        memberships,
        embeddings,
        max_categories=2,
    )
    reordered = semantic_graph.consolidate_communities(
        dict(reversed(tuple(memberships.items()))),
        tuple(reversed(embeddings)),
        max_categories=2,
    )

    assert first == reordered
    assert first.memberships == expected
    assert first.source_community_counts == {"category-0001": 2, "category-0002": 2}

    original_import = semantic_graph.import_module

    def unavailable_import(name: str):
        if name in {"numpy", "sklearn.cluster"}:
            raise ImportError(name)
        return original_import(name)

    monkeypatch.setattr(semantic_graph, "import_module", unavailable_import)
    fallback = semantic_graph.consolidate_communities(
        memberships,
        embeddings,
        max_categories=2,
    )
    assert fallback.memberships == expected
    assert fallback.source_community_counts == first.source_community_counts


class _TrackingEmbeddings(LocalHashingEmbeddingProvider):
    """Count provider calls while retaining production local behavior."""

    def __init__(self) -> None:
        super().__init__(dimensions=64)
        self.calls = 0

    def embed(self, items):  # type: ignore[no-untyped-def]
        """Count and delegate one embedding batch."""

        self.calls += 1
        return super().embed(items)


class _TrackingAnalysis(LocalHeuristicAnalysisProvider):
    """Count all structured-provider stages for cache testing."""

    def __init__(self) -> None:
        self.analysis_calls = 0
        self.taxonomy_calls = 0
        self.synthesis_calls = 0

    def analyze_conversations(self, items):  # type: ignore[no-untyped-def]
        """Count and delegate one profile batch."""

        self.analysis_calls += 1
        return super().analyze_conversations(items)

    def interpret_taxonomy(self, request):  # type: ignore[no-untyped-def]
        """Count and delegate one taxonomy call."""

        self.taxonomy_calls += 1
        return super().interpret_taxonomy(request)

    def synthesize_archive(self, request):  # type: ignore[no-untyped-def]
        """Count and delegate one synthesis call."""

        self.synthesis_calls += 1
        return super().synthesize_archive(request)


class _OrthogonalEmbeddings:
    """Return fixed disjoint vectors so graph discovery starts with two communities."""

    provider_name = "test-orthogonal"
    model_name = "orthogonal-v1"

    def embed(self, items):  # type: ignore[no-untyped-def]
        """Embed the two-item synthetic fixture without source-derived behavior."""

        vectors = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        return tuple(
            EmbeddingRecord(conversation_key=item.conversation_key, vector=vectors[index])
            for index, item in enumerate(items)
        )


def test_local_pipeline_exports_artifacts_and_resumes_from_cache(tmp_path: Path) -> None:
    """A complete offline run emits interoperable artifacts and skips cached provider work."""

    archive = _synthetic_archive()
    embeddings = _TrackingEmbeddings()
    analysis = _TrackingAnalysis()
    progress_events: list[tuple[str, int, int, int]] = []
    options = SemanticAtlasOptions(
        embedding_batch_size=1,
        analysis_batch_size=1,
        min_similarity=-1.0,
    )

    first = build_semantic_atlas(
        archive,
        tmp_path / "atlas",
        embedding_provider=embeddings,
        analysis_provider=analysis,
        options=options,
        progress_callback=lambda stage, completed, total, cached: progress_events.append(
            (stage, completed, total, cached)
        ),
    )
    initial_calls = (
        embeddings.calls,
        analysis.analysis_calls,
        analysis.taxonomy_calls,
        analysis.synthesis_calls,
    )
    second = build_semantic_atlas(
        archive,
        tmp_path / "atlas",
        embedding_provider=embeddings,
        analysis_provider=analysis,
        options=options,
        progress_callback=lambda stage, completed, total, cached: progress_events.append(
            (stage, completed, total, cached)
        ),
    )

    assert initial_calls == (2, 2, 1, 1)
    assert (
        embeddings.calls,
        analysis.analysis_calls,
        analysis.taxonomy_calls,
        analysis.synthesis_calls,
    ) == initial_calls
    assert first.catalog_path.is_file()
    assert first.graph_path.is_file()
    assert first.taxonomy_path.is_file()
    assert first.synthesis_path.is_file()
    assert first.atlas_html_path.is_file()
    assert second.manifest_path.is_file()
    html_text = first.atlas_html_path.read_text(encoding="utf-8")
    assert "Semantic Atlas" in html_text
    assert "Genomics notebook design" in html_text
    manifest_text = first.manifest_path.read_text(encoding="utf-8")
    assert "Design a reproducible" not in manifest_text
    assert "Genomics notebook design" not in manifest_text
    assert ("embeddings", 2, 2, 2) in progress_events
    assert ("conversation_profiles", 2, 2, 2) in progress_events
    assert ("artifact_export", 1, 1, 0) in progress_events


def test_pipeline_caps_leaf_categories_and_exports_consolidation_review(
    tmp_path: Path,
) -> None:
    """The end-to-end artifact run applies the cap before taxonomy interpretation."""

    result = build_semantic_atlas(
        _synthetic_archive(),
        tmp_path / "capped-atlas",
        embedding_provider=_OrthogonalEmbeddings(),
        analysis_provider=LocalHeuristicAnalysisProvider(),
        options=SemanticAtlasOptions(
            min_similarity=0.99,
            max_leaf_categories=1,
        ),
    )

    leaves = tuple(category for category in result.atlas.taxonomy.categories if category.level == 1)
    assert len(leaves) == 1
    assert result.atlas.review_queue.counts_by_code["consolidated_category"] == 1
    review_payload = json.loads(result.review_queue_path.read_text(encoding="utf-8"))
    consolidated = [
        item for item in review_payload["items"] if item["code"] == "consolidated_category"
    ]
    assert consolidated[0]["diagnostics"] == {"source_community_count": 2}
    assert result.taxonomy_path.is_file()


def test_pipeline_refines_only_bounded_representatives_and_resumes_separate_caches(
    tmp_path: Path,
) -> None:
    """External-quality profiles replace selected local records without cache thrashing."""

    archive = _synthetic_archive()
    baseline = _TrackingAnalysis()
    refinement = _TrackingAnalysis()
    options = SemanticAtlasOptions(
        analysis_batch_size=8,
        min_similarity=0.99,
        max_refined_conversations=1,
        refined_conversations_per_category=1,
    )
    arguments = {
        "archive": archive,
        "output_directory": tmp_path / "selective-atlas",
        "embedding_provider": _OrthogonalEmbeddings(),
        "analysis_provider": baseline,
        "refinement_provider": refinement,
        "interpretation_provider": refinement,
        "options": options,
    }

    first = build_semantic_atlas(**arguments)  # type: ignore[arg-type]
    calls = (
        baseline.analysis_calls,
        refinement.analysis_calls,
        refinement.taxonomy_calls,
        refinement.synthesis_calls,
    )
    second = build_semantic_atlas(**arguments)  # type: ignore[arg-type]

    assert calls == (1, 1, 1, 1)
    assert (
        baseline.analysis_calls,
        refinement.analysis_calls,
        refinement.taxonomy_calls,
        refinement.synthesis_calls,
    ) == calls
    assert len(first.atlas.analyses) == len(archive.conversations)
    assert second.manifest_path.is_file()
    assert (tmp_path / "selective-atlas/.semantic_cache/analyses.jsonl").is_file()
    assert (tmp_path / "selective-atlas/.semantic_cache/refined_analyses.jsonl").is_file()
