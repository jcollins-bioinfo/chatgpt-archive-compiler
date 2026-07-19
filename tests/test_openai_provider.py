"""Tests for the optional OpenAI semantic provider using synthetic client doubles only."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chatgpt_archive_compiler.exceptions import ApiBudgetExceededError, SemanticProviderError
from chatgpt_archive_compiler.semantic.budget import ApiBudget, ModelTokenPrice
from chatgpt_archive_compiler.semantic.models import (
    ArchiveSynthesis,
    ArchiveSynthesisBundle,
    ArchiveSynthesisRequest,
    Category,
    CategoryAssignment,
    CategoryProfile,
    ConversationAnalysis,
    ConversationReference,
    ConversationRepresentation,
    EmbeddingRecord,
    GraphSummary,
    Taxonomy,
)
from chatgpt_archive_compiler.semantic.openai_provider import (
    OpenAIEmbeddingProvider,
    OpenAIStructuredAnalysisProvider,
    _AnalysisBatch,
    _stratified_analyses,
)


def test_budgeted_providers_disable_sdk_level_retries(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Every budgeted transmission must receive its own durable ledger reservation."""

    configured_retries: list[int] = []

    def fake_client_loader(
        *, api_key: str | None, timeout_seconds: float, max_retries: int
    ) -> object:
        del api_key, timeout_seconds
        configured_retries.append(max_retries)
        return object()

    monkeypatch.setattr(
        "chatgpt_archive_compiler.semantic.openai_provider._load_openai_client",
        fake_client_loader,
    )
    budget = ApiBudget(max_cost_usd="1", ledger_path=tmp_path / "ledger.json")
    price = ModelTokenPrice(input_usd_per_million="1", output_usd_per_million="1")

    OpenAIEmbeddingProvider(budget=budget, token_price=price, max_retries=9)
    OpenAIStructuredAnalysisProvider(budget=budget, token_price=price, max_retries=9)

    assert configured_retries == [0, 0]


def _representation(key: str = "conversation-key-0001") -> ConversationRepresentation:
    return ConversationRepresentation(
        conversation_key=key,
        source_index=0,
        title="Synthetic subject",
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        text=(
            "Title:\nSynthetic subject\n\n"
            "USER: A synthetic research question.\nASSISTANT: A synthetic conclusion."
        ),
        message_count=2,
        character_count=69,
        original_character_count=69,
        text_sha256="a" * 64,
    )


def _analysis(key: str = "conversation-key-0001") -> ConversationAnalysis:
    return ConversationAnalysis(
        conversation_key=key,
        synopsis="A synthetic question and conclusion.",
        primary_subject="Synthetic research",
        conversation_type="research",
        confidence=0.9,
    )


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> SimpleNamespace:
        self.requests.append(request)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(index), 1.0])
                for index, _ in enumerate(request["input"])
            ]
        )


class _FakeResponses:
    def __init__(self, parsed: BaseException | object, usage: object | None = None) -> None:
        self.parsed = parsed
        self.usage = usage
        self.requests: list[dict[str, Any]] = []

    def parse(self, **request: Any) -> SimpleNamespace:
        self.requests.append(request)
        if isinstance(self.parsed, BaseException):
            raise self.parsed
        return SimpleNamespace(output_parsed=self.parsed, usage=self.usage)


def test_embedding_provider_preserves_keys_and_requests_shortened_vectors() -> None:
    embeddings = _FakeEmbeddings()
    client = SimpleNamespace(embeddings=embeddings)
    provider = OpenAIEmbeddingProvider(client=client, dimensions=256)

    records = provider.embed([_representation()])

    assert records == (
        EmbeddingRecord(conversation_key="conversation-key-0001", vector=(0.0, 1.0)),
    )
    assert embeddings.requests[0]["model"] == "text-embedding-3-large"
    assert embeddings.requests[0]["dimensions"] == 256
    assert embeddings.requests[0]["input"][0].count("Synthetic subject") == 1
    assert "dimensions=256" in provider.provider_name


def test_structured_provider_uses_nonstored_responses_and_validates_keys() -> None:
    parsed = _AnalysisBatch(analyses=(_analysis(),))
    responses = _FakeResponses(parsed)
    provider = OpenAIStructuredAnalysisProvider(client=SimpleNamespace(responses=responses))

    analyses = provider.analyze_conversations([_representation()])

    assert analyses == (_analysis(),)
    assert responses.requests[0]["store"] is False
    assert responses.requests[0]["text_format"] is _AnalysisBatch
    assert responses.requests[0]["max_output_tokens"] == 32_000
    assert "profile=high:8000:32000" in provider.provider_name


def test_structured_provider_accounts_reported_usage_in_shared_budget(tmp_path: Path) -> None:
    """A successful request replaces its conservative reservation with actual SDK usage."""

    budget = ApiBudget(max_cost_usd="0.10", ledger_path=tmp_path / "budget.json")
    price = ModelTokenPrice(input_usd_per_million=1, output_usd_per_million=6)
    responses = _FakeResponses(
        _AnalysisBatch(analyses=(_analysis(),)),
        usage=SimpleNamespace(input_tokens=500, output_tokens=100),
    )
    provider = OpenAIStructuredAnalysisProvider(
        client=SimpleNamespace(responses=responses),
        profile_max_output_tokens=1_000,
        taxonomy_max_output_tokens=1_000,
        synthesis_max_output_tokens=1_000,
        budget=budget,
        token_price=price,
    )

    provider.analyze_conversations([_representation()])

    snapshot = budget.snapshot()
    assert snapshot.request_count == 1
    assert snapshot.completed_request_count == 1
    assert snapshot.charged_cost_usd == Decimal("0.0011")


def test_structured_provider_stops_before_call_when_reserve_exceeds_budget(
    tmp_path: Path,
) -> None:
    """The client receives no request when worst-case configured output is unaffordable."""

    budget = ApiBudget(max_cost_usd="0.001", ledger_path=tmp_path / "budget.json")
    responses = _FakeResponses(_AnalysisBatch(analyses=(_analysis(),)))
    provider = OpenAIStructuredAnalysisProvider(
        client=SimpleNamespace(responses=responses),
        profile_max_output_tokens=1_000,
        taxonomy_max_output_tokens=1_000,
        synthesis_max_output_tokens=1_000,
        budget=budget,
        token_price=ModelTokenPrice(input_usd_per_million=1, output_usd_per_million=6),
    )

    with pytest.raises(ApiBudgetExceededError, match="before transmission"):
        provider.analyze_conversations([_representation()])

    assert responses.requests == []


def test_structured_provider_suppresses_source_derived_provider_errors() -> None:
    provider = OpenAIStructuredAnalysisProvider(
        client=SimpleNamespace(responses=_FakeResponses(RuntimeError("private source fragment")))
    )

    with pytest.raises(SemanticProviderError, match=r"failed safely \(RuntimeError\)") as caught:
        provider.analyze_conversations([_representation()])

    assert "private source fragment" not in str(caught.value)


def test_empty_structured_response_is_rejected() -> None:
    provider = OpenAIStructuredAnalysisProvider(
        client=SimpleNamespace(responses=_FakeResponses(None))
    )

    with pytest.raises(SemanticProviderError, match="empty or refused"):
        provider.analyze_conversations([_representation()])


def test_synthesis_schema_can_be_returned_by_fake_client() -> None:
    bundle = ArchiveSynthesisBundle(
        category_profiles=(),
        project_timelines=(),
        synthesis=ArchiveSynthesis(executive_summary="Synthetic global synthesis."),
    )
    responses = _FakeResponses(bundle)
    provider = OpenAIStructuredAnalysisProvider(client=SimpleNamespace(responses=responses))

    parsed = provider._parse(
        system_prompt="Synthetic system prompt.",
        payload={"synthetic": True},
        response_model=ArchiveSynthesisBundle,
        reasoning_effort="low",
        max_output_tokens=1_000,
        stage="synthetic",
    )

    assert parsed == bundle


def test_synthesis_profiles_all_taxonomy_levels_and_preserves_temporal_evidence() -> None:
    analyses = tuple(
        _analysis(f"conversation-key-{index:04d}").model_copy(
            update={"projects": ("Synthetic project",), "confidence": index / 20}
        )
        for index in range(20)
    )
    references = tuple(
        ConversationReference(
            conversation_key=analysis.conversation_key,
            title=f"Synthetic conversation {index}",
            created_at=datetime(2025, 1, index + 1, tzinfo=UTC),
        )
        for index, analysis in enumerate(analyses)
    )
    parent = Category(
        category_id="domain-synthetic",
        name="Synthetic domain",
        description="A broad synthetic domain.",
        level=0,
        conversation_keys=tuple(item.conversation_key for item in analyses),
        child_ids=("category-synthetic",),
        confidence=0.9,
    )
    leaf = Category(
        category_id="category-synthetic",
        name="Synthetic category",
        description="A focused synthetic category.",
        parent_id=parent.category_id,
        level=1,
        conversation_keys=tuple(item.conversation_key for item in analyses),
        confidence=0.9,
    )
    taxonomy = Taxonomy(
        categories=(parent, leaf),
        assignments=tuple(
            CategoryAssignment(
                conversation_key=item.conversation_key,
                primary_category_id=leaf.category_id,
                confidence=0.9,
            )
            for item in analyses
        ),
    )
    bundle = ArchiveSynthesisBundle(
        category_profiles=tuple(
            CategoryProfile(category_id=item.category_id, overview=item.description)
            for item in (parent, leaf)
        ),
        project_timelines=(),
        synthesis=ArchiveSynthesis(executive_summary="Synthetic global synthesis."),
    )
    responses = _FakeResponses(bundle)
    provider = OpenAIStructuredAnalysisProvider(client=SimpleNamespace(responses=responses))

    result = provider.synthesize_archive(
        ArchiveSynthesisRequest(
            analyses=analyses,
            references=references,
            taxonomy=taxonomy,
            graph_summary=GraphSummary(
                node_count=20,
                edge_count=19,
                category_count=1,
                isolated_node_count=0,
            ),
        )
    )

    sampled = _stratified_analyses(
        analyses, {item.conversation_key: item for item in references}, limit=10
    )
    assert analyses[0].conversation_key in {item.conversation_key for item in sampled}
    assert analyses[-1].conversation_key in {item.conversation_key for item in sampled}
    assert {item.category_id for item in result.category_profiles} == {
        parent.category_id,
        leaf.category_id,
    }
    request_payload = json.loads(responses.requests[0]["input"][1]["content"])
    assert {item["category"]["category_id"] for item in request_payload["categories"]} == {
        parent.category_id,
        leaf.category_id,
    }
