"""Injectable provider interfaces for semantic embeddings and structured analysis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from chatgpt_archive_compiler.semantic.models import (
    ArchiveSynthesisBundle,
    ArchiveSynthesisRequest,
    ConversationAnalysis,
    ConversationRepresentation,
    EmbeddingRecord,
    TaxonomyInterpretation,
    TaxonomyInterpretationRequest,
)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Provider capable of embedding a batch of deterministic representations."""

    @property
    def provider_name(self) -> str:
        """Return a stable provider identifier for provenance and cache keys."""

    @property
    def model_name(self) -> str:
        """Return the exact embedding model identifier."""

    def embed(self, items: Sequence[ConversationRepresentation]) -> Sequence[EmbeddingRecord]:
        """Return exactly one keyed embedding record for every supplied item."""


@runtime_checkable
class StructuredAnalysisProvider(Protocol):
    """Provider for per-conversation, taxonomy, and global structured interpretation."""

    @property
    def provider_name(self) -> str:
        """Return a stable provider identifier for provenance and cache keys."""

    @property
    def model_name(self) -> str:
        """Return the exact analysis model identifier."""

    def analyze_conversations(
        self, items: Sequence[ConversationRepresentation]
    ) -> Sequence[ConversationAnalysis]:
        """Return exactly one keyed semantic profile for every supplied item."""

    def interpret_taxonomy(self, request: TaxonomyInterpretationRequest) -> TaxonomyInterpretation:
        """Name, describe, and relate all locally discovered graph communities."""

    def synthesize_archive(self, request: ArchiveSynthesisRequest) -> ArchiveSynthesisBundle:
        """Return category profiles, project timelines, and global synthesis."""


class RoutedStructuredAnalysisProvider:
    """Route profile, taxonomy, and synthesis stages to independently chosen providers.

    This makes the cost/quality allocation explicit: a high-volume representative-profile stage
    can use an economical model while the one archive-wide synthesis request can use a stronger
    model, all behind the same :class:`StructuredAnalysisProvider` interface.
    """

    def __init__(
        self,
        *,
        profile_provider: StructuredAnalysisProvider,
        taxonomy_provider: StructuredAnalysisProvider | None = None,
        synthesis_provider: StructuredAnalysisProvider | None = None,
    ) -> None:
        self._profile_provider = profile_provider
        self._taxonomy_provider = taxonomy_provider or profile_provider
        self._synthesis_provider = synthesis_provider or self._taxonomy_provider

    @property
    def provider_name(self) -> str:
        """Return a cache-safe identifier containing every routed provider configuration."""

        return (
            f"routed/profile={self._profile_provider.provider_name}/"
            f"taxonomy={self._taxonomy_provider.provider_name}/"
            f"synthesis={self._synthesis_provider.provider_name}"
        )

    @property
    def model_name(self) -> str:
        """Return the exact model allocation used for all three structured stages."""

        return (
            f"profile={self._profile_provider.model_name};"
            f"taxonomy={self._taxonomy_provider.model_name};"
            f"synthesis={self._synthesis_provider.model_name}"
        )

    def analyze_conversations(
        self, items: Sequence[ConversationRepresentation]
    ) -> Sequence[ConversationAnalysis]:
        """Delegate representative conversation profiling to the configured profile provider."""

        return self._profile_provider.analyze_conversations(items)

    def interpret_taxonomy(self, request: TaxonomyInterpretationRequest) -> TaxonomyInterpretation:
        """Delegate category interpretation to the configured taxonomy provider."""

        return self._taxonomy_provider.interpret_taxonomy(request)

    def synthesize_archive(self, request: ArchiveSynthesisRequest) -> ArchiveSynthesisBundle:
        """Delegate the single global synthesis to the configured synthesis provider."""

        return self._synthesis_provider.synthesize_archive(request)
