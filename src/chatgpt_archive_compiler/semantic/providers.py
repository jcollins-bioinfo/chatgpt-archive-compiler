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
