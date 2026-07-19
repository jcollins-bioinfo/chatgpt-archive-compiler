"""Deterministic local-only providers for private, no-network semantic runs."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

from chatgpt_archive_compiler.semantic.models import (
    ArchiveSynthesis,
    ArchiveSynthesisBundle,
    ArchiveSynthesisRequest,
    CategoryInterpretation,
    CategoryProfile,
    ConversationAnalysis,
    ConversationRepresentation,
    EmbeddingRecord,
    ProjectTimeline,
    TaxonomyInterpretation,
    TaxonomyInterpretationRequest,
    TimelineEvent,
)

_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{2,}")
_WHITESPACE = re.compile(r"\s+")
_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "assistant",
    "because",
    "been",
    "before",
    "being",
    "but",
    "can",
    "could",
    "does",
    "for",
    "from",
    "have",
    "how",
    "into",
    "its",
    "just",
    "more",
    "not",
    "please",
    "should",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "user",
    "was",
    "what",
    "when",
    "which",
    "will",
    "with",
    "would",
    "you",
    "your",
}

_DOMAIN_KEYWORDS: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "Scientific and Technical Work",
        frozenset(
            {
                "analysis",
                "bioinformatics",
                "biology",
                "code",
                "data",
                "github",
                "model",
                "notebook",
                "python",
                "research",
                "science",
                "software",
            }
        ),
    ),
    (
        "Health and Wellbeing",
        frozenset(
            {
                "anxiety",
                "doctor",
                "health",
                "medical",
                "medication",
                "sleep",
                "symptom",
                "treatment",
            }
        ),
    ),
    (
        "Music, Arts, and Culture",
        frozenset(
            {
                "art",
                "bach",
                "book",
                "composition",
                "literature",
                "midi",
                "music",
                "piano",
            }
        ),
    ),
    (
        "Career, Learning, and Finance",
        frozenset(
            {
                "career",
                "education",
                "finance",
                "interview",
                "job",
                "learn",
                "resume",
                "salary",
            }
        ),
    ),
    (
        "Home and Practical Life",
        frozenset(
            {
                "apartment",
                "car",
                "drive",
                "fan",
                "home",
                "mac",
                "plant",
                "storage",
            }
        ),
    ),
    (
        "Society, History, and Ideas",
        frozenset(
            {
                "bible",
                "history",
                "humanity",
                "philosophy",
                "politics",
                "religion",
                "society",
                "universe",
            }
        ),
    ),
)


def _tokens(text: str) -> list[str]:
    """Tokenize text into stable, content-bearing lowercase terms."""

    return [
        token.casefold()
        for token in _TOKEN_PATTERN.findall(text)
        if token.casefold() not in _STOPWORDS
    ]


def _label(words: Sequence[str], fallback: str) -> str:
    """Turn up to three keywords into a compact human-readable label."""

    selected = [word.replace("_", " ").title() for word in words[:3]]
    return " · ".join(selected) if selected else fallback


def _domain_for(words: Sequence[str]) -> str:
    """Select a broad local-only parent domain from transparent keyword overlap."""

    word_set = {word.casefold() for word in words}
    scored = [(len(word_set & terms), name) for name, terms in _DOMAIN_KEYWORDS]
    score, name = min(scored, key=lambda item: (-item[0], item[1]))
    return name if score else "General and Personal Conversations"


class LocalHashingEmbeddingProvider:
    """Stateless signed feature-hashing embeddings that never leave the runtime."""

    def __init__(self, dimensions: int = 512) -> None:
        """Create a provider with a fixed positive feature-vector dimension."""

        if dimensions < 64:
            raise ValueError("Local hashing embeddings require at least 64 dimensions.")
        self._dimensions = dimensions

    @property
    def provider_name(self) -> str:
        """Return the stable local provider identifier."""

        return "local"

    @property
    def model_name(self) -> str:
        """Return a versioned algorithm and dimension identifier."""

        return f"signed-feature-hashing-v1-{self._dimensions}d"

    def embed(self, items: Sequence[ConversationRepresentation]) -> Sequence[EmbeddingRecord]:
        """Embed title-weighted unigrams and bigrams using deterministic BLAKE2 hashing."""

        records: list[EmbeddingRecord] = []
        for item in items:
            title_tokens = _tokens(item.title)
            body_tokens = _tokens(item.text)
            features = [*title_tokens, *title_tokens, *body_tokens]
            features.extend(
                f"{left}::{right}"
                for left, right in zip(body_tokens, body_tokens[1:], strict=False)
            )
            vector = [0.0] * self._dimensions
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
                index = int.from_bytes(digest[:8], "big") % self._dimensions
                sign = 1.0 if digest[8] & 1 else -1.0
                vector[index] += sign
            norm = math.sqrt(sum(value * value for value in vector))
            if norm:
                vector = [value / norm for value in vector]
            records.append(
                EmbeddingRecord(conversation_key=item.conversation_key, vector=tuple(vector))
            )
        return records


class LocalHeuristicAnalysisProvider:
    """Transparent heuristic analysis for fully offline baselines and private previews."""

    @property
    def provider_name(self) -> str:
        """Return the stable local provider identifier."""

        return "local"

    @property
    def model_name(self) -> str:
        """Return the versioned heuristic-rule identifier."""

        return "transparent-semantic-heuristics-v1"

    def analyze_conversations(
        self, items: Sequence[ConversationRepresentation]
    ) -> Sequence[ConversationAnalysis]:
        """Create deterministic keyword, purpose, and synopsis profiles independently per item."""

        analyses: list[ConversationAnalysis] = []
        for item in items:
            title_words = _tokens(item.title)
            counts = Counter(_tokens(f"{item.title} {item.title} {item.text}"))
            keywords = [word for word, _ in counts.most_common(12)]
            primary_words = title_words[:3] or keywords[:3]
            lowered = item.text.casefold()
            if any(marker in lowered for marker in ("traceback", "error", "failed", "debug")):
                conversation_type = "troubleshooting"
            elif any(marker in lowered for marker in ("create", "implement", "write", "build")):
                conversation_type = "implementation"
            elif any(marker in lowered for marker in ("recommend", "compare", "should i", "best")):
                conversation_type = "decision support"
            elif any(marker in lowered for marker in ("why", "explain", "describe", "what is")):
                conversation_type = "explanation and inquiry"
            else:
                conversation_type = "general discussion"
            project_markers = ("project", "repository", "repo", "notebook", "manuscript")
            projects = (
                (item.title.strip(),)
                if any(
                    marker in f"{item.title} {item.text[:500]}".casefold()
                    for marker in project_markers
                )
                else ()
            )
            compact = _WHITESPACE.sub(" ", item.text).strip()
            synopsis = compact[:360].rstrip()
            if len(compact) > 360:
                synopsis += "…"
            analyses.append(
                ConversationAnalysis(
                    conversation_key=item.conversation_key,
                    synopsis=synopsis or f"Conversation titled {item.title!r}.",
                    primary_subject=_label(primary_words, "General conversation"),
                    secondary_subjects=tuple(_label((word,), word) for word in keywords[3:8]),
                    projects=projects,
                    conversation_type=conversation_type,
                    entities=tuple(_label((word,), word) for word in keywords[:6]),
                    goals=(),
                    decisions=(),
                    unresolved_questions=(),
                    recurring_themes=tuple(_label((word,), word) for word in keywords[8:12]),
                    temporal_role=None,
                    related_conversation_keys=(),
                    confidence=0.5,
                )
            )
        return analyses

    def interpret_taxonomy(self, request: TaxonomyInterpretationRequest) -> TaxonomyInterpretation:
        """Name graph communities from their most frequent provider-independent labels."""

        interpretations: list[CategoryInterpretation] = []
        label_sets: dict[str, set[str]] = {}
        for cluster in request.clusters:
            labels = [label for label in cluster.candidate_labels if label.strip()]
            label_sets[cluster.category_id] = {label.casefold() for label in labels}
            interpretations.append(
                CategoryInterpretation(
                    category_id=cluster.category_id,
                    name=" · ".join(labels[:2]) or "Unclassified conversations",
                    description=(
                        "A locally discovered semantic community characterized by "
                        + (", ".join(labels[:5]) if labels else "mixed subjects")
                        + "."
                    ),
                    parent_name=_domain_for(labels),
                    defining_concepts=tuple(labels[:8]),
                    related_category_ids=(),
                    confidence=0.5,
                )
            )
        enriched: list[CategoryInterpretation] = []
        for interpretation in interpretations:
            overlaps = sorted(
                (
                    (len(label_sets[interpretation.category_id] & other_labels), other_id)
                    for other_id, other_labels in label_sets.items()
                    if other_id != interpretation.category_id
                ),
                key=lambda item: (-item[0], item[1]),
            )
            related = tuple(other_id for score, other_id in overlaps[:3] if score > 0)
            enriched.append(interpretation.model_copy(update={"related_category_ids": related}))
        return TaxonomyInterpretation(categories=tuple(enriched))

    def synthesize_archive(self, request: ArchiveSynthesisRequest) -> ArchiveSynthesisBundle:
        """Summarize local categories and project history without any external model."""

        analyses = {analysis.conversation_key: analysis for analysis in request.analyses}
        references = {reference.conversation_key: reference for reference in request.references}
        profiles: list[CategoryProfile] = []
        for category in request.taxonomy.categories:
            member_analyses = [
                analyses[key] for key in category.conversation_keys if key in analyses
            ]
            profiles.append(
                CategoryProfile(
                    category_id=category.category_id,
                    overview=(
                        f"{category.name} contains "
                        f"{len(category.conversation_keys)} conversations. "
                        f"{category.description}"
                    ),
                    characteristic_questions=tuple(
                        item.synopsis for item in member_analyses[:3] if item.synopsis
                    ),
                    major_conclusions=tuple(
                        conclusion for item in member_analyses for conclusion in item.decisions
                    )[:8],
                    unresolved_threads=tuple(
                        question
                        for item in member_analyses
                        for question in item.unresolved_questions
                    )[:8],
                    temporal_evolution=None,
                    representative_conversation_keys=tuple(
                        item.conversation_key for item in member_analyses[:5]
                    ),
                )
            )

        project_members: defaultdict[str, list[ConversationAnalysis]] = defaultdict(list)
        for analysis in request.analyses:
            for project in analysis.projects:
                project_members[project].append(analysis)
        timelines: list[ProjectTimeline] = []
        for project_name, members in sorted(project_members.items()):

            def timeline_sort_key(item: ConversationAnalysis) -> tuple[bool, str, str]:
                reference = references.get(item.conversation_key)
                occurred_at = (
                    reference.created_at or reference.updated_at if reference is not None else None
                )
                return (
                    occurred_at is None,
                    occurred_at.isoformat() if occurred_at is not None else "",
                    item.conversation_key,
                )

            ordered = sorted(
                members,
                key=timeline_sort_key,
            )
            events = tuple(
                TimelineEvent(
                    label=(
                        references[item.conversation_key].title
                        if item.conversation_key in references
                        else item.primary_subject
                    ),
                    summary=item.synopsis,
                    conversation_keys=(item.conversation_key,),
                    occurred_at=(
                        references[item.conversation_key].created_at
                        or references[item.conversation_key].updated_at
                        if item.conversation_key in references
                        else None
                    ),
                )
                for item in ordered[:12]
            )
            timelines.append(
                ProjectTimeline(
                    project_name=project_name,
                    overview=f"A recurring project represented by {len(members)} conversations.",
                    events=events,
                    current_state=None,
                    unresolved_work=tuple(
                        question for item in members for question in item.unresolved_questions
                    )[:10],
                )
            )

        parent_categories = [
            category for category in request.taxonomy.categories if category.parent_id is None
        ]
        synthesis = ArchiveSynthesis(
            executive_summary=(
                f"The semantic atlas organizes {len(request.analyses)} conversations into "
                f"{len(request.taxonomy.categories)} hierarchical categories and "
                f"{len(timelines)} recurring project timelines."
            ),
            dominant_domains=tuple(category.name for category in parent_categories[:12]),
            cross_domain_connections=(),
            recurring_patterns=tuple(
                label
                for label, _ in Counter(
                    theme for item in request.analyses for theme in item.recurring_themes
                ).most_common(10)
            ),
            temporal_evolution=(),
            tensions_and_reversals=(),
            dormant_threads=(),
            opportunities=(),
        )
        return ArchiveSynthesisBundle(
            category_profiles=tuple(profiles),
            project_timelines=tuple(timelines),
            synthesis=synthesis,
        )
