"""Deterministic local-only providers for private, no-network semantic runs."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import timedelta

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
    "conversation",
    "help",
    "need",
    "using",
    "work",
    "review",
    "analysis",
    "design",
    "result",
    "results",
    "approach",
    "project",
    "title",
    "call",
    "must",
    "fictional",
}

_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
_ARTIFACT_PATTERN = re.compile(
    r"\b(?:[a-z][a-z0-9]+(?:[-_][a-z0-9]+)+|[A-Za-z0-9_-]+\.(?:py|json|parquet|ipynb|md|csv|mid|midi|wav|pdf))\b"
)
_QUESTION_MARKERS = (
    "unresolved",
    "open question",
    "still need",
    "remains to",
    "not yet",
    "defer",
    "follow up",
)
_DECISION_MARKERS = (
    "decided",
    "decision",
    "we will",
    "switch to",
    "adopt",
    "keep",
    "drop",
    "defer",
    "pivot",
)
_REVERSAL_MARKERS = (
    "reversal",
    "reframe",
    "revised",
    "changed direction",
    "instead of",
    "no longer",
    "abandon",
    "pivot",
    "weakened",
    "disappeared",
)
_RESUMPTION_MARKERS = ("resumed", "reactivated", "returned to", "after the hiatus")


def _sentences(text: str) -> tuple[str, ...]:
    """Return bounded normalized sentences for transparent feature extraction."""

    compact = _WHITESPACE.sub(" ", text).strip()
    return tuple(part.strip()[:500] for part in _SENTENCE_PATTERN.split(compact) if part.strip())


def _matching_sentences(text: str, markers: Sequence[str], *, limit: int = 4) -> tuple[str, ...]:
    """Select source sentences containing any declared marker."""

    return tuple(
        sentence
        for sentence in _sentences(text)
        if any(marker in sentence.casefold() for marker in markers)
    )[:limit]


def _project_candidates(text: str) -> tuple[str, ...]:
    """Extract stable artifact/repository handles that can link evolving project language."""

    return tuple(dict.fromkeys(match.casefold() for match in _ARTIFACT_PATTERN.findall(text)))


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

        return f"signed-feature-hashing-v2-{self._dimensions}d"

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

        return "transparent-semantic-heuristics-v2"

    def analyze_conversations(
        self, items: Sequence[ConversationRepresentation]
    ) -> Sequence[ConversationAnalysis]:
        """Create deterministic keyword, purpose, and synopsis profiles independently per item."""

        candidate_counts = Counter(
            candidate
            for item in items
            for candidate in _project_candidates(f"{item.title} {item.text}")
        )
        recurring_artifacts = {
            candidate
            for candidate, count in candidate_counts.items()
            if count >= 3 and count <= max(3, int(len(items) * 0.45))
        }
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
            candidates = _project_candidates(f"{item.title} {item.text}")
            local_candidate_counts = Counter(
                match.casefold() for match in _ARTIFACT_PATTERN.findall(f"{item.title} {item.text}")
            )
            projects = tuple(
                _label((candidate,), candidate)
                for candidate in candidates
                if candidate in recurring_artifacts
                or (
                    "-" in candidate
                    and "." not in candidate
                    and local_candidate_counts[candidate] >= 2
                )
            )[:2]
            compact = _WHITESPACE.sub(" ", item.text).strip()
            synopsis = compact[:360].rstrip()
            if len(compact) > 360:
                synopsis += "…"
            decisions = _matching_sentences(item.text, _DECISION_MARKERS)
            unresolved = _matching_sentences(item.text, _QUESTION_MARKERS)
            goals = _matching_sentences(
                item.text,
                ("goal", "aim", "requirement", "hypothesis", "objective", "want to"),
                limit=3,
            )
            temporal_role: str | None = None
            if any(marker in lowered for marker in _RESUMPTION_MARKERS):
                temporal_role = "resumption"
            elif any(marker in lowered for marker in _REVERSAL_MARKERS):
                temporal_role = "reversal or reframing"
            elif any(
                marker in lowered for marker in ("completed", "shipped", "validated", "milestone")
            ):
                temporal_role = "milestone"
            elif any(marker in lowered for marker in ("start", "initial", "first prototype")):
                temporal_role = "initiation"
            analyses.append(
                ConversationAnalysis(
                    conversation_key=item.conversation_key,
                    synopsis=synopsis or f"Conversation titled {item.title!r}.",
                    primary_subject=_label(primary_words, "General conversation"),
                    secondary_subjects=tuple(_label((word,), word) for word in keywords[3:8]),
                    projects=projects,
                    conversation_type=conversation_type,
                    entities=tuple(_label((word,), word) for word in keywords[:6]),
                    goals=goals,
                    decisions=decisions,
                    unresolved_questions=unresolved,
                    recurring_themes=tuple(_label((word,), word) for word in keywords[8:12]),
                    temporal_role=temporal_role,
                    related_conversation_keys=(),
                    confidence=0.5,
                )
            )
        return analyses

    def interpret_taxonomy(self, request: TaxonomyInterpretationRequest) -> TaxonomyInterpretation:
        """Name graph communities from their most frequent provider-independent labels."""

        interpretations: list[CategoryInterpretation] = []
        label_sets: dict[str, set[str]] = {}
        used_names: Counter[str] = Counter()
        for cluster in request.clusters:
            labels = [label for label in cluster.candidate_labels if label.strip()]
            label_sets[cluster.category_id] = {label.casefold() for label in labels}
            base_name = " · ".join(labels[:2]) or "Unclassified conversations"
            name = base_name
            next_label = 2
            while used_names[name.casefold()] and next_label < len(labels):
                name = f"{base_name} · {labels[next_label]}"
                next_label += 1
            used_names[name.casefold()] += 1
            if used_names[name.casefold()] > 1:
                name = f"{name} · related thread"
            interpretations.append(
                CategoryInterpretation(
                    category_id=cluster.category_id,
                    name=name,
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
            if len(members) < 3:
                continue

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
            event_members = ordered
            if len(ordered) > 12:
                indexes = sorted({round(index * (len(ordered) - 1) / 11) for index in range(12)})
                event_members = [ordered[index] for index in indexes]
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
                for item in event_members
            )
            unresolved_work = tuple(
                dict.fromkeys(
                    question for item in members for question in item.unresolved_questions
                )
            )[:10]
            timelines.append(
                ProjectTimeline(
                    project_name=project_name,
                    overview=f"A recurring project represented by {len(members)} conversations.",
                    events=events,
                    current_state=ordered[-1].synopsis if ordered else None,
                    unresolved_work=unresolved_work,
                )
            )

        parent_categories = [
            category for category in request.taxonomy.categories if category.parent_id is None
        ]
        cross_connections = tuple(
            dict.fromkeys(
                f"{left} connects with {right} through shared evidence in "
                f"{analysis.primary_subject}."
                for analysis in request.analyses
                for left, right in zip(analysis.projects, analysis.projects[1:], strict=False)
                if left != right
            )
        )[:12]
        reversals = tuple(
            dict.fromkeys(
                decision
                for analysis in request.analyses
                if analysis.temporal_role == "reversal or reframing"
                for decision in (analysis.decisions or (analysis.synopsis,))
            )
        )[:12]
        temporal = tuple(
            f"{references[item.conversation_key].title}: {item.temporal_role}."
            for item in request.analyses
            if item.temporal_role and item.conversation_key in references
        )[:12]
        dormant: list[str] = []
        for project_name, members in sorted(project_members.items()):
            dated_values = []
            for member in members:
                reference = references.get(member.conversation_key)
                occurred_at = (
                    reference.created_at or reference.updated_at if reference is not None else None
                )
                if occurred_at is not None:
                    dated_values.append(occurred_at)
            dated = sorted(dated_values)
            if any(
                later - earlier >= timedelta(days=75)
                for earlier, later in zip(dated, dated[1:], strict=False)
            ):
                dormant.append(
                    f"{project_name} contains a dormant interval followed by renewed activity."
                )
        opportunities = tuple(
            dict.fromkeys(
                question for item in request.analyses for question in item.unresolved_questions
            )
        )[:12]
        repeated_projects = tuple(
            project for project, members in project_members.items() if len(members) >= 3
        )
        synthesis = ArchiveSynthesis(
            executive_summary=(
                f"The semantic atlas organizes {len(request.analyses)} conversations into "
                f"{len(request.taxonomy.categories)} hierarchical categories and "
                f"{len(timelines)} recurring project timelines."
            ),
            dominant_domains=tuple(category.name for category in parent_categories[:12]),
            cross_domain_connections=cross_connections,
            recurring_patterns=(
                *repeated_projects[:6],
                *tuple(
                    label
                    for label, _ in Counter(
                        theme for item in request.analyses for theme in item.recurring_themes
                    ).most_common(6)
                ),
            ),
            temporal_evolution=temporal,
            tensions_and_reversals=reversals,
            dormant_threads=tuple(dormant[:12]),
            opportunities=opportunities,
        )
        return ArchiveSynthesisBundle(
            category_profiles=tuple(profiles),
            project_timelines=tuple(timelines),
            synthesis=synthesis,
        )
