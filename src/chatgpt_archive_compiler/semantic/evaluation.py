"""Ground-truth-driven diagnostics for the fictional contest archive."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from chatgpt_archive_compiler.semantic.models import SemanticAtlas


@dataclass(frozen=True)
class DemoAcceptance:
    """Separate acceptance metrics and actionable failure diagnostics."""

    project_timeline_count: int
    matched_project_count: int
    longitudinal_insight_count: int
    open_loop_count: int
    dormant_thread_count: int
    reversal_count: int
    cross_domain_connection_count: int
    duplicate_category_names: tuple[str, ...]
    evidence_link_failures: int
    unsupported_insight_count: int
    missed_projects: tuple[str, ...]
    fragmented_projects: tuple[str, ...]
    merged_projects: tuple[str, ...]
    missed_milestones: tuple[str, ...]
    dominant_generic_terms: tuple[str, ...]
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible validation report."""

        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in self.__dict__.items()
        }


def evaluate_contest_atlas(atlas: SemanticAtlas, truth: dict[str, Any]) -> DemoAcceptance:
    """Match inferred projects by conversation overlap and report independent quality signals."""

    source_id_by_key = {
        item.conversation_key: item.source_conversation_id for item in atlas.representations
    }
    inferred_sets = {
        timeline.project_name: {
            source_id_by_key.get(key)
            for event in timeline.events
            for key in event.conversation_keys
            if source_id_by_key.get(key)
        }
        for timeline in atlas.project_timelines
    }
    truth_sets = {
        item["project_id"]: {reference["conversation_id"] for reference in item["conversations"]}
        for item in truth["projects"]
    }
    overlaps: dict[str, list[tuple[float, str]]] = {}
    for project_id, expected in truth_sets.items():
        matches: list[tuple[float, str]] = []
        for label, actual in inferred_sets.items():
            union = expected | actual
            score = len(expected & actual) / len(union) if union else 0.0
            matches.append((score, label))
        overlaps[project_id] = sorted(matches, reverse=True)
    matched = {project for project, matches in overlaps.items() if matches and matches[0][0] >= 0.2}
    missed = tuple(sorted(set(truth_sets) - matched))
    fragmented = tuple(
        sorted(
            project
            for project, matches in overlaps.items()
            if sum(score >= 0.12 for score, _label in matches) > 1
        )
    )
    best_truth_by_inferred: defaultdict_list = defaultdict_list()
    for project, matches in overlaps.items():
        if matches and matches[0][0] >= 0.12:
            best_truth_by_inferred.add(matches[0][1], project)
    merged = tuple(
        sorted(label for label, projects in best_truth_by_inferred.items() if len(projects) > 1)
    )

    leaf_names = [
        item.name.casefold().strip() for item in atlas.taxonomy.categories if item.level == 1
    ]
    duplicate_names = tuple(
        sorted(name for name, count in Counter(leaf_names).items() if count > 1)
    )
    known_keys = set(source_id_by_key)
    evidence_failures = sum(
        key not in known_keys
        for timeline in atlas.project_timelines
        for event in timeline.events
        for key in event.conversation_keys
    )
    inferred_milestones = {
        source_id_by_key.get(key)
        for timeline in atlas.project_timelines
        for event in timeline.events
        for key in event.conversation_keys
    }
    missed_milestones = tuple(
        item["conversation_id"]
        for item in truth["milestones"]
        if item["conversation_id"] not in inferred_milestones
    )
    generic = {"analysis", "design", "review", "conversation", "project", "result", "using"}
    defining = Counter(
        term.casefold()
        for category in atlas.taxonomy.categories
        if category.level == 1
        for term in category.defining_concepts
    )
    dominant_generic = tuple(
        term for term, count in defining.most_common() if term in generic and count >= 2
    )
    synthesis = atlas.synthesis
    open_count = sum(len(item.unresolved_work) for item in atlas.project_timelines)
    unsupported = sum(not timeline.events for timeline in atlas.project_timelines)
    passed = all(
        (
            len(atlas.project_timelines) >= 4,
            len(matched) >= 4,
            len(synthesis.temporal_evolution) >= 5,
            open_count >= 4,
            len(synthesis.dormant_threads) >= 2,
            len(synthesis.tensions_and_reversals) >= 2,
            len(synthesis.cross_domain_connections) >= 4,
            not duplicate_names,
            not dominant_generic,
            evidence_failures == 0,
            unsupported == 0,
        )
    )
    return DemoAcceptance(
        project_timeline_count=len(atlas.project_timelines),
        matched_project_count=len(matched),
        longitudinal_insight_count=len(synthesis.temporal_evolution),
        open_loop_count=open_count,
        dormant_thread_count=len(synthesis.dormant_threads),
        reversal_count=len(synthesis.tensions_and_reversals),
        cross_domain_connection_count=len(synthesis.cross_domain_connections),
        duplicate_category_names=duplicate_names,
        evidence_link_failures=evidence_failures,
        unsupported_insight_count=unsupported,
        missed_projects=missed,
        fragmented_projects=fragmented,
        merged_projects=merged,
        missed_milestones=missed_milestones,
        dominant_generic_terms=dominant_generic,
        passed=passed,
    )


class defaultdict_list(dict[str, list[str]]):
    """Tiny typed multimap avoiding a default-factory cast in strict mypy."""

    def add(self, key: str, value: str) -> None:
        self.setdefault(key, []).append(value)
