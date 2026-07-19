"""Deterministic sparse similarity graph and local community discovery."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module

from chatgpt_archive_compiler.semantic.errors import SemanticAtlasError
from chatgpt_archive_compiler.semantic.models import (
    ConversationAnalysis,
    EmbeddingRecord,
    GraphEdge,
    SemanticAtlasOptions,
)


@dataclass(frozen=True)
class CommunityConsolidation:
    """Final memberships and provenance for categories merged to satisfy a cap."""

    memberships: dict[str, str]
    source_community_counts: dict[str, int]


def _validate_embeddings(records: Sequence[EmbeddingRecord]) -> int:
    """Validate unique keys, fixed dimensions, and finite vector components."""

    if not records:
        return 0
    dimensions = len(records[0].vector)
    keys: set[str] = set()
    for record in records:
        if record.conversation_key in keys:
            raise SemanticAtlasError(
                "The embedding provider returned a duplicate conversation key."
            )
        keys.add(record.conversation_key)
        if len(record.vector) != dimensions:
            raise SemanticAtlasError("Embedding vectors do not all have the same dimension.")
        if any(not math.isfinite(value) for value in record.vector):
            raise SemanticAtlasError("An embedding vector contains a non-finite value.")
    return dimensions


def _cosine(
    left: tuple[float, ...],
    right: tuple[float, ...],
    left_norm: float,
    right_norm: float,
) -> float:
    """Compute cosine similarity for two validated equal-length vectors."""

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _top_dimensions(vector: tuple[float, ...], count: int) -> tuple[int, ...]:
    """Return dimensions with greatest absolute magnitude using stable tie-breaking."""

    return tuple(
        index
        for index, _ in sorted(
            enumerate(vector),
            key=lambda item: (-abs(item[1]), item[0]),
        )[:count]
    )


def _candidate_pairs(
    records: Sequence[EmbeddingRecord], options: SemanticAtlasOptions
) -> set[tuple[int, int]]:
    """Construct bounded deterministic candidates using top-dimension buckets."""

    count = len(records)
    if count <= options.graph_brute_force_threshold:
        return {(left, right) for left in range(count) for right in range(left + 1, count)}

    dimensions = len(records[0].vector)
    signature_size = min(options.graph_candidate_dimensions, dimensions)
    signatures = [_top_dimensions(record.vector, signature_size) for record in records]
    buckets: defaultdict[tuple[int, bool], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        for dimension in signatures[index]:
            buckets[(dimension, record.vector[dimension] >= 0.0)].append(index)

    coarse_dimensions = tuple(
        sorted(
            {
                min(dimensions - 1, (offset * dimensions) // 48)
                for offset in range(min(48, dimensions))
            }
        )
    )
    candidate_limit = max(options.max_neighbors * 8, 32)
    retained_limit = max(options.max_neighbors * 4, 16)
    pairs: set[tuple[int, int]] = set()
    for index, record in enumerate(records):
        candidates: set[int] = set()
        for dimension in signatures[index]:
            candidates.update(buckets[(dimension, record.vector[dimension] >= 0.0)])
        candidates.discard(index)

        # Deterministic supplementation prevents unusual vectors from receiving no comparison.
        offset = 1
        target_count = min(candidate_limit, count - 1)
        while len(candidates) < target_count:
            candidates.add((index + offset * 104_729) % count)
            candidates.discard(index)
            offset += 1

        if len(candidates) > candidate_limit:
            candidates = set(
                sorted(
                    candidates,
                    key=lambda candidate: (
                        -sum(
                            record.vector[dimension] * records[candidate].vector[dimension]
                            for dimension in coarse_dimensions
                        ),
                        records[candidate].conversation_key,
                    ),
                )[:candidate_limit]
            )
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                -sum(
                    record.vector[dimension] * records[candidate].vector[dimension]
                    for dimension in coarse_dimensions
                ),
                records[candidate].conversation_key,
            ),
        )[:retained_limit]
        for candidate in ranked:
            pairs.add((min(index, candidate), max(index, candidate)))
    return pairs


def _shared_signals(
    source: ConversationAnalysis | None,
    target: ConversationAnalysis | None,
) -> tuple[str, ...]:
    """Return compact overlapping analysis labels for an edge."""

    if source is None or target is None:
        return ()
    source_values = {
        value.casefold(): value
        for value in (
            source.primary_subject,
            *source.secondary_subjects,
            *source.projects,
            *source.recurring_themes,
        )
        if value.strip()
    }
    target_values = {
        value.casefold()
        for value in (
            target.primary_subject,
            *target.secondary_subjects,
            *target.projects,
            *target.recurring_themes,
        )
        if value.strip()
    }
    return tuple(source_values[key] for key in sorted(source_values.keys() & target_values)[:6])


def build_similarity_graph(
    embeddings: Sequence[EmbeddingRecord],
    *,
    options: SemanticAtlasOptions,
    analyses: Mapping[str, ConversationAnalysis] | None = None,
) -> tuple[GraphEdge, ...]:
    """Build a deterministic sparse undirected graph from provider embeddings."""

    _validate_embeddings(embeddings)
    if len(embeddings) < 2:
        return ()
    scored_by_node: defaultdict[int, list[tuple[float, int]]] = defaultdict(list)
    exact_neighbors_available = False
    if len(embeddings) > options.graph_brute_force_threshold:
        try:
            numpy = import_module("numpy")
            sklearn_neighbors = import_module("sklearn.neighbors")
            matrix = numpy.asarray(
                [record.vector for record in embeddings],
                dtype=numpy.float32,
            )
            neighbor_count = min(len(embeddings), options.max_neighbors + 1)
            estimator = sklearn_neighbors.NearestNeighbors(
                n_neighbors=neighbor_count,
                algorithm="brute",
                metric="cosine",
                n_jobs=-1,
            )
            distances, indices = estimator.fit(matrix).kneighbors(matrix)
            for left, (row_distances, row_indices) in enumerate(
                zip(distances, indices, strict=True)
            ):
                for distance, right_value in zip(row_distances, row_indices, strict=True):
                    right = int(right_value)
                    if left == right:
                        continue
                    similarity = 1.0 - float(distance)
                    if similarity >= options.min_similarity:
                        scored_by_node[left].append((similarity, right))
            exact_neighbors_available = True
        except (ImportError, AttributeError, TypeError, ValueError):
            exact_neighbors_available = False

    if not exact_neighbors_available:
        norms = [math.sqrt(sum(value * value for value in record.vector)) for record in embeddings]
        for left, right in _candidate_pairs(embeddings, options):
            similarity = _cosine(
                embeddings[left].vector,
                embeddings[right].vector,
                norms[left],
                norms[right],
            )
            if similarity >= options.min_similarity:
                scored_by_node[left].append((similarity, right))
                scored_by_node[right].append((similarity, left))

    selected_pairs: dict[tuple[int, int], float] = {}
    for index, candidates in scored_by_node.items():
        for similarity, neighbor in sorted(
            candidates,
            key=lambda item: (-item[0], embeddings[item[1]].conversation_key),
        )[: options.max_neighbors]:
            pair = (min(index, neighbor), max(index, neighbor))
            selected_pairs[pair] = max(similarity, selected_pairs.get(pair, -1.0))

    analysis_map = analyses or {}
    edges = [
        GraphEdge(
            source_key=embeddings[left].conversation_key,
            target_key=embeddings[right].conversation_key,
            similarity=max(-1.0, min(1.0, similarity)),
            shared_signals=_shared_signals(
                analysis_map.get(embeddings[left].conversation_key),
                analysis_map.get(embeddings[right].conversation_key),
            ),
        )
        for (left, right), similarity in selected_pairs.items()
    ]
    return tuple(sorted(edges, key=lambda edge: (edge.source_key, edge.target_key)))


def discover_communities(
    conversation_keys: Sequence[str],
    edges: Sequence[GraphEdge],
    *,
    resolution: float = 1.0,
    max_passes: int = 40,
) -> dict[str, str]:
    """Discover deterministic weighted communities using modularity local moving."""

    keys = tuple(sorted(set(conversation_keys)))
    adjacency: dict[str, dict[str, float]] = {key: {} for key in keys}
    for edge in edges:
        if edge.source_key not in adjacency or edge.target_key not in adjacency:
            raise SemanticAtlasError("A graph edge refers to an unknown conversation key.")
        adjacency[edge.source_key][edge.target_key] = edge.similarity
        adjacency[edge.target_key][edge.source_key] = edge.similarity

    labels = {key: key for key in keys}
    degrees = {key: sum(adjacency[key].values()) for key in keys}
    totals = dict(degrees)
    twice_total_weight = sum(degrees.values())
    if twice_total_weight > 0.0:
        for _ in range(max_passes):
            moved = False
            for key in keys:
                degree = degrees[key]
                if degree == 0.0:
                    continue
                old_label = labels[key]
                weights_by_label: defaultdict[str, float] = defaultdict(float)
                for neighbor, weight in adjacency[key].items():
                    weights_by_label[labels[neighbor]] += weight
                totals[old_label] -= degree
                candidates = set(weights_by_label)
                candidates.add(old_label)
                scored = [
                    (
                        weights_by_label[label]
                        - resolution * totals.get(label, 0.0) * degree / twice_total_weight,
                        label,
                    )
                    for label in candidates
                ]
                best_score, best_label = min(
                    scored,
                    key=lambda item: (-item[0], item[1]),
                )
                old_score = weights_by_label[old_label] - (
                    resolution * totals.get(old_label, 0.0) * degree / twice_total_weight
                )
                if best_score <= old_score + 1e-12:
                    best_label = old_label
                labels[key] = best_label
                totals[best_label] = totals.get(best_label, 0.0) + degree
                if best_label != old_label:
                    moved = True
            if not moved:
                break

    members_by_label: defaultdict[str, list[str]] = defaultdict(list)
    for key, label in labels.items():
        members_by_label[label].append(key)
    ordered_groups = sorted(
        (tuple(sorted(members)) for members in members_by_label.values()),
        key=lambda members: members[0],
    )
    return {
        key: f"category-{index:04d}"
        for index, members in enumerate(ordered_groups, start=1)
        for key in members
    }


def _normalized_community_centroids(
    community_members: Sequence[tuple[str, tuple[str, ...]]],
    vectors_by_key: Mapping[str, tuple[float, ...]],
    dimensions: int,
) -> tuple[tuple[float, ...], ...]:
    """Return unit-normalized mean vectors in stable community order."""

    centroids: list[tuple[float, ...]] = []
    for _, members in community_members:
        totals = [0.0] * dimensions
        for key in members:
            vector = vectors_by_key[key]
            for index, value in enumerate(vector):
                totals[index] += value
        centroid = tuple(value / len(members) for value in totals)
        norm = math.sqrt(sum(value * value for value in centroid))
        if norm:
            centroid = tuple(value / norm for value in centroid)
        centroids.append(centroid)
    return tuple(centroids)


def _fallback_centroid_groups(
    centroids: Sequence[tuple[float, ...]],
    target_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Cluster normalized centroids deterministically when numeric extras are absent.

    Farthest-first seeds bound fallback work to ``O(communities * target_count)``;
    each remaining community joins its closest seed with index-based tie-breaking.
    """

    if len(centroids) <= target_count:
        return tuple((index,) for index in range(len(centroids)))

    def similarity(left: int, right: int) -> float:
        return sum(
            first * second for first, second in zip(centroids[left], centroids[right], strict=True)
        )

    seeds = [0]
    remaining = set(range(1, len(centroids)))
    while len(seeds) < target_count:
        next_seed = min(
            remaining,
            key=lambda candidate: (
                max(similarity(candidate, seed) for seed in seeds),
                candidate,
            ),
        )
        seeds.append(next_seed)
        remaining.remove(next_seed)

    groups: dict[int, list[int]] = {seed: [seed] for seed in seeds}
    for candidate in sorted(remaining):
        closest_seed = min(
            seeds,
            key=lambda seed: (-similarity(candidate, seed), seed),
        )
        groups[closest_seed].append(candidate)
    return tuple(tuple(sorted(groups[seed])) for seed in seeds)


def _sklearn_centroid_groups(
    centroids: Sequence[tuple[float, ...]],
    target_count: int,
) -> tuple[tuple[int, ...], ...] | None:
    """Use optional agglomerative clustering and return ``None`` when unavailable."""

    try:
        numpy = import_module("numpy")
        sklearn_cluster = import_module("sklearn.cluster")
        matrix = numpy.asarray(centroids, dtype=numpy.float64)
        estimator = sklearn_cluster.AgglomerativeClustering(
            n_clusters=target_count,
            metric="euclidean",
            linkage="average",
        )
        labels = estimator.fit_predict(matrix)
    except (ImportError, AttributeError, TypeError, ValueError):
        return None

    groups: defaultdict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[int(label)].append(index)
    return tuple(tuple(sorted(indices)) for indices in groups.values())


def consolidate_communities(
    memberships: Mapping[str, str],
    embeddings: Sequence[EmbeddingRecord],
    *,
    max_categories: int,
) -> CommunityConsolidation:
    """Consolidate discovered communities to a deterministic maximum category count.

    Community means are normalized before clustering so category size and embedding
    magnitude do not dominate proximity. Final IDs are assigned by the smallest
    conversation key in each merged group, independent of provider or input order.
    """

    if max_categories < 1:
        raise ValueError("max_categories must be at least one.")
    dimensions = _validate_embeddings(embeddings)
    embedding_keys = {record.conversation_key for record in embeddings}
    if embedding_keys != set(memberships):
        raise SemanticAtlasError(
            "Community memberships and embedding records do not cover the same conversations."
        )
    if not memberships:
        return CommunityConsolidation(memberships={}, source_community_counts={})

    members_by_id: defaultdict[str, list[str]] = defaultdict(list)
    for key, category_id in memberships.items():
        members_by_id[category_id].append(key)
    community_members = tuple(
        sorted(
            (
                (category_id, tuple(sorted(members)))
                for category_id, members in members_by_id.items()
            ),
            key=lambda item: (item[1][0], item[0]),
        )
    )
    if len(community_members) <= max_categories:
        counts = {category_id: 1 for category_id, _ in community_members}
        return CommunityConsolidation(
            memberships=dict(memberships),
            source_community_counts=counts,
        )

    vectors_by_key = {record.conversation_key: record.vector for record in embeddings}
    centroids = _normalized_community_centroids(
        community_members,
        vectors_by_key,
        dimensions,
    )
    groups = _sklearn_centroid_groups(centroids, max_categories)
    if groups is None:
        groups = _fallback_centroid_groups(centroids, max_categories)

    merged_groups = [
        (
            tuple(
                sorted(
                    key
                    for community_index in group
                    for key in community_members[community_index][1]
                )
            ),
            len(group),
        )
        for group in groups
    ]
    merged_groups.sort(key=lambda item: item[0][0])
    final_memberships: dict[str, str] = {}
    source_counts: dict[str, int] = {}
    for index, (members, source_count) in enumerate(merged_groups, start=1):
        category_id = f"category-{index:04d}"
        for key in members:
            final_memberships[key] = category_id
        source_counts[category_id] = source_count
    return CommunityConsolidation(
        memberships=final_memberships,
        source_community_counts=source_counts,
    )
