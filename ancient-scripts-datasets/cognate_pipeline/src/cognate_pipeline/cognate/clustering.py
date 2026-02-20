"""Clustering algorithms for cognate set formation."""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict

from cognate_pipeline.cognate.models import CognateLink, CognateSet, CognateSetMember
from cognate_pipeline.config.schema import ClusteringAlgorithm

logger = logging.getLogger(__name__)


def cluster_links(
    links: list[CognateLink],
    algorithm: ClusteringAlgorithm = ClusteringAlgorithm.CONNECTED_COMPONENTS,
) -> list[CognateSet]:
    """Cluster pairwise CognateLinks into CognateSets."""
    if algorithm == ClusteringAlgorithm.CONNECTED_COMPONENTS:
        return _connected_components(links)
    elif algorithm == ClusteringAlgorithm.UPGMA:
        return _upgma(links)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")


def _connected_components(links: list[CognateLink]) -> list[CognateSet]:
    """Form cognate sets via connected components on the link graph."""
    if not links:
        return []

    # Build adjacency list grouped by concept
    by_concept: dict[str, list[CognateLink]] = defaultdict(list)
    for link in links:
        by_concept[link.concept_id].append(link)

    sets: list[CognateSet] = []

    for concept_id, concept_links in by_concept.items():
        # Union-Find
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for link in concept_links:
            parent.setdefault(link.lexeme_id_a, link.lexeme_id_a)
            parent.setdefault(link.lexeme_id_b, link.lexeme_id_b)
            union(link.lexeme_id_a, link.lexeme_id_b)

        # Collect components
        components: dict[str, list[str]] = defaultdict(list)
        for node in parent:
            components[find(node)].append(node)

        for members in components.values():
            if len(members) < 2:
                continue
            set_id = f"cs_{uuid.uuid4().hex[:12]}"
            sets.append(
                CognateSet(
                    id=set_id,
                    concept_id=concept_id,
                    method="connected_components",
                    members=[
                        CognateSetMember(lexeme_id=m) for m in sorted(members)
                    ],
                    quality={"size": len(members)},
                )
            )

    logger.info("Formed %d cognate sets via connected components", len(sets))
    return sets


def _upgma(links: list[CognateLink]) -> list[CognateSet]:
    """Form cognate sets via UPGMA hierarchical clustering.

    Uses a simple agglomerative approach: iteratively merge the closest
    pair of clusters based on average link score.
    """
    if not links:
        return []

    by_concept: dict[str, list[CognateLink]] = defaultdict(list)
    for link in links:
        by_concept[link.concept_id].append(link)

    sets: list[CognateSet] = []

    for concept_id, concept_links in by_concept.items():
        # Build distance matrix
        nodes: set[str] = set()
        score_map: dict[tuple[str, str], float] = {}
        for link in concept_links:
            nodes.add(link.lexeme_id_a)
            nodes.add(link.lexeme_id_b)
            key = (
                min(link.lexeme_id_a, link.lexeme_id_b),
                max(link.lexeme_id_a, link.lexeme_id_b),
            )
            score_map[key] = link.score

        if len(nodes) < 2:
            continue

        # Initialize clusters
        clusters: dict[str, list[str]] = {n: [n] for n in nodes}

        while len(clusters) > 1:
            # Find best merge (highest average score)
            best_score = -1.0
            best_pair = None
            cluster_ids = list(clusters.keys())
            for i in range(len(cluster_ids)):
                for j in range(i + 1, len(cluster_ids)):
                    ci, cj = cluster_ids[i], cluster_ids[j]
                    # Average score between clusters
                    total = 0.0
                    count = 0
                    for a in clusters[ci]:
                        for b in clusters[cj]:
                            key = (min(a, b), max(a, b))
                            if key in score_map:
                                total += score_map[key]
                                count += 1
                    avg = total / count if count > 0 else 0.0
                    if avg > best_score:
                        best_score = avg
                        best_pair = (ci, cj)

            if best_pair is None or best_score <= 0:
                break

            # Merge
            ci, cj = best_pair
            merged = clusters[ci] + clusters[cj]
            new_id = f"{ci}+{cj}"
            del clusters[ci]
            del clusters[cj]
            clusters[new_id] = merged

        for members in clusters.values():
            if len(members) < 2:
                continue
            set_id = f"cs_{uuid.uuid4().hex[:12]}"
            sets.append(
                CognateSet(
                    id=set_id,
                    concept_id=concept_id,
                    method="upgma",
                    members=[
                        CognateSetMember(lexeme_id=m) for m in sorted(members)
                    ],
                    quality={"size": len(members)},
                )
            )

    logger.info("Formed %d cognate sets via UPGMA", len(sets))
    return sets
