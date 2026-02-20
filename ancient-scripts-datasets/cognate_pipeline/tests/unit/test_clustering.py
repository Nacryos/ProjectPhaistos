"""Tests for cognate clustering algorithms."""

from __future__ import annotations

from cognate_pipeline.cognate.clustering import cluster_links
from cognate_pipeline.cognate.models import CognateLink
from cognate_pipeline.config.schema import ClusteringAlgorithm


def _link(a: str, b: str, concept: str, score: float = 0.8) -> CognateLink:
    return CognateLink(
        lexeme_id_a=min(a, b),
        lexeme_id_b=max(a, b),
        concept_id=concept,
        score=score,
        method="test",
    )


class TestConnectedComponents:
    def test_single_pair(self):
        links = [_link("a", "b", "water")]
        sets = cluster_links(links, ClusteringAlgorithm.CONNECTED_COMPONENTS)
        assert len(sets) == 1
        assert len(sets[0].members) == 2

    def test_transitive_closure(self):
        """a-b and b-c should form one set {a, b, c}."""
        links = [
            _link("a", "b", "water"),
            _link("b", "c", "water"),
        ]
        sets = cluster_links(links, ClusteringAlgorithm.CONNECTED_COMPONENTS)
        assert len(sets) == 1
        member_ids = {m.lexeme_id for m in sets[0].members}
        assert member_ids == {"a", "b", "c"}

    def test_separate_concepts(self):
        links = [
            _link("a", "b", "water"),
            _link("c", "d", "fire"),
        ]
        sets = cluster_links(links, ClusteringAlgorithm.CONNECTED_COMPONENTS)
        assert len(sets) == 2

    def test_empty_links(self):
        sets = cluster_links([], ClusteringAlgorithm.CONNECTED_COMPONENTS)
        assert sets == []

    def test_set_metadata(self):
        links = [_link("a", "b", "water")]
        sets = cluster_links(links, ClusteringAlgorithm.CONNECTED_COMPONENTS)
        assert sets[0].method == "connected_components"
        assert sets[0].concept_id == "water"
        assert sets[0].quality["size"] == 2


class TestUPGMA:
    def test_single_pair(self):
        links = [_link("a", "b", "water", score=0.9)]
        sets = cluster_links(links, ClusteringAlgorithm.UPGMA)
        assert len(sets) == 1
        assert len(sets[0].members) == 2

    def test_three_nodes(self):
        links = [
            _link("a", "b", "water", score=0.9),
            _link("b", "c", "water", score=0.8),
            _link("a", "c", "water", score=0.7),
        ]
        sets = cluster_links(links, ClusteringAlgorithm.UPGMA)
        assert len(sets) == 1
        member_ids = {m.lexeme_id for m in sets[0].members}
        assert member_ids == {"a", "b", "c"}

    def test_empty(self):
        sets = cluster_links([], ClusteringAlgorithm.UPGMA)
        assert sets == []

    def test_roundtrip(self):
        links = [_link("a", "b", "water")]
        sets = cluster_links(links, ClusteringAlgorithm.UPGMA)
        d = sets[0].to_dict()
        from cognate_pipeline.cognate.models import CognateSet
        restored = CognateSet.from_dict(d)
        assert len(restored.members) == 2
