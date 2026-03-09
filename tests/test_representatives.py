"""Tests for Phase 6: Representative Selection."""

import numpy as np
import pytest
from scipy import sparse

from clustkit.graph import build_similarity_graph
from clustkit.representatives import select_representatives


# ---------------------------------------------------------------------------
# Helper to build a small graph quickly
# ---------------------------------------------------------------------------
def _make_graph(num_sequences, pairs, sims):
    return build_similarity_graph(
        num_sequences,
        np.array(pairs, dtype=np.int32),
        np.array(sims, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Tests for "longest" (existing behaviour — must not break)
# ---------------------------------------------------------------------------
class TestLongest:
    def test_basic(self):
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        lengths = np.array([100, 200, 150, 50], dtype=np.int32)
        reps = select_representatives(labels, lengths, method="longest")
        assert reps[0] == 1  # longest in cluster 0
        assert reps[1] == 2  # longest in cluster 1

    def test_ties_broken_by_lowest_index(self):
        labels = np.array([0, 0, 0], dtype=np.int32)
        lengths = np.array([100, 100, 100], dtype=np.int32)
        reps = select_representatives(labels, lengths, method="longest")
        assert reps[0] == 0  # all equal length — first wins

    def test_does_not_require_graph(self):
        labels = np.array([0], dtype=np.int32)
        lengths = np.array([42], dtype=np.int32)
        # graph=None is the default and should work fine for "longest"
        reps = select_representatives(labels, lengths, method="longest")
        assert reps[0] == 0


# ---------------------------------------------------------------------------
# Tests for "centroid"
# ---------------------------------------------------------------------------
class TestCentroid:
    def test_requires_graph(self):
        labels = np.array([0, 0], dtype=np.int32)
        lengths = np.array([100, 100], dtype=np.int32)
        with pytest.raises(ValueError, match="requires a similarity graph"):
            select_representatives(labels, lengths, method="centroid")

    def test_singleton_cluster(self):
        """A cluster with a single member should select that member."""
        graph = _make_graph(1, [], [])
        labels = np.array([0], dtype=np.int32)
        lengths = np.array([50], dtype=np.int32)
        reps = select_representatives(labels, lengths, method="centroid", graph=graph)
        assert reps[0] == 0

    def test_picks_highest_avg_similarity(self):
        """Node 1 is connected to both 0 and 2 with high similarity;
        nodes 0 and 2 are only connected to node 1.  Node 1 should be
        the centroid because it has the highest average similarity."""
        # Edges: 0-1 (0.9), 1-2 (0.9)
        graph = _make_graph(3, [[0, 1], [1, 2]], [0.9, 0.9])
        labels = np.array([0, 0, 0], dtype=np.int32)
        lengths = np.array([100, 100, 100], dtype=np.int32)
        reps = select_representatives(labels, lengths, method="centroid", graph=graph)
        assert reps[0] == 1

    def test_centroid_with_weighted_edges(self):
        """When edge weights differ, the centroid should reflect the weights."""
        # 3 nodes in one cluster.
        # Edges: 0-1 (0.5), 0-2 (0.5), 1-2 (0.99)
        # Avg sim: node0 = (0.5+0.5)/2 = 0.5
        #          node1 = (0.5+0.99)/2 = 0.745
        #          node2 = (0.5+0.99)/2 = 0.745
        # Tie between 1 and 2 — argmax picks node 1 (first).
        graph = _make_graph(3, [[0, 1], [0, 2], [1, 2]], [0.5, 0.5, 0.99])
        labels = np.array([0, 0, 0], dtype=np.int32)
        lengths = np.array([100, 100, 100], dtype=np.int32)
        reps = select_representatives(labels, lengths, method="centroid", graph=graph)
        assert reps[0] == 1

    def test_multiple_clusters(self):
        """Two clusters in the same graph — each should get its own centroid."""
        # Cluster 0: nodes 0, 1, 2.  Edges: 0-1 (0.9), 0-2 (0.9), 1-2 (0.5)
        #   avg sim: node0 = (0.9+0.9)/2 = 0.9  <-- centroid
        #            node1 = (0.9+0.5)/2 = 0.7
        #            node2 = (0.9+0.5)/2 = 0.7
        # Cluster 1: nodes 3, 4.  Edge: 3-4 (0.8)
        #   avg sim: node3 = 0.8, node4 = 0.8  -> tie, argmax picks 3
        graph = _make_graph(
            5,
            [[0, 1], [0, 2], [1, 2], [3, 4]],
            [0.9, 0.9, 0.5, 0.8],
        )
        labels = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        lengths = np.array([100, 100, 100, 100, 100], dtype=np.int32)
        reps = select_representatives(labels, lengths, method="centroid", graph=graph)
        assert reps[0] == 0
        assert reps[1] == 3


# ---------------------------------------------------------------------------
# Tests for "most_connected"
# ---------------------------------------------------------------------------
class TestMostConnected:
    def test_requires_graph(self):
        labels = np.array([0, 0], dtype=np.int32)
        lengths = np.array([100, 100], dtype=np.int32)
        with pytest.raises(ValueError, match="requires a similarity graph"):
            select_representatives(labels, lengths, method="most_connected")

    def test_singleton_cluster(self):
        graph = _make_graph(1, [], [])
        labels = np.array([0], dtype=np.int32)
        lengths = np.array([50], dtype=np.int32)
        reps = select_representatives(
            labels, lengths, method="most_connected", graph=graph
        )
        assert reps[0] == 0

    def test_picks_highest_degree(self):
        """Node 1 has degree 2 (connected to 0 and 2); others have degree 1."""
        graph = _make_graph(3, [[0, 1], [1, 2]], [0.9, 0.9])
        labels = np.array([0, 0, 0], dtype=np.int32)
        lengths = np.array([100, 100, 100], dtype=np.int32)
        reps = select_representatives(
            labels, lengths, method="most_connected", graph=graph
        )
        assert reps[0] == 1

    def test_degree_tie_broken_by_lowest_index(self):
        """All nodes have degree 1 — tie broken by lowest index."""
        # Linear chain: 0-1, 2-3 but all in one cluster
        # Actually let's make a triangle so all have degree 2
        graph = _make_graph(3, [[0, 1], [1, 2], [0, 2]], [0.9, 0.9, 0.9])
        labels = np.array([0, 0, 0], dtype=np.int32)
        lengths = np.array([100, 100, 100], dtype=np.int32)
        reps = select_representatives(
            labels, lengths, method="most_connected", graph=graph
        )
        # All have same degree — first (index 0) wins
        assert reps[0] == 0

    def test_multiple_clusters(self):
        """Two clusters — each picks its own most-connected node."""
        # Cluster 0: nodes 0, 1, 2.  Node 0 connects to both 1 and 2.
        # Cluster 1: nodes 3, 4.     3-4 edge.
        graph = _make_graph(
            5,
            [[0, 1], [0, 2], [3, 4]],
            [0.9, 0.9, 0.9],
        )
        labels = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        lengths = np.array([50, 50, 50, 50, 50], dtype=np.int32)
        reps = select_representatives(
            labels, lengths, method="most_connected", graph=graph
        )
        assert reps[0] == 0  # degree 2, others have degree 1
        assert reps[1] == 3  # tie, first wins

    def test_cross_cluster_edges_ignored(self):
        """Edges between clusters should not affect within-cluster degree."""
        # Nodes 0,1 in cluster 0;  Nodes 2,3 in cluster 1.
        # Edges: 0-1 (intra), 0-2 (cross), 2-3 (intra)
        # Within cluster 0: node 0 degree 1, node 1 degree 1 -> tie -> node 0
        # The cross-edge 0-2 should NOT inflate node 0's within-cluster degree.
        graph = _make_graph(4, [[0, 1], [0, 2], [2, 3]], [0.9, 0.9, 0.9])
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        lengths = np.array([50, 50, 50, 50], dtype=np.int32)
        reps = select_representatives(
            labels, lengths, method="most_connected", graph=graph
        )
        # Both within-cluster degrees are 1 in each cluster — first member wins
        assert reps[0] == 0
        assert reps[1] == 2


# ---------------------------------------------------------------------------
# Test for unknown method
# ---------------------------------------------------------------------------
class TestUnknownMethod:
    def test_raises_value_error(self):
        labels = np.array([0], dtype=np.int32)
        lengths = np.array([10], dtype=np.int32)
        with pytest.raises(ValueError, match="Unknown representative selection method"):
            select_representatives(labels, lengths, method="invalid")
