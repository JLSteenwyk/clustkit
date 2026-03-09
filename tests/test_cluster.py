"""Tests for Phase 5: Clustering and Phase 4: Graph construction."""

import numpy as np
import pytest
from scipy import sparse

from clustkit.cluster import cluster_connected_components, cluster_greedy, cluster_sequences
from clustkit.graph import build_similarity_graph


class TestBuildSimilarityGraph:
    def test_basic_graph(self):
        pairs = np.array([[0, 1], [1, 2]], dtype=np.int32)
        sims = np.array([0.95, 0.90], dtype=np.float32)
        graph = build_similarity_graph(3, pairs, sims)

        assert graph.shape == (3, 3)
        assert graph.nnz == 4  # symmetric: each edge stored twice
        assert abs(graph[0, 1] - 0.95) < 1e-6
        assert abs(graph[1, 0] - 0.95) < 1e-6

    def test_empty_graph(self):
        pairs = np.empty((0, 2), dtype=np.int32)
        sims = np.empty(0, dtype=np.float32)
        graph = build_similarity_graph(5, pairs, sims)
        assert graph.shape == (5, 5)
        assert graph.nnz == 0

    def test_single_edge(self):
        pairs = np.array([[0, 4]], dtype=np.int32)
        sims = np.array([0.85], dtype=np.float32)
        graph = build_similarity_graph(5, pairs, sims)
        assert abs(graph[0, 4] - 0.85) < 1e-6
        assert abs(graph[4, 0] - 0.85) < 1e-6
        assert graph[0, 1] == 0.0


class TestClusterConnectedComponents:
    def test_two_components(self):
        # Component 1: {0, 1, 2}, Component 2: {3, 4}
        pairs = np.array([[0, 1], [1, 2], [3, 4]], dtype=np.int32)
        sims = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        graph = build_similarity_graph(5, pairs, sims)

        labels = cluster_connected_components(graph)
        assert len(labels) == 5
        # 0, 1, 2 should share a label
        assert labels[0] == labels[1] == labels[2]
        # 3, 4 should share a different label
        assert labels[3] == labels[4]
        assert labels[0] != labels[3]

    def test_all_singletons(self):
        graph = sparse.csr_matrix((5, 5), dtype=np.float32)
        labels = cluster_connected_components(graph)
        assert len(np.unique(labels)) == 5

    def test_single_cluster(self):
        pairs = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
        sims = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        graph = build_similarity_graph(4, pairs, sims)
        labels = cluster_connected_components(graph)
        assert len(np.unique(labels)) == 1


class TestClusterGreedy:
    def test_basic_greedy(self):
        pairs = np.array([[0, 1], [0, 2], [3, 4]], dtype=np.int32)
        sims = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        graph = build_similarity_graph(5, pairs, sims)
        lengths = np.array([100, 50, 50, 80, 40], dtype=np.int32)

        labels = cluster_greedy(graph, lengths)
        assert len(labels) == 5
        # Node 0 has highest degree (2), should be a rep
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4]


class TestClusterDispatch:
    def test_unknown_method_raises(self):
        graph = sparse.csr_matrix((3, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown clustering method"):
            cluster_sequences(graph, method="invalid")

    def test_leiden_not_implemented(self):
        graph = sparse.csr_matrix((3, 3), dtype=np.float32)
        with pytest.raises(NotImplementedError):
            cluster_sequences(graph, method="leiden")
