"""Tests for graph-edge pruning."""

import numpy as np

from clustkit.graph import build_similarity_graph, prune_bridge_edges


def test_prune_bridge_edges_removes_unsupported_weak_edge():
    pairs = np.array([[0, 1], [1, 2], [0, 2], [2, 3]], dtype=np.int32)
    sims = np.array([0.9, 0.9, 0.9, 0.51], dtype=np.float32)
    graph = build_similarity_graph(4, pairs, sims)

    pruned = prune_bridge_edges(
        graph,
        weak_similarity_threshold=0.6,
        min_shared_neighbors=1,
    )

    assert pruned[2, 3] == 0
    assert pruned[0, 1] > 0


def test_prune_bridge_edges_keeps_supported_weak_edge():
    pairs = np.array([[0, 1], [1, 2], [0, 2], [0, 3], [1, 3]], dtype=np.int32)
    sims = np.array([0.9, 0.9, 0.9, 0.55, 0.55], dtype=np.float32)
    graph = build_similarity_graph(4, pairs, sims)

    pruned = prune_bridge_edges(
        graph,
        weak_similarity_threshold=0.6,
        min_shared_neighbors=1,
    )

    assert pruned[0, 3] > 0
    assert pruned[1, 3] > 0


def test_prune_bridge_edges_keeps_leaf_edge_below_min_degree():
    pairs = np.array([[0, 1], [1, 2]], dtype=np.int32)
    sims = np.array([0.55, 0.95], dtype=np.float32)
    graph = build_similarity_graph(3, pairs, sims)

    pruned = prune_bridge_edges(
        graph,
        weak_similarity_threshold=0.6,
        min_shared_neighbors=1,
        min_endpoint_degree=2,
    )

    assert pruned[0, 1] > 0
