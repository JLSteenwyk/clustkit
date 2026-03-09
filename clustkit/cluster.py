"""Phase 5: Clustering — Parallel connected components / graph clustering (CPU reference)."""

import numpy as np
from scipy import sparse


def _leiden_clustering(graph: sparse.csr_matrix, resolution: float = 1.0) -> np.ndarray:
    """Cluster sequences using the Leiden algorithm.

    Args:
        graph: (N, N) sparse CSR similarity graph.
        resolution: Resolution parameter for Leiden (higher = more clusters).

    Returns:
        (N,) int32 array of cluster labels.
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        raise ImportError(
            "Leiden clustering requires the 'leidenalg' and 'python-igraph' packages. "
            "Install them with: pip install clustkit[leiden]"
        )

    n = graph.shape[0]

    # Ensure the graph is symmetric and extract upper triangle to avoid duplicate edges
    sym = sparse.triu(graph, format="coo")

    # Build igraph Graph from edge list and weights
    edges = list(zip(sym.row.tolist(), sym.col.tolist()))
    weights = sym.data.tolist()

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
    )

    labels = np.array(partition.membership, dtype=np.int32)
    return labels


def cluster_connected_components(graph: sparse.csr_matrix) -> np.ndarray:
    """Cluster sequences using connected components of the similarity graph.

    Args:
        graph: (N, N) sparse CSR similarity graph.

    Returns:
        (N,) int32 array of cluster labels (0-indexed).
    """
    n_components, labels = sparse.csgraph.connected_components(
        graph, directed=False, return_labels=True
    )
    return labels.astype(np.int32)


def cluster_greedy(
    graph: sparse.csr_matrix,
    lengths: np.ndarray,
) -> np.ndarray:
    """Greedy graph clustering: high-degree nodes become representatives first.

    Processes nodes by descending degree. Each unassigned node with neighbors
    becomes a representative and claims its unassigned neighbors.

    Args:
        graph: (N, N) sparse CSR similarity graph.
        lengths: (N,) sequence lengths (used for tiebreaking).

    Returns:
        (N,) int32 array of cluster labels.
    """
    n = graph.shape[0]
    degrees = np.array(graph.getnnz(axis=1)).flatten()

    # Sort by degree descending, then by length descending for tiebreaking
    order = np.lexsort((-lengths, -degrees))

    labels = np.full(n, -1, dtype=np.int32)
    next_label = 0

    for node in order:
        if labels[node] != -1:
            continue

        # This node becomes a representative
        labels[node] = next_label

        # Assign unassigned neighbors to this cluster
        neighbors = graph[node].indices
        for nb in neighbors:
            if labels[nb] == -1:
                labels[nb] = next_label

        next_label += 1

    return labels


def cluster_sequences(
    graph: sparse.csr_matrix,
    method: str = "connected",
    lengths: np.ndarray | None = None,
) -> np.ndarray:
    """Dispatch to the requested clustering method.

    Args:
        graph: (N, N) sparse CSR similarity graph.
        method: "connected", "greedy", or "leiden".
        lengths: Sequence lengths (required for greedy method).

    Returns:
        (N,) int32 array of cluster labels.
    """
    if method == "connected":
        return cluster_connected_components(graph)
    elif method == "greedy":
        if lengths is None:
            raise ValueError("Greedy clustering requires sequence lengths.")
        return cluster_greedy(graph, lengths)
    elif method == "leiden":
        return _leiden_clustering(graph)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
