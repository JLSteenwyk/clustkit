"""Phase 4: Graph Construction — Build sparse similarity graph (CPU reference)."""

import numpy as np
from scipy import sparse


def build_similarity_graph(
    num_sequences: int,
    filtered_pairs: np.ndarray,
    similarities: np.ndarray,
) -> sparse.csr_matrix:
    """Build a sparse symmetric similarity graph in CSR format.

    Args:
        num_sequences: Total number of sequences (N).
        filtered_pairs: (K, 2) int32 array of edges (i, j).
        similarities: (K,) float32 array of edge weights.

    Returns:
        (N, N) sparse CSR matrix with symmetric edges weighted by similarity.
    """
    if len(filtered_pairs) == 0:
        return sparse.csr_matrix((num_sequences, num_sequences), dtype=np.float32)

    rows = filtered_pairs[:, 0]
    cols = filtered_pairs[:, 1]

    # Make symmetric: add both (i,j) and (j,i)
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    all_data = np.concatenate([similarities, similarities])

    graph = sparse.coo_matrix(
        (all_data, (all_rows, all_cols)),
        shape=(num_sequences, num_sequences),
        dtype=np.float32,
    )

    return graph.tocsr()


def prune_bridge_edges(
    graph: sparse.csr_matrix,
    weak_similarity_threshold: float,
    min_shared_neighbors: int,
    min_endpoint_degree: int = 0,
) -> sparse.csr_matrix:
    """Prune weak bridge edges that are not supported by shared neighbors."""
    if graph.nnz == 0 or min_shared_neighbors <= 0:
        return graph

    upper = sparse.triu(graph, k=1, format="coo")
    if upper.nnz == 0:
        return graph

    weak_mask = upper.data < weak_similarity_threshold
    if not np.any(weak_mask):
        return graph

    binary = graph.copy()
    binary.data = np.ones_like(binary.data, dtype=np.int8)
    shared = (binary @ binary).tocsr()
    degrees = np.asarray(binary.getnnz(axis=1)).ravel()

    keep = np.ones(upper.nnz, dtype=np.bool_)
    weak_idx = np.where(weak_mask)[0]
    for idx in weak_idx:
        src = upper.row[idx]
        dst = upper.col[idx]
        if min(degrees[src], degrees[dst]) <= min_endpoint_degree:
            continue
        if shared[src, dst] < min_shared_neighbors:
            keep[idx] = False

    kept = sparse.coo_matrix(
        (upper.data[keep], (upper.row[keep], upper.col[keep])),
        shape=graph.shape,
        dtype=np.float32,
    )
    pruned = kept + kept.T
    return pruned.tocsr()
