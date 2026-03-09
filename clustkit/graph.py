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
