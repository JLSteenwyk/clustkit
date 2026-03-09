"""Phase 6: Representative Selection — Choose one representative per cluster."""

import numpy as np
from scipy import sparse


def select_representatives(
    cluster_labels: np.ndarray,
    lengths: np.ndarray,
    method: str = "longest",
    graph: sparse.csr_matrix | None = None,
) -> np.ndarray:
    """Select a representative sequence for each cluster.

    Args:
        cluster_labels: (N,) int32 array of cluster labels.
        lengths: (N,) int32 array of sequence lengths.
        method: Selection strategy — "longest", "centroid", or "most_connected".
        graph: (N, N) sparse CSR similarity matrix. Required for "centroid"
            and "most_connected" methods.

    Returns:
        1D int32 array of representative sequence indices (one per cluster).
    """
    if method not in ("longest", "centroid", "most_connected"):
        raise ValueError(f"Unknown representative selection method: {method}")

    if method in ("centroid", "most_connected") and graph is None:
        raise ValueError(
            f"Representative method '{method}' requires a similarity graph. "
            "Pass the graph argument."
        )

    unique_labels = np.unique(cluster_labels)
    representatives = np.empty(len(unique_labels), dtype=np.int32)

    for idx, label in enumerate(unique_labels):
        members = np.where(cluster_labels == label)[0]

        if method == "longest":
            # Pick the longest sequence; ties broken by lowest index
            member_lengths = lengths[members]
            best_local = np.argmax(member_lengths)
            representatives[idx] = members[best_local]

        elif method == "centroid":
            # Pick the sequence with the highest average similarity to all
            # other members in the cluster.
            if len(members) == 1:
                representatives[idx] = members[0]
            else:
                # Extract the sub-matrix for this cluster's members
                sub = graph[np.ix_(members, members)]
                # Average similarity per member (exclude self — diagonal is 0
                # in this graph, but divide by (n-1) to average over others)
                avg_sim = np.asarray(sub.sum(axis=1)).ravel() / (len(members) - 1)
                best_local = np.argmax(avg_sim)
                representatives[idx] = members[best_local]

        elif method == "most_connected":
            # Pick the sequence with the most edges (highest degree) within
            # the cluster.  Ties broken by lowest index (np.argmax returns
            # the first occurrence).
            if len(members) == 1:
                representatives[idx] = members[0]
            else:
                sub = graph[np.ix_(members, members)]
                # Degree = number of non-zero entries per row
                degrees = np.asarray(sub.getnnz(axis=1)).ravel()
                best_local = np.argmax(degrees)
                representatives[idx] = members[best_local]

    return representatives
