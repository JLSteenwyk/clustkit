"""Phase 6: Representative Selection — Choose one representative per cluster."""

import numpy as np


def select_representatives(
    cluster_labels: np.ndarray,
    lengths: np.ndarray,
    method: str = "longest",
) -> np.ndarray:
    """Select a representative sequence for each cluster.

    Args:
        cluster_labels: (N,) int32 array of cluster labels.
        lengths: (N,) int32 array of sequence lengths.
        method: Selection strategy — "longest", "centroid", or "most_connected".
            Currently only "longest" is implemented for CPU reference.

    Returns:
        1D int32 array of representative sequence indices (one per cluster).
    """
    if method not in ("longest", "centroid", "most_connected"):
        raise ValueError(f"Unknown representative selection method: {method}")

    if method in ("centroid", "most_connected"):
        raise NotImplementedError(
            f"Representative method '{method}' requires pairwise similarity data. "
            "Falling back is not supported yet. Use --representative longest."
        )

    unique_labels = np.unique(cluster_labels)
    representatives = np.empty(len(unique_labels), dtype=np.int32)

    for idx, label in enumerate(unique_labels):
        members = np.where(cluster_labels == label)[0]
        # Pick the longest sequence; ties broken by lowest index
        member_lengths = lengths[members]
        best_local = np.argmax(member_lengths)
        representatives[idx] = members[best_local]

    return representatives
