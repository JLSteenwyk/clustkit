"""Phase 3: Pairwise Similarity — Compute identity for candidate pairs (CPU reference)."""

import numpy as np


def jaccard_from_sketches(
    sketch_a: np.ndarray,
    sketch_b: np.ndarray,
) -> float:
    """Estimate Jaccard similarity from two sorted MinHash sketches.

    Uses the merge-based MinHash Jaccard estimator:
    count shared hashes / total unique hashes in the union of bottom-s.
    """
    s = len(sketch_a)
    max_val = np.iinfo(np.uint64).max

    shared = 0
    i, j = 0, 0
    union_count = 0

    while union_count < s and i < s and j < s:
        a_val = sketch_a[i]
        b_val = sketch_b[j]

        # Skip padding values
        if a_val == max_val and b_val == max_val:
            break

        if a_val == b_val:
            shared += 1
            i += 1
            j += 1
        elif a_val < b_val:
            i += 1
        else:
            j += 1
        union_count += 1

    if union_count == 0:
        return 0.0

    return shared / union_count


def compute_pairwise_jaccard(
    candidate_pairs: np.ndarray,
    sketches: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Jaccard similarity for all candidate pairs and filter by threshold.

    Args:
        candidate_pairs: (M, 2) int32 array of candidate pairs (i, j).
        sketches: (N, sketch_size) uint64 array of MinHash sketches.
        threshold: Minimum Jaccard similarity to keep a pair.

    Returns:
        Tuple of:
        - filtered_pairs: (K, 2) int32 array of pairs above threshold.
        - similarities: (K,) float32 array of Jaccard similarities.
    """
    if len(candidate_pairs) == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float32)

    kept_pairs = []
    kept_sims = []

    for pair_idx in range(len(candidate_pairs)):
        i, j = candidate_pairs[pair_idx]
        sim = jaccard_from_sketches(sketches[i], sketches[j])
        if sim >= threshold:
            kept_pairs.append((i, j))
            kept_sims.append(sim)

    if not kept_pairs:
        return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float32)

    return (
        np.array(kept_pairs, dtype=np.int32),
        np.array(kept_sims, dtype=np.float32),
    )
