"""Phase 3: Pairwise Similarity — Compute identity for candidate pairs.

Uses Numba JIT for the Jaccard kernel; batch processing in parallel.
"""

import numpy as np
from numba import njit, prange, uint64, float32


@njit(float32(uint64[:], uint64[:]), cache=True)
def _jaccard_sorted(sketch_a, sketch_b):
    """Jaccard similarity from two sorted MinHash sketches (merge-based)."""
    s = len(sketch_a)
    max_val = uint64(0xFFFFFFFFFFFFFFFF)

    shared = 0
    i = 0
    j = 0
    union_count = 0

    while union_count < s and i < s and j < s:
        a_val = sketch_a[i]
        b_val = sketch_b[j]

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
        return float32(0.0)

    return float32(shared / union_count)


@njit(parallel=True, cache=True)
def _batch_jaccard(pairs, sketches, threshold):
    """Compute Jaccard for all candidate pairs in parallel.

    Returns similarities array and a mask of which pairs pass the threshold.
    """
    m = pairs.shape[0]
    sims = np.empty(m, dtype=np.float32)
    mask = np.empty(m, dtype=np.bool_)

    for idx in prange(m):
        i = pairs[idx, 0]
        j = pairs[idx, 1]
        sim = _jaccard_sorted(sketches[i], sketches[j])
        sims[idx] = sim
        mask[idx] = sim >= threshold

    return sims, mask


def jaccard_from_sketches(
    sketch_a: np.ndarray,
    sketch_b: np.ndarray,
) -> float:
    """Estimate Jaccard similarity from two sorted MinHash sketches.

    Uses the merge-based MinHash Jaccard estimator:
    count shared hashes / total unique hashes in the union of bottom-s.
    """
    return float(_jaccard_sorted(sketch_a, sketch_b))


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

    sims, mask = _batch_jaccard(candidate_pairs, sketches, np.float32(threshold))

    filtered_pairs = candidate_pairs[mask]
    filtered_sims = sims[mask]

    return filtered_pairs, filtered_sims
