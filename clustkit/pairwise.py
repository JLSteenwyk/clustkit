"""Phase 3: Pairwise Similarity — Compute identity for candidate pairs.

Two modes:
  - "kmer": fast k-mer Jaccard from MinHash sketches (good for ≥60% identity)
  - "align": Needleman-Wunsch global alignment identity (accurate at all thresholds)

Both modes use Numba JIT and parallel batch processing.
"""

import numpy as np
from numba import njit, prange, uint64, int32, int8, float32

# ──────────────────────────────────────────────────────────────────────
# BLOSUM62-derived scoring for protein alignment (Numba-compatible)
# Simplified: match = +2, mismatch = -1, gap_open = -10, gap_extend = -1
# We compute identity from the alignment, not the score, so simple scoring
# is fine — it just needs to produce a reasonable alignment path.
# ──────────────────────────────────────────────────────────────────────

GAP_OPEN = int32(-10)
GAP_EXTEND = int32(-1)
MATCH_SCORE = int32(2)
MISMATCH_SCORE = int32(-1)


@njit(cache=True)
def _nw_identity(seq_a, len_a, seq_b, len_b, band_width):
    """Banded Needleman-Wunsch global alignment, returns sequence identity.

    Uses affine gap penalties and a banded DP matrix for speed.
    Identity = number of identical aligned residues / length of shorter sequence.

    Args:
        seq_a, seq_b: uint8 encoded sequences (padded arrays).
        len_a, len_b: actual lengths.
        band_width: half-width of the DP band (0 = full matrix).

    Returns:
        Sequence identity as float32 in [0, 1].
    """
    # For very short sequences, return 0
    if len_a == 0 or len_b == 0:
        return float32(0.0)

    # Full DP if band_width is 0 or sequences are short
    if band_width <= 0 or max(len_a, len_b) <= 50:
        band_width = max(len_a, len_b)

    # DP matrices: H = match/mismatch, E = gap in seq_b, F = gap in seq_a
    # Using 1D rolling arrays to save memory: only need current and previous row
    cols = len_b + 1

    # Previous and current row for H, E, F
    NEG_INF = int32(-1000000)

    prev_H = np.full(cols, NEG_INF, dtype=np.int32)
    prev_E = np.full(cols, NEG_INF, dtype=np.int32)
    curr_H = np.full(cols, NEG_INF, dtype=np.int32)
    curr_E = np.full(cols, NEG_INF, dtype=np.int32)
    curr_F = int32(NEG_INF)

    # Traceback direction: 0=diag, 1=up(gap in b), 2=left(gap in a)
    trace = np.zeros((len_a + 1, cols), dtype=np.int8)

    # Initialize first row
    prev_H[0] = int32(0)
    for j in range(1, cols):
        prev_H[j] = GAP_OPEN + GAP_EXTEND * int32(j - 1)
        trace[0, j] = int8(2)

    # Fill DP
    for i in range(1, len_a + 1):
        # Band boundaries
        j_start = max(1, i - band_width)
        j_end = min(cols, i + band_width + 1)

        # Reset current row
        for j in range(cols):
            curr_H[j] = NEG_INF
            curr_E[j] = NEG_INF

        # First column
        if j_start == 1:
            curr_H[0] = GAP_OPEN + GAP_EXTEND * int32(i - 1)
            trace[i, 0] = int8(1)

        for j in range(j_start, j_end):
            # Match/mismatch
            if seq_a[i - 1] == seq_b[j - 1]:
                s = MATCH_SCORE
            else:
                s = MISMATCH_SCORE

            diag = prev_H[j - 1] + s

            # Gap in seq_b (insertion): extend from E or open from H
            e_extend = prev_E[j] + GAP_EXTEND
            e_open = prev_H[j] + GAP_OPEN
            curr_E[j] = max(e_extend, e_open)

            # Gap in seq_a (deletion): extend from F or open from H
            f_extend = curr_F + GAP_EXTEND if j > j_start else NEG_INF
            f_open = curr_H[j - 1] + GAP_OPEN
            curr_F = max(f_extend, f_open)

            # Best score
            best = diag
            tb = int8(0)
            if curr_E[j] > best:
                best = curr_E[j]
                tb = int8(1)
            if curr_F > best:
                best = curr_F
                tb = int8(2)

            curr_H[j] = best
            trace[i, j] = tb

        # Swap rows
        for j in range(cols):
            prev_H[j] = curr_H[j]
            prev_E[j] = curr_E[j]

    # Traceback to count identical positions
    i = len_a
    j = len_b
    matches = 0
    aln_len = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and trace[i, j] == 0:
            if seq_a[i - 1] == seq_b[j - 1]:
                matches += 1
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or trace[i, j] == 1):
            i -= 1
        else:
            j -= 1
        aln_len += 1

    if aln_len == 0:
        return float32(0.0)

    # Identity = matches / shorter sequence length (CD-HIT convention)
    shorter = min(len_a, len_b)
    return float32(matches / shorter)


@njit(parallel=True, cache=True)
def _batch_align(pairs, sequences, lengths, threshold, band_width):
    """Compute alignment identity for all candidate pairs in parallel."""
    m = pairs.shape[0]
    sims = np.empty(m, dtype=np.float32)
    mask = np.empty(m, dtype=np.bool_)

    for idx in prange(m):
        i = pairs[idx, 0]
        j = pairs[idx, 1]
        identity = _nw_identity(
            sequences[i], lengths[i],
            sequences[j], lengths[j],
            band_width,
        )
        sims[idx] = identity
        mask[idx] = identity >= threshold

    return sims, mask


def compute_pairwise_alignment(
    candidate_pairs: np.ndarray,
    encoded_sequences: np.ndarray,
    lengths: np.ndarray,
    threshold: float,
    band_width: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pairwise identity via alignment for candidate pairs.

    Args:
        candidate_pairs: (M, 2) int32 array of candidate pairs (i, j).
        encoded_sequences: (N, max_len) uint8 matrix.
        lengths: (N,) int32 array of sequence lengths.
        threshold: Minimum sequence identity to keep a pair.
        band_width: Half-width of the alignment band (0 = full).

    Returns:
        Tuple of:
        - filtered_pairs: (K, 2) int32 array of pairs above threshold.
        - identities: (K,) float32 array of sequence identities.
    """
    if len(candidate_pairs) == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float32)

    sims, mask = _batch_align(
        candidate_pairs, encoded_sequences, lengths,
        np.float32(threshold), int32(band_width),
    )

    return candidate_pairs[mask], sims[mask]


# ──────────────────────────────────────────────────────────────────────
# K-mer Jaccard (kept for backwards compatibility / fast mode)
# ──────────────────────────────────────────────────────────────────────

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
    """Compute Jaccard for all candidate pairs in parallel."""
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
    """Estimate Jaccard similarity from two sorted MinHash sketches."""
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

    return candidate_pairs[mask], sims[mask]
