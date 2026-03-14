"""Phase 3: Pairwise Similarity — Compute identity for candidate pairs.

Two modes:
  - "kmer": fast k-mer Jaccard from MinHash sketches (good for >=60% identity)
  - "align": Needleman-Wunsch global alignment identity (accurate at all thresholds)

Alignment uses Numba JIT with adaptive banded NW, forward match counting
(no traceback needed), and early termination. The band width adapts per pair
based on sequence length difference: narrow bands for similar-length
sequences (typical case), wider bands when needed.

GPU uses CuPy raw kernels for k-mer Jaccard mode.
"""

import logging
import numpy as np
from numba import njit, prange, uint64, int32, int8, float32

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

_logger = logging.getLogger("clustkit")

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

# ──────────────────────────────────────────────────────────────────────
# BLOSUM62 substitution matrix for protein search mode
# Alphabet order matches ClustKIT encoding: ACDEFGHIKLMNPQRSTVWY
# Standard gap penalties for BLOSUM62: open=-11, extend=-1
# ──────────────────────────────────────────────────────────────────────

BLOSUM62 = np.array([
    #  A   C   D   E   F   G   H   I   K   L   M   N   P   Q   R   S   T   V   W   Y
    [ 4,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2],  # A
    [ 0,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],  # C
    [-2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3],  # D
    [-1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2],  # E
    [-2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3],  # F
    [ 0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3],  # G
    [-2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2],  # H
    [-1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1],  # I
    [-1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2],  # K
    [-1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1],  # L
    [-1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1],  # M
    [-2, -3,  0,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2],  # N
    [-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3],  # P
    [-1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1],  # Q
    [-1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2],  # R
    [ 1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2],  # S
    [ 0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2],  # T
    [ 0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1],  # V
    [-3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2],  # W
    [-2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7],  # Y
], dtype=np.int8)

BLOSUM62_GAP_OPEN = int32(-11)
BLOSUM62_GAP_EXTEND = int32(-1)


# ──────────────────────────────────────────────────────────────────────
# Ungapped extension pre-filter (Stage 1.5 for search)
# ──────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _ungapped_diag_score(seq_a, len_a, seq_b, len_b, diag_offset, sub_matrix):
    """Max-segment score along one diagonal using Kadane's algorithm.

    diag_offset: column in seq_b where row 0 of seq_a aligns.
    Positive = seq_b shifted right, negative = seq_a shifted right.

    Returns the maximum contiguous segment score (>= 0).
    """
    if diag_offset >= 0:
        a_start = int32(0)
        b_start = int32(diag_offset)
    else:
        a_start = int32(-diag_offset)
        b_start = int32(0)

    a_remain = len_a - a_start
    b_remain = len_b - b_start
    diag_len = min(a_remain, b_remain)
    if diag_len <= 0:
        return int32(0)

    max_score = int32(0)
    curr_score = int32(0)
    for k in range(diag_len):
        a_res = seq_a[a_start + k]
        b_res = seq_b[b_start + k]
        if a_res < 20 and b_res < 20:
            s = int32(sub_matrix[a_res, b_res])
        else:
            s = int32(-4)
        curr_score += s
        if curr_score > max_score:
            max_score = curr_score
        if curr_score < 0:
            curr_score = int32(0)
    return max_score


@njit(parallel=True, cache=True)
def _batch_ungapped_prefilter(pairs, flat_sequences, offsets, lengths,
                              sub_matrix, min_score):
    """Pre-filter pairs using ungapped extension on 3 diagonals.

    For each pair, computes the Kadane max-segment score at start-aligned,
    center-aligned, and end-aligned diagonals using the substitution matrix.
    If the best score across all 3 diagonals is below min_score, the pair
    is filtered out.

    This is O(L) per pair per diagonal — roughly 3-5x cheaper than banded NW.
    """
    m = pairs.shape[0]
    mask = np.empty(m, dtype=np.bool_)

    for idx in prange(m):
        i = pairs[idx, 0]
        j = pairs[idx, 1]

        len_i = lengths[i]
        len_j = lengths[j]

        seq_i = flat_sequences[offsets[i]:offsets[i] + len_i]
        seq_j = flat_sequences[offsets[j]:offsets[j] + len_j]

        # Ensure shorter is seq_a for consistent diagonal calculation
        if len_i <= len_j:
            len_a = len_i
            len_b = len_j
            seq_a = seq_i
            seq_b = seq_j
        else:
            len_a = len_j
            len_b = len_i
            seq_a = seq_j
            seq_b = seq_i

        diff = int32(len_b - len_a)

        # Diagonal 1: start-aligned (a[0] <-> b[0])
        s0 = _ungapped_diag_score(seq_a, len_a, seq_b, len_b, int32(0), sub_matrix)

        # Diagonal 2: center-aligned (a[0] <-> b[diff/2])
        s1 = _ungapped_diag_score(seq_a, len_a, seq_b, len_b, int32(diff // 2), sub_matrix)

        # Diagonal 3: end-aligned (a[-1] <-> b[-1])
        s2 = _ungapped_diag_score(seq_a, len_a, seq_b, len_b, diff, sub_matrix)

        best = s0
        if s1 > best:
            best = s1
        if s2 > best:
            best = s2

        mask[idx] = best >= min_score

    return mask


@njit(cache=True)
def _nw_identity(seq_a, len_a, seq_b, len_b, band_width, threshold):
    """Banded Needleman-Wunsch global alignment, returns sequence identity.

    Uses affine gap penalties with forward match counting (no traceback needed).
    Supports early termination when the threshold becomes unreachable.

    Identity = number of identical aligned residues / length of shorter sequence.

    Caller should ensure len_a <= len_b for efficiency (fewer DP rows).

    Uses 2D arrays with row toggling to avoid O(cols) per-row copy. Only
    border cells (O(1) per row) need cleaning instead of full-row reset.

    Args:
        seq_a, seq_b: uint8 encoded sequences (padded arrays).
        len_a, len_b: actual lengths.
        band_width: half-width of the DP band (0 = full matrix).
        threshold: identity threshold for early termination (0 = disabled).

    Returns:
        Sequence identity as float32 in [0, 1].
    """
    if len_a == 0 or len_b == 0:
        return float32(0.0)

    # Full DP if band_width is 0 or sequences are short
    if band_width <= 0 or max(len_a, len_b) <= 50:
        band_width = max(len_a, len_b)

    cols = len_b + 1
    shorter = min(len_a, len_b)
    NEG_INF = int32(-1000000)

    # 2D rolling arrays with row toggling — eliminates O(cols) copy per row
    H = np.full((2, cols), NEG_INF, dtype=np.int32)
    E = np.full((2, cols), NEG_INF, dtype=np.int32)
    Hm = np.zeros((2, cols), dtype=np.int32)
    Em = np.zeros((2, cols), dtype=np.int32)

    pr = int32(0)  # previous row index
    cr = int32(1)  # current row index

    # Initialize first row (only within band of row 1)
    H[pr, 0] = int32(0)
    j_end_init = min(cols, band_width + 2)
    for j in range(1, j_end_init):
        H[pr, j] = GAP_OPEN + GAP_EXTEND * int32(j - 1)

    # Fill DP
    for i in range(1, len_a + 1):
        j_start = max(1, i - band_width)
        j_end = min(cols, i + band_width + 1)

        # Clean left border of current row (O(1))
        if j_start == 1:
            H[cr, 0] = GAP_OPEN + GAP_EXTEND * int32(i - 1)
            Hm[cr, 0] = int32(0)
        else:
            H[cr, j_start - 1] = NEG_INF
            Hm[cr, j_start - 1] = int32(0)

        # Clean right border of prev row — the new column entering the band
        # from the right may have a stale value from 2 rows ago
        prev_j_end = min(cols, i + band_width)  # row i-1's j_end
        if j_end > prev_j_end and prev_j_end < cols:
            H[pr, prev_j_end] = NEG_INF
            E[pr, prev_j_end] = NEG_INF
            Hm[pr, prev_j_end] = int32(0)
            Em[pr, prev_j_end] = int32(0)

        curr_F = int32(NEG_INF)
        curr_Fm = int32(0)
        max_Hm = int32(0)

        for j in range(j_start, j_end):
            is_match = int32(seq_a[i - 1] == seq_b[j - 1])
            s = MATCH_SCORE if is_match else MISMATCH_SCORE

            diag = H[pr, j - 1] + s
            diag_m = Hm[pr, j - 1] + is_match

            # E: gap in seq_b (vertical)
            e_extend = E[pr, j] + GAP_EXTEND
            e_open = H[pr, j] + GAP_OPEN
            if e_extend >= e_open:
                E[cr, j] = e_extend
                Em[cr, j] = Em[pr, j]
            else:
                E[cr, j] = e_open
                Em[cr, j] = Hm[pr, j]

            # F: gap in seq_a (horizontal)
            f_extend = curr_F + GAP_EXTEND if j > j_start else NEG_INF
            f_extend_m = curr_Fm
            f_open = H[cr, j - 1] + GAP_OPEN
            f_open_m = Hm[cr, j - 1]
            if f_extend >= f_open:
                curr_F = f_extend
                curr_Fm = f_extend_m
            else:
                curr_F = f_open
                curr_Fm = f_open_m

            best = diag
            best_m = diag_m
            if E[cr, j] > best:
                best = E[cr, j]
                best_m = Em[cr, j]
            if curr_F > best:
                best = curr_F
                best_m = curr_Fm

            H[cr, j] = best
            Hm[cr, j] = best_m

            if best_m > max_Hm:
                max_Hm = best_m

        # Toggle rows (O(1) instead of O(cols) copy)
        cr, pr = pr, cr

        # Early termination
        remaining = len_a - i
        max_possible = float32((max_Hm + remaining) / shorter)
        if max_possible < threshold:
            return float32(0.0)

    # After loop, pr points to the last computed row
    return float32(Hm[pr, len_b] / shorter)


@njit(cache=True)
def _nw_identity_and_score(seq_a, len_a, seq_b, len_b, band_width, threshold, sub_matrix):
    """Banded NW with substitution matrix, returns (identity, raw_score).

    Uses the provided substitution matrix for scoring (e.g., BLOSUM62).
    Identity is still computed as exact matches / shorter length.
    The raw alignment score uses the substitution matrix values.

    Gap penalties: open=-11, extend=-1 (standard BLOSUM62 pairing).
    """
    if len_a == 0 or len_b == 0:
        return float32(0.0), int32(0)

    gap_open = int32(-11)
    gap_extend = int32(-1)

    if band_width <= 0 or max(len_a, len_b) <= 50:
        band_width = max(len_a, len_b)

    cols = len_b + 1
    shorter = min(len_a, len_b)
    NEG_INF = int32(-1000000)

    H = np.full((2, cols), NEG_INF, dtype=np.int32)
    E = np.full((2, cols), NEG_INF, dtype=np.int32)
    Hm = np.zeros((2, cols), dtype=np.int32)
    Em = np.zeros((2, cols), dtype=np.int32)

    pr = int32(0)
    cr = int32(1)

    H[pr, 0] = int32(0)
    j_end_init = min(cols, band_width + 2)
    for j in range(1, j_end_init):
        H[pr, j] = gap_open + gap_extend * int32(j - 1)

    for i in range(1, len_a + 1):
        j_start = max(1, i - band_width)
        j_end = min(cols, i + band_width + 1)

        if j_start == 1:
            H[cr, 0] = gap_open + gap_extend * int32(i - 1)
            Hm[cr, 0] = int32(0)
        else:
            H[cr, j_start - 1] = NEG_INF
            Hm[cr, j_start - 1] = int32(0)

        prev_j_end = min(cols, i + band_width)
        if j_end > prev_j_end and prev_j_end < cols:
            H[pr, prev_j_end] = NEG_INF
            E[pr, prev_j_end] = NEG_INF
            Hm[pr, prev_j_end] = int32(0)
            Em[pr, prev_j_end] = int32(0)

        curr_F = int32(NEG_INF)
        curr_Fm = int32(0)
        max_Hm = int32(0)

        for j in range(j_start, j_end):
            a_res = seq_a[i - 1]
            b_res = seq_b[j - 1]
            is_match = int32(a_res == b_res)

            # Use substitution matrix for scoring
            if a_res < 20 and b_res < 20:
                s = int32(sub_matrix[a_res, b_res])
            else:
                s = int32(-4)

            diag = H[pr, j - 1] + s
            diag_m = Hm[pr, j - 1] + is_match

            e_extend = E[pr, j] + gap_extend
            e_open = H[pr, j] + gap_open
            if e_extend >= e_open:
                E[cr, j] = e_extend
                Em[cr, j] = Em[pr, j]
            else:
                E[cr, j] = e_open
                Em[cr, j] = Hm[pr, j]

            f_extend = curr_F + gap_extend if j > j_start else NEG_INF
            f_extend_m = curr_Fm
            f_open = H[cr, j - 1] + gap_open
            f_open_m = Hm[cr, j - 1]
            if f_extend >= f_open:
                curr_F = f_extend
                curr_Fm = f_extend_m
            else:
                curr_F = f_open
                curr_Fm = f_open_m

            best = diag
            best_m = diag_m
            if E[cr, j] > best:
                best = E[cr, j]
                best_m = Em[cr, j]
            if curr_F > best:
                best = curr_F
                best_m = curr_Fm

            H[cr, j] = best
            Hm[cr, j] = best_m

            if best_m > max_Hm:
                max_Hm = best_m

        cr, pr = pr, cr

        remaining = len_a - i
        max_possible = float32((max_Hm + remaining) / shorter)
        if max_possible < threshold:
            return float32(0.0), int32(NEG_INF)

    return float32(Hm[pr, len_b] / shorter), H[pr, len_b]


@njit(cache=True)
def _sw_identity_and_score(seq_a, len_a, seq_b, len_b, band_width, threshold, sub_matrix):
    """Smith-Waterman local alignment returning (identity, raw_score).

    Computes a banded Smith-Waterman local alignment using the given
    substitution matrix.  Returns the best local alignment score and an
    approximate identity (matches in aligned region / shorter length).

    Unlike NW, local alignment does not penalise terminal gaps, making it
    far more suitable for homology search where sequences may differ in
    length or share only a sub-domain.

    Gap penalties: open=-11, extend=-1 (standard BLOSUM62 pairing).
    """
    if len_a == 0 or len_b == 0:
        return float32(0.0), int32(0)

    gap_open = int32(-11)
    gap_extend = int32(-1)

    if band_width <= 0 or max(len_a, len_b) <= 50:
        band_width = max(len_a, len_b)

    cols = len_b + 1
    shorter = min(len_a, len_b)
    NEG_INF = int32(-1000000)

    # H: SW score matrix (non-negative), Hm: match counts along path
    H = np.zeros((2, cols), dtype=np.int32)
    E = np.full((2, cols), NEG_INF, dtype=np.int32)
    Hm = np.zeros((2, cols), dtype=np.int32)
    Em = np.zeros((2, cols), dtype=np.int32)

    pr = int32(0)
    cr = int32(1)

    max_score = int32(0)
    max_matches = int32(0)

    # First row: all zeros (local alignment — no gap init)

    for i in range(1, len_a + 1):
        j_start = max(1, i - band_width)
        j_end = min(cols, i + band_width + 1)

        # Boundary: left of band → H = 0 for SW
        if j_start == 1:
            H[cr, 0] = int32(0)
            Hm[cr, 0] = int32(0)
        else:
            H[cr, j_start - 1] = int32(0)
            Hm[cr, j_start - 1] = int32(0)

        # Boundary: right of band from previous row
        prev_j_end = min(cols, i + band_width)
        if j_end > prev_j_end and prev_j_end < cols:
            H[pr, prev_j_end] = int32(0)
            E[pr, prev_j_end] = NEG_INF
            Hm[pr, prev_j_end] = int32(0)
            Em[pr, prev_j_end] = int32(0)

        curr_F = NEG_INF
        curr_Fm = int32(0)

        for j in range(j_start, j_end):
            a_res = seq_a[i - 1]
            b_res = seq_b[j - 1]
            is_match = int32(a_res == b_res)

            if a_res < 20 and b_res < 20:
                s = int32(sub_matrix[a_res, b_res])
            else:
                s = int32(-4)

            diag = H[pr, j - 1] + s
            diag_m = Hm[pr, j - 1] + is_match

            # E: gap in seq_b (vertical)
            e_extend = E[pr, j] + gap_extend
            e_open = H[pr, j] + gap_open
            if e_extend >= e_open:
                E[cr, j] = e_extend
                Em[cr, j] = Em[pr, j]
            else:
                E[cr, j] = e_open
                Em[cr, j] = Hm[pr, j]

            # F: gap in seq_a (horizontal)
            f_extend = curr_F + gap_extend if j > j_start else NEG_INF
            f_extend_m = curr_Fm
            f_open = H[cr, j - 1] + gap_open
            f_open_m = Hm[cr, j - 1]
            if f_extend >= f_open:
                curr_F = f_extend
                curr_Fm = f_extend_m
            else:
                curr_F = f_open
                curr_Fm = f_open_m

            # SW recurrence: max(0, diag, E, F)
            best = int32(0)
            best_m = int32(0)
            if diag > best:
                best = diag
                best_m = diag_m
            if E[cr, j] > best:
                best = E[cr, j]
                best_m = Em[cr, j]
            if curr_F > best:
                best = curr_F
                best_m = curr_Fm

            H[cr, j] = best
            Hm[cr, j] = best_m

            if best > max_score:
                max_score = best
                max_matches = best_m

        cr, pr = pr, cr

    if max_score <= 0:
        return float32(0.0), int32(0)

    identity = float32(max_matches) / float32(shorter)
    return identity, max_score


@njit(parallel=True, cache=True)
def _batch_sw_compact_scored(pairs, flat_sequences, offsets, lengths,
                             threshold, band_width, sub_matrix):
    """Banded Smith-Waterman local alignment, returns (identities, scores, mask).

    Uses the full band_width (no adaptive reduction) since local alignment
    doesn't need to span both full sequences.  No length-ratio pre-filter
    because SW naturally handles different-length sequences.
    """
    m = pairs.shape[0]
    sims = np.empty(m, dtype=np.float32)
    scores = np.empty(m, dtype=np.int32)
    mask = np.empty(m, dtype=np.bool_)

    for idx in prange(m):
        i = pairs[idx, 0]
        j = pairs[idx, 1]

        len_i = lengths[i]
        len_j = lengths[j]

        if len_i == 0 or len_j == 0:
            sims[idx] = float32(0.0)
            scores[idx] = int32(0)
            mask[idx] = False
            continue

        seq_i = flat_sequences[offsets[i]:offsets[i] + len_i]
        seq_j = flat_sequences[offsets[j]:offsets[j] + len_j]

        if len_i <= len_j:
            identity, score = _sw_identity_and_score(
                seq_i, len_i, seq_j, len_j, band_width, threshold, sub_matrix,
            )
        else:
            identity, score = _sw_identity_and_score(
                seq_j, len_j, seq_i, len_i, band_width, threshold, sub_matrix,
            )
        sims[idx] = identity
        scores[idx] = score
        # For SW, use score-based filtering: any positive local alignment
        # passes.  Identity-based filtering is too conservative because
        # matches/shorter_length underestimates local identity.
        mask[idx] = score > int32(0)

    return sims, scores, mask


@njit(parallel=True, cache=True)
def _batch_align_compact_scored(pairs, flat_sequences, offsets, lengths,
                                threshold, band_width, sub_matrix):
    """Banded NW with substitution matrix, returns (identities, scores, mask).

    Same as _batch_align_compact but uses a substitution matrix for scoring
    and returns raw alignment scores alongside identity values.
    """
    m = pairs.shape[0]
    sims = np.empty(m, dtype=np.float32)
    scores = np.empty(m, dtype=np.int32)
    mask = np.empty(m, dtype=np.bool_)

    NEG_INF = int32(-1000000)

    for idx in prange(m):
        i = pairs[idx, 0]
        j = pairs[idx, 1]

        len_i = lengths[i]
        len_j = lengths[j]

        shorter = min(len_i, len_j)
        longer = max(len_i, len_j)
        if longer > 0 and float32(shorter) / float32(longer) < threshold:
            sims[idx] = float32(0.0)
            scores[idx] = NEG_INF
            mask[idx] = False
            continue

        len_diff = abs(int32(len_i) - int32(len_j))
        if len_diff > band_width:
            sims[idx] = float32(0.0)
            scores[idx] = NEG_INF
            mask[idx] = False
            continue

        adaptive_band = min(band_width, max(int32(10), len_diff + int32(10)))

        seq_i = flat_sequences[offsets[i]:offsets[i] + len_i]
        seq_j = flat_sequences[offsets[j]:offsets[j] + len_j]

        if len_i <= len_j:
            identity, score = _nw_identity_and_score(
                seq_i, len_i, seq_j, len_j, adaptive_band, threshold, sub_matrix,
            )
        else:
            identity, score = _nw_identity_and_score(
                seq_j, len_j, seq_i, len_i, adaptive_band, threshold, sub_matrix,
            )
        sims[idx] = identity
        scores[idx] = score
        mask[idx] = identity >= threshold

    return sims, scores, mask


@njit(parallel=True, cache=True)
def _batch_align(pairs, sequences, lengths, threshold, band_width):
    """Compute alignment identity for all candidate pairs in parallel.

    Optimizations (all zero accuracy loss):
    - Length-ratio pre-filter: if min(len)/max(len) < threshold, identity
      cannot reach threshold (mathematical bound: identity <= shorter/longer)
    - Length-diff pre-filter: skips pairs where len_diff > band_width (the
      banded DP can never reach the final cell, so identity is always 0.0)
    - Sequence swap: ensures shorter sequence is seq_a (fewer DP rows)
    - Adaptive band width: narrows the band for similar-length sequences
    """
    m = pairs.shape[0]
    sims = np.empty(m, dtype=np.float32)
    mask = np.empty(m, dtype=np.bool_)

    for idx in prange(m):
        i = pairs[idx, 0]
        j = pairs[idx, 1]

        len_i = lengths[i]
        len_j = lengths[j]

        # B1: Length-ratio pre-filter (mathematically exact, zero accuracy loss).
        # Identity = matches/shorter. Since matches <= shorter <= longer,
        # identity <= shorter/longer. If shorter/longer < threshold, skip.
        shorter = min(len_i, len_j)
        longer = max(len_i, len_j)
        if longer > 0 and float32(shorter) / float32(longer) < threshold:
            sims[idx] = float32(0.0)
            mask[idx] = False
            continue

        len_diff = abs(int32(len_i) - int32(len_j))

        # Pre-filter: if length difference exceeds band_width, the banded DP
        # cannot reach cell (len_a, len_b) and will always return identity 0.
        if len_diff > band_width:
            sims[idx] = float32(0.0)
            mask[idx] = False
            continue

        # Adaptive band: narrow for similar lengths, wider for different lengths
        adaptive_band = min(band_width, max(int32(10), len_diff + int32(10)))

        # Swap so shorter sequence is seq_a (fewer DP rows, same result)
        if len_i <= len_j:
            identity = _nw_identity(
                sequences[i], len_i,
                sequences[j], len_j,
                adaptive_band, threshold,
            )
        else:
            identity = _nw_identity(
                sequences[j], len_j,
                sequences[i], len_i,
                adaptive_band, threshold,
            )
        sims[idx] = identity
        mask[idx] = identity >= threshold

    return sims, mask


@njit(parallel=True, cache=True)
def _batch_align_compact(pairs, flat_sequences, offsets, lengths, threshold, band_width):
    """Compute alignment identity for all candidate pairs in parallel (compact storage).

    Same logic as _batch_align but reads sequences from a flat 1D array + offsets
    instead of a padded 2D matrix.  Better cache behaviour and ~25x less memory
    when outlier sequences inflate max_len.
    """
    m = pairs.shape[0]
    sims = np.empty(m, dtype=np.float32)
    mask = np.empty(m, dtype=np.bool_)

    for idx in prange(m):
        i = pairs[idx, 0]
        j = pairs[idx, 1]

        len_i = lengths[i]
        len_j = lengths[j]

        shorter = min(len_i, len_j)
        longer = max(len_i, len_j)
        if longer > 0 and float32(shorter) / float32(longer) < threshold:
            sims[idx] = float32(0.0)
            mask[idx] = False
            continue

        len_diff = abs(int32(len_i) - int32(len_j))

        if len_diff > band_width:
            sims[idx] = float32(0.0)
            mask[idx] = False
            continue

        adaptive_band = min(band_width, max(int32(10), len_diff + int32(10)))

        seq_i = flat_sequences[offsets[i]:offsets[i] + len_i]
        seq_j = flat_sequences[offsets[j]:offsets[j] + len_j]

        if len_i <= len_j:
            identity = _nw_identity(seq_i, len_i, seq_j, len_j, adaptive_band, threshold)
        else:
            identity = _nw_identity(seq_j, len_j, seq_i, len_i, adaptive_band, threshold)
        sims[idx] = identity
        mask[idx] = identity >= threshold

    return sims, mask


def _calibrate_device(
    candidate_pairs: np.ndarray,
    encoded_sequences: np.ndarray,
    lengths: np.ndarray,
    threshold: float,
    band_width: int,
    gpu_device: int = 0,
    sample_size: int = 2000,
) -> str:
    """Pick fastest device by timing a small sample on CPU and GPU.

    Returns "cpu" or a GPU device ID string (e.g. "0").
    """
    import time

    m = len(candidate_pairs)
    n_sample = min(sample_size, m)

    # Use a random but deterministic subset
    rng = np.random.RandomState(0)
    idx = rng.choice(m, size=n_sample, replace=False)
    sample_pairs = candidate_pairs[idx]

    # Warm up Numba (already compiled from module import, but ensure it)
    if n_sample > 10:
        _batch_align(
            sample_pairs[:10], encoded_sequences, lengths,
            np.float32(threshold), int32(band_width),
        )

    # Time CPU
    t0 = time.perf_counter()
    _batch_align(
        sample_pairs, encoded_sequences, lengths,
        np.float32(threshold), int32(band_width),
    )
    cpu_time = time.perf_counter() - t0
    cpu_rate = n_sample / cpu_time  # pairs/sec

    # Time GPU (includes first-call kernel compilation)
    try:
        # Warm up kernel compilation with tiny batch
        _compute_pairwise_alignment_gpu(
            sample_pairs[:10], encoded_sequences, lengths,
            threshold, band_width, gpu_device,
        )
        t0 = time.perf_counter()
        _compute_pairwise_alignment_gpu(
            sample_pairs, encoded_sequences, lengths,
            threshold, band_width, gpu_device,
        )
        gpu_time = time.perf_counter() - t0
        gpu_rate = n_sample / gpu_time
    except Exception:
        _logger.debug("  GPU calibration failed, using CPU")
        return "cpu"

    # Extrapolate — add a 10% margin for GPU to account for batch overhead
    cpu_est = m / cpu_rate
    gpu_est = m / gpu_rate * 1.1

    _logger.info(
        f"  Device calibration ({n_sample} pairs): "
        f"CPU {cpu_rate:.0f} pairs/s ({cpu_est:.1f}s est), "
        f"GPU {gpu_rate:.0f} pairs/s ({gpu_est:.1f}s est)"
    )

    if gpu_est < cpu_est:
        _logger.info(f"  Auto-selected: GPU (device {gpu_device})")
        return str(gpu_device)
    else:
        _logger.info(f"  Auto-selected: CPU")
        return "cpu"


def _jaccard_prefilter(
    candidate_pairs: np.ndarray,
    sketches: np.ndarray,
    threshold: float,
    min_filter_rate: float = 0.05,
) -> np.ndarray | None:
    """Pre-filter candidate pairs using Jaccard similarity from sketches.

    Uses a conservative floor well below the expected Jaccard for true pairs.
    At identity t with k-mer size k, expected Jaccard ~ t^k.  We use a floor
    of 0.5 * t^k_lsh (where k_lsh is the sketch k-mer size, inferred from
    the relationship between threshold and expected Jaccard).

    Before running on all pairs, samples 2000 pairs to estimate the filtering
    rate. If fewer than ``min_filter_rate`` (default 5%) of pairs would be
    removed, returns None to skip the pre-filter entirely — the overhead of
    computing Jaccard for all pairs exceeds the savings from removing so few.

    Returns:
        Boolean mask of pairs that pass the Jaccard pre-filter, or None if
        the pre-filter would not be cost-effective.
    """
    m = len(candidate_pairs)
    if m == 0:
        return np.ones(0, dtype=np.bool_)

    # Conservative Jaccard floor: well below minimum expected for true pairs
    # For protein k=3: t=0.4 → J~0.064, floor=0.003; t=0.7 → J~0.34, floor=0.05
    jaccard_floor = max(0.001, 0.3 * threshold ** 5)

    # Cost-benefit check: sample a small subset to estimate filtering rate.
    # At low thresholds with small k, the Jaccard floor is too low to
    # discriminate and the pre-filter overhead exceeds alignment savings.
    sample_size = min(2000, m)
    if sample_size < m:
        rng = np.random.RandomState(0)
        sample_idx = rng.choice(m, size=sample_size, replace=False)
        sample_sims, _ = _batch_jaccard(
            candidate_pairs[sample_idx], sketches, np.float32(jaccard_floor),
        )
        est_filter_rate = float(np.mean(sample_sims < jaccard_floor))
        if est_filter_rate < min_filter_rate:
            _logger.info(
                f"  Jaccard pre-filter skipped: estimated {est_filter_rate:.1%} "
                f"removal rate below {min_filter_rate:.0%} threshold"
            )
            return None

    sims, _ = _batch_jaccard(
        candidate_pairs, sketches, np.float32(jaccard_floor),
    )

    return sims >= jaccard_floor


def compute_pairwise_alignment(
    candidate_pairs: np.ndarray,
    encoded_sequences: np.ndarray,
    lengths: np.ndarray,
    threshold: float,
    band_width: int = 50,
    device: str = "cpu",
    mode: str = "protein",
    sketches: np.ndarray | None = None,
    flat_sequences: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pairwise identity via alignment for candidate pairs.

    Args:
        candidate_pairs: (M, 2) int32 array of candidate pairs (i, j).
        encoded_sequences: (N, max_len) uint8 matrix.  May be None when
            flat_sequences and offsets are provided (compact format).
        lengths: (N,) int32 array of sequence lengths.
        threshold: Minimum sequence identity to keep a pair.
        band_width: Max half-width of the alignment band (adaptive per pair).
        device: "cpu", "auto", or GPU device ID (e.g. "0").
        mode: "protein" or "nucleotide".
        sketches: (N, sketch_size) uint64 array of MinHash sketches for
            Jaccard pre-filtering. If None, Jaccard pre-filter is skipped.
        flat_sequences: 1D uint8 array of concatenated sequences (compact format).
        offsets: (N,) int64 array of start positions in flat_sequences.

    Returns:
        Tuple of:
        - filtered_pairs: (K, 2) int32 array of pairs above threshold.
        - identities: (K,) float32 array of sequence identities.
    """
    if len(candidate_pairs) == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float32)

    # B2: Jaccard pre-filter — discard pairs with near-zero sketch overlap
    if sketches is not None:
        m_before = len(candidate_pairs)
        jaccard_mask = _jaccard_prefilter(candidate_pairs, sketches, threshold)
        if jaccard_mask is not None:
            candidate_pairs = candidate_pairs[jaccard_mask]
            m_after = len(candidate_pairs)
            if m_before > m_after:
                _logger.info(
                    f"  Jaccard pre-filter: {m_before} → {m_after} pairs "
                    f"({m_before - m_after} removed, {100*(m_before-m_after)/m_before:.1f}%)"
                )
            if len(candidate_pairs) == 0:
                return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float32)

    # B3: Cache-friendly pair sorting — sort by min(i,j) so adjacent threads
    # access nearby rows in the encoded_sequences matrix
    sort_key = np.minimum(candidate_pairs[:, 0], candidate_pairs[:, 1])
    sort_order = np.argsort(sort_key, kind="mergesort")
    candidate_pairs = candidate_pairs[sort_order]

    if device == "auto" and _CUPY_AVAILABLE:
        device = _calibrate_device(
            candidate_pairs, encoded_sequences, lengths,
            threshold, band_width,
        )

    if device != "cpu" and _CUPY_AVAILABLE:
        return _compute_pairwise_alignment_gpu(
            candidate_pairs, encoded_sequences, lengths,
            threshold, band_width, int(device),
        )

    # Prefer compact format on CPU (better cache behaviour, less memory)
    if flat_sequences is not None and offsets is not None:
        sims, mask = _batch_align_compact(
            candidate_pairs, flat_sequences, offsets, lengths,
            np.float32(threshold), int32(band_width),
        )
    else:
        sims, mask = _batch_align(
            candidate_pairs, encoded_sequences, lengths,
            np.float32(threshold), int32(band_width),
        )

    return candidate_pairs[mask], sims[mask]


# ──────────────────────────────────────────────────────────────────────
# GPU path for banded NW alignment (CuPy)
# ──────────────────────────────────────────────────────────────────────

_NW_ALIGN_KERNEL_CODE = r"""
extern "C" __global__
void nw_align_kernel(
    const int*           pairs,
    const unsigned char* sequences,
    const int*           lengths,
    float*               sims,
    int*                 mask_out,
    int*                 workspace,
    int M,
    int max_len,
    int max_band_cols,
    int band_width,
    float threshold
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= M) return;

    int si = pairs[idx * 2];
    int sj = pairs[idx * 2 + 1];
    int len_i = lengths[si];
    int len_j = lengths[sj];

    /* B1: Length-ratio pre-filter (mathematically exact).
       Identity = matches/shorter <= shorter/longer.
       If shorter/longer < threshold, identity can never reach threshold. */
    int shorter = (len_i < len_j) ? len_i : len_j;
    int longer  = (len_i > len_j) ? len_i : len_j;
    if (longer > 0 && (float)shorter / (float)longer < threshold) {
        sims[idx] = 0.0f;
        mask_out[idx] = 0;
        return;
    }

    int len_diff = len_i - len_j;
    if (len_diff < 0) len_diff = -len_diff;
    if (len_diff > band_width) {
        sims[idx] = 0.0f;
        mask_out[idx] = 0;
        return;
    }

    int bw = len_diff + 10;
    if (bw > band_width) bw = band_width;

    const unsigned char* seq_a;
    const unsigned char* seq_b;
    int len_a, len_b;
    if (len_i <= len_j) {
        seq_a = sequences + (long long)si * max_len;
        seq_b = sequences + (long long)sj * max_len;
        len_a = len_i; len_b = len_j;
    } else {
        seq_a = sequences + (long long)sj * max_len;
        seq_b = sequences + (long long)si * max_len;
        len_a = len_j; len_b = len_i;
    }

    if (len_a == 0 || len_b == 0) {
        sims[idx] = 0.0f;
        mask_out[idx] = 0;
        return;
    }

    int max_ab = len_a > len_b ? len_a : len_b;
    if (bw <= 0 || max_ab <= 50) bw = max_ab;

    int cols = len_b + 1;
    shorter = len_a;

    const int NEG_INF    = -1000000;
    const int GAP_OPEN   = -10;
    const int GAP_EXTEND = -1;
    const int MATCH_SC   = 2;
    const int MISMATCH_SC = -1;

    /* Band-relative indexing:
       For row i, column j: bc = j - (i - bw) + 1
       Previous row i-1:    bc_prev(j) = bc + 1
       This gives:
         H_prev(j-1) = Hp[bc],   H_prev(j) = Hp[bc+1]
         H_curr(j-1) = Hc[bc-1], H_curr(j) = Hc[bc]
       max_band_cols = 2*bw + 3 covers all indices. */

    int mbc = 2 * bw + 3;  // actual band cols needed for this pair's bw

    long long base = (long long)idx * 8 * max_band_cols;
    int* Hp  = workspace + base;
    int* Hc  = workspace + base + (long long)max_band_cols;
    int* Ep  = workspace + base + 2LL * max_band_cols;
    int* Ec  = workspace + base + 3LL * max_band_cols;
    int* Hmp = workspace + base + 4LL * max_band_cols;
    int* Hmc = workspace + base + 5LL * max_band_cols;
    int* Emp = workspace + base + 6LL * max_band_cols;
    int* Emc = workspace + base + 7LL * max_band_cols;

    // Initialize only the band cells (not full width!)
    for (int k = 0; k < mbc; k++) {
        Hp[k] = NEG_INF; Ep[k] = NEG_INF; Hmp[k] = 0; Emp[k] = 0;
        Hc[k] = NEG_INF; Ec[k] = NEG_INF; Hmc[k] = 0; Emc[k] = 0;
    }

    // Init first row: H(0,j) at bc = j - (0 - bw) + 1 = j + bw + 1
    Hp[bw + 1] = 0;  // H(0, 0)
    int j_end_init = cols;
    if (bw + 2 < j_end_init) j_end_init = bw + 2;
    for (int j = 1; j < j_end_init; j++) {
        Hp[j + bw + 1] = GAP_OPEN + GAP_EXTEND * (j - 1);
    }

    for (int i = 1; i <= len_a; i++) {
        int j_start = 1;
        if (i - bw > j_start) j_start = i - bw;
        int j_end = cols;
        if (i + bw + 1 < j_end) j_end = i + bw + 1;

        // Left border: bc(j_start - 1) = (j_start - 1) - (i - bw) + 1
        int bc_left = j_start - i + bw;
        if (j_start == 1) {
            Hc[bc_left] = GAP_OPEN + GAP_EXTEND * (i - 1);
            Hmc[bc_left] = 0;
        } else {
            Hc[bc_left] = NEG_INF;
            Hmc[bc_left] = 0;
        }

        // Right border of prev row
        int prev_j_end = cols;
        if (i + bw < prev_j_end) prev_j_end = i + bw;
        if (j_end > prev_j_end && prev_j_end < cols) {
            int bc_rp = prev_j_end - i + bw + 2; // bc_prev(prev_j_end)
            Hp[bc_rp] = NEG_INF; Ep[bc_rp] = NEG_INF;
            Hmp[bc_rp] = 0; Emp[bc_rp] = 0;
        }

        int curr_F = NEG_INF;
        int curr_Fm = 0;
        int max_Hm = 0;

        for (int j = j_start; j < j_end; j++) {
            int bc = j - i + bw + 1;  // band column for (i, j)

            int is_match = (seq_a[i-1] == seq_b[j-1]) ? 1 : 0;
            int s = is_match ? MATCH_SC : MISMATCH_SC;

            // Diagonal: H(i-1, j-1) -> Hp[bc]
            int diag   = Hp[bc] + s;
            int diag_m = Hmp[bc] + is_match;

            // E: vertical gap — H(i-1,j)->Hp[bc+1], E(i-1,j)->Ep[bc+1]
            int e_ext = Ep[bc+1] + GAP_EXTEND;
            int e_opn = Hp[bc+1] + GAP_OPEN;
            int e_val, e_m;
            if (e_ext >= e_opn) { e_val = e_ext; e_m = Emp[bc+1]; }
            else                { e_val = e_opn; e_m = Hmp[bc+1]; }
            Ec[bc] = e_val; Emc[bc] = e_m;

            // F: horizontal gap — H(i,j-1)->Hc[bc-1]
            int f_val = (j > j_start) ? (curr_F + GAP_EXTEND) : NEG_INF;
            int f_m = curr_Fm;
            int f_opn = Hc[bc-1] + GAP_OPEN;
            int f_opn_m = Hmc[bc-1];
            if (f_val >= f_opn) { curr_F = f_val; curr_Fm = f_m; }
            else                { curr_F = f_opn; curr_Fm = f_opn_m; }

            int best = diag; int best_m = diag_m;
            if (e_val > best)  { best = e_val;  best_m = e_m; }
            if (curr_F > best) { best = curr_F; best_m = curr_Fm; }

            Hc[bc] = best; Hmc[bc] = best_m;
            if (best_m > max_Hm) max_Hm = best_m;
        }

        int* tmp;
        tmp = Hp; Hp = Hc; Hc = tmp;
        tmp = Ep; Ep = Ec; Ec = tmp;
        tmp = Hmp; Hmp = Hmc; Hmc = tmp;
        tmp = Emp; Emp = Emc; Emc = tmp;

        int remaining = len_a - i;
        float max_possible = (float)(max_Hm + remaining) / (float)shorter;
        if (max_possible < threshold) {
            sims[idx] = 0.0f;
            mask_out[idx] = 0;
            return;
        }
    }

    // Final cell (len_a, len_b): bc = len_b - len_a + bw + 1
    int bc_final = len_b - len_a + bw + 1;
    float identity = (float)Hmp[bc_final] / (float)shorter;
    sims[idx] = identity;
    mask_out[idx] = (identity >= threshold) ? 1 : 0;
}
"""


def _compute_pairwise_alignment_gpu(
    candidate_pairs: np.ndarray,
    encoded_sequences: np.ndarray,
    lengths: np.ndarray,
    threshold: float,
    band_width: int,
    device_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU-accelerated banded NW alignment using a CuPy raw kernel.

    Uses band-relative indexing so workspace per thread is O(band_width),
    not O(sequence_length). Sorts pairs by adaptive band width and batches
    them for tight workspace allocation.
    """
    with cp.cuda.Device(device_id):
        m = len(candidate_pairs)
        max_len = encoded_sequences.shape[1]

        # Compute per-pair adaptive band width for sorting/batching
        len_i = lengths[candidate_pairs[:, 0]]
        len_j = lengths[candidate_pairs[:, 1]]
        len_diff = np.abs(len_i.astype(np.int32) - len_j.astype(np.int32))
        adaptive_bw = np.minimum(band_width, np.maximum(10, len_diff + 10))
        # For short sequences, bw = max(len_a, len_b)
        max_lens = np.maximum(len_i, len_j)
        short_mask = max_lens <= 50
        adaptive_bw[short_mask] = max_lens[short_mask]

        sort_order = np.argsort(adaptive_bw)
        sorted_pairs = candidate_pairs[sort_order]
        sorted_bw = adaptive_bw[sort_order]

        d_seqs = cp.asarray(encoded_sequences)
        d_lens = cp.asarray(lengths)
        d_pairs = cp.asarray(sorted_pairs)
        d_sims = cp.empty(m, dtype=cp.float32)
        d_mask = cp.empty(m, dtype=cp.int32)

        kernel = cp.RawKernel(_NW_ALIGN_KERNEL_CODE, "nw_align_kernel")
        threads = 128
        target_ws_bytes = 2_000_000_000  # ~2 GB workspace budget

        start = 0
        n_batches = 0
        while start < m:
            # max_band_cols for this batch = 2 * max_bw_in_batch + 3
            # Try a large batch first, then adjust
            end = min(start + 500_000, m)
            batch_max_bw = int(sorted_bw[end - 1])
            max_band_cols = 2 * batch_max_bw + 3
            ws_per_pair = 8 * max_band_cols * 4

            batch_size = min(end - start, max(1024, target_ws_bytes // ws_per_pair))
            end = start + batch_size
            batch_max_bw = int(sorted_bw[end - 1])
            max_band_cols = 2 * batch_max_bw + 3

            batch_m = end - start
            d_workspace = cp.empty(batch_m * 8 * max_band_cols, dtype=cp.int32)
            blocks = (batch_m + threads - 1) // threads

            kernel(
                (blocks,), (threads,),
                (d_pairs[start:end], d_seqs, d_lens,
                 d_sims[start:end], d_mask[start:end],
                 d_workspace,
                 np.int32(batch_m), np.int32(max_len),
                 np.int32(max_band_cols),
                 np.int32(band_width), np.float32(threshold)),
            )

            del d_workspace
            cp.cuda.Stream.null.synchronize()
            n_batches += 1
            start = end

        _logger.info(
            f"  GPU alignment: {m} pairs in {n_batches} batches, device={device_id}"
        )

        d_bool_mask = d_mask.astype(cp.bool_)
        filtered_pairs = cp.asnumpy(d_pairs[d_bool_mask])
        filtered_sims = cp.asnumpy(d_sims[d_bool_mask])

        return filtered_pairs, filtered_sims


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
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Jaccard similarity for all candidate pairs and filter by threshold.

    Args:
        candidate_pairs: (M, 2) int32 array of candidate pairs (i, j).
        sketches: (N, sketch_size) uint64 array of MinHash sketches.
        threshold: Minimum Jaccard similarity to keep a pair.
        device: "cpu" or GPU device ID (e.g., "0").

    Returns:
        Tuple of:
        - filtered_pairs: (K, 2) int32 array of pairs above threshold.
        - similarities: (K,) float32 array of Jaccard similarities.
    """
    if len(candidate_pairs) == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float32)

    if device != "cpu" and _CUPY_AVAILABLE:
        return _compute_pairwise_jaccard_gpu(
            candidate_pairs, sketches, threshold, int(device),
        )

    sims, mask = _batch_jaccard(candidate_pairs, sketches, np.float32(threshold))

    return candidate_pairs[mask], sims[mask]


# ──────────────────────────────────────────────────────────────────────
# GPU path for k-mer Jaccard (CuPy)
# ──────────────────────────────────────────────────────────────────────

# Raw CUDA kernel: one thread per candidate pair.
# Each thread walks the two sorted sketch arrays (merge-join) and
# computes the Jaccard similarity.
_JACCARD_KERNEL_CODE = r"""
extern "C" __global__
void jaccard_kernel(
    const int*                pairs,      // (M, 2) row-major — int32
    const unsigned long long* sketches,   // (N, sketch_size) row-major
    float*                    sims,       // (M,) output similarities
    int*                      mask_out,   // (M,) output: 1 if >= threshold
    int M,
    int sketch_size,
    float threshold
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= M) return;

    int si = pairs[idx * 2];
    int sj = pairs[idx * 2 + 1];

    const unsigned long long* sk_a = sketches + (long long)si * sketch_size;
    const unsigned long long* sk_b = sketches + (long long)sj * sketch_size;

    unsigned long long MAX_VAL = 0xFFFFFFFFFFFFFFFFULL;

    int shared = 0;
    int i = 0, j = 0;
    int union_count = 0;

    while (union_count < sketch_size && i < sketch_size && j < sketch_size) {
        unsigned long long a_val = sk_a[i];
        unsigned long long b_val = sk_b[j];

        if (a_val == MAX_VAL && b_val == MAX_VAL) break;

        if (a_val == b_val) {
            shared++;
            i++;
            j++;
        } else if (a_val < b_val) {
            i++;
        } else {
            j++;
        }
        union_count++;
    }

    float sim = (union_count > 0) ? ((float)shared / (float)union_count) : 0.0f;
    sims[idx] = sim;
    mask_out[idx] = (sim >= threshold) ? 1 : 0;
}
"""


def _compute_pairwise_jaccard_gpu(
    candidate_pairs: np.ndarray,
    sketches: np.ndarray,
    threshold: float,
    device_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU-accelerated pairwise Jaccard using a CuPy raw kernel."""
    with cp.cuda.Device(device_id):
        m = len(candidate_pairs)
        sketch_size = sketches.shape[1]

        d_pairs = cp.asarray(candidate_pairs)       # (M, 2) int32
        d_sketches = cp.asarray(sketches)            # (N, sketch_size) uint64
        d_sims = cp.empty(m, dtype=cp.float32)
        d_mask = cp.empty(m, dtype=cp.int32)

        kernel = cp.RawKernel(_JACCARD_KERNEL_CODE, "jaccard_kernel")
        threads_per_block = 256
        blocks = (m + threads_per_block - 1) // threads_per_block

        kernel(
            (blocks,), (threads_per_block,),
            (d_pairs, d_sketches, d_sims, d_mask,
             np.int32(m), np.int32(sketch_size), np.float32(threshold)),
        )

        # Filter on GPU
        d_bool_mask = d_mask.astype(cp.bool_)
        filtered_pairs = cp.asnumpy(d_pairs[d_bool_mask])
        filtered_sims = cp.asnumpy(d_sims[d_bool_mask])

        return filtered_pairs, filtered_sims
