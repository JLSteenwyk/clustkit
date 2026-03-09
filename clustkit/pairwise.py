"""Phase 3: Pairwise Similarity — Compute identity for candidate pairs.

Two modes:
  - "kmer": fast k-mer Jaccard from MinHash sketches (good for >=60% identity)
  - "align": Needleman-Wunsch global alignment identity (accurate at all thresholds)

CPU uses Numba JIT and parallel batch processing.
GPU uses CuPy raw kernels (k-mer Jaccard). Alignment stays on CPU — banded
Needleman-Wunsch is inherently sequential (row-by-row DP) with per-thread
private workspace, making Numba JIT faster than GPU global memory access.
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


@njit(cache=True)
def _nw_identity(seq_a, len_a, seq_b, len_b, band_width, threshold):
    """Banded Needleman-Wunsch global alignment, returns sequence identity.

    Uses affine gap penalties with forward match counting (no traceback needed).
    Supports early termination when the threshold becomes unreachable.

    Identity = number of identical aligned residues / length of shorter sequence.

    Args:
        seq_a, seq_b: uint8 encoded sequences (padded arrays).
        len_a, len_b: actual lengths.
        band_width: half-width of the DP band (0 = full matrix).
        threshold: identity threshold for early termination (0 = disabled).

    Returns:
        Sequence identity as float32 in [0, 1].
    """
    # For very short sequences, return 0
    if len_a == 0 or len_b == 0:
        return float32(0.0)

    # Full DP if band_width is 0 or sequences are short
    if band_width <= 0 or max(len_a, len_b) <= 50:
        band_width = max(len_a, len_b)

    cols = len_b + 1
    shorter = min(len_a, len_b)
    NEG_INF = int32(-1000000)

    # Rolling arrays for scores and match counts (no traceback matrix needed)
    prev_H = np.full(cols, NEG_INF, dtype=np.int32)
    prev_E = np.full(cols, NEG_INF, dtype=np.int32)
    prev_Hm = np.zeros(cols, dtype=np.int32)
    prev_Em = np.zeros(cols, dtype=np.int32)

    curr_H = np.full(cols, NEG_INF, dtype=np.int32)
    curr_E = np.full(cols, NEG_INF, dtype=np.int32)
    curr_Hm = np.zeros(cols, dtype=np.int32)
    curr_Em = np.zeros(cols, dtype=np.int32)

    # Initialize first row
    prev_H[0] = int32(0)
    for j in range(1, cols):
        prev_H[j] = GAP_OPEN + GAP_EXTEND * int32(j - 1)

    # Fill DP
    for i in range(1, len_a + 1):
        j_start = max(1, i - band_width)
        j_end = min(cols, i + band_width + 1)

        for j in range(cols):
            curr_H[j] = NEG_INF
            curr_E[j] = NEG_INF
            curr_Hm[j] = int32(0)
            curr_Em[j] = int32(0)

        if j_start == 1:
            curr_H[0] = GAP_OPEN + GAP_EXTEND * int32(i - 1)
            curr_Hm[0] = int32(0)

        curr_F = int32(NEG_INF)
        curr_Fm = int32(0)
        max_Hm = int32(0)

        for j in range(j_start, j_end):
            is_match = int32(seq_a[i - 1] == seq_b[j - 1])
            s = MATCH_SCORE if is_match else MISMATCH_SCORE

            diag = prev_H[j - 1] + s
            diag_m = prev_Hm[j - 1] + is_match

            # E: gap in seq_b
            e_extend = prev_E[j] + GAP_EXTEND
            e_open = prev_H[j] + GAP_OPEN
            if e_extend >= e_open:
                curr_E[j] = e_extend
                curr_Em[j] = prev_Em[j]
            else:
                curr_E[j] = e_open
                curr_Em[j] = prev_Hm[j]

            # F: gap in seq_a
            f_extend = curr_F + GAP_EXTEND if j > j_start else NEG_INF
            f_extend_m = curr_Fm
            f_open = curr_H[j - 1] + GAP_OPEN
            f_open_m = curr_Hm[j - 1]
            if f_extend >= f_open:
                curr_F = f_extend
                curr_Fm = f_extend_m
            else:
                curr_F = f_open
                curr_Fm = f_open_m

            # Best score (same tie-breaking as before: diag > E > F)
            best = diag
            best_m = diag_m
            if curr_E[j] > best:
                best = curr_E[j]
                best_m = curr_Em[j]
            if curr_F > best:
                best = curr_F
                best_m = curr_Fm

            curr_H[j] = best
            curr_Hm[j] = best_m

            if best_m > max_Hm:
                max_Hm = best_m

        # Swap rows
        for j in range(cols):
            prev_H[j] = curr_H[j]
            prev_E[j] = curr_E[j]
            prev_Hm[j] = curr_Hm[j]
            prev_Em[j] = curr_Em[j]

        # Early termination: even if all remaining rows produce matches,
        # can we still reach the threshold?
        remaining = len_a - i
        max_possible = float32((max_Hm + remaining) / shorter)
        if max_possible < threshold:
            return float32(0.0)

    # Identity from match count at (len_a, len_b)
    return float32(prev_Hm[len_b] / shorter)


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
            threshold,
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
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pairwise identity via alignment for candidate pairs.

    Args:
        candidate_pairs: (M, 2) int32 array of candidate pairs (i, j).
        encoded_sequences: (N, max_len) uint8 matrix.
        lengths: (N,) int32 array of sequence lengths.
        threshold: Minimum sequence identity to keep a pair.
        band_width: Half-width of the alignment band (0 = full).
        device: "cpu" or GPU device ID. Alignment always runs on CPU;
                if a GPU device is requested a log message is emitted.

    Returns:
        Tuple of:
        - filtered_pairs: (K, 2) int32 array of pairs above threshold.
        - identities: (K,) float32 array of sequence identities.
    """
    if len(candidate_pairs) == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float32)

    if device != "cpu":
        _logger.info(
            "  Banded NW alignment runs on CPU (Numba JIT is faster than GPU "
            "for sequential DP workloads). Use --alignment kmer for full GPU acceleration."
        )

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
