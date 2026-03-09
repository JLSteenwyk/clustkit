"""Phase 2: LSH Bucketing — Find candidate pairs via locality-sensitive hashing.

Uses Numba JIT for hashing and pair extraction on CPU.
Uses CuPy vectorized operations on GPU.

Deduplication via sort-based approach on packed int64 pairs instead of Python sets.
"""

import numpy as np
from numba import njit, prange, uint64, int32, int64

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


@njit(uint64(uint64[:], int32[:], uint64), cache=True)
def _hash_band_numba(sketch, band_indices, seed):
    """Hash a band of sketch values into a single bucket ID."""
    h = seed
    for idx in band_indices:
        val = sketch[idx]
        h = h * uint64(0x517CC1B727220A95) + val
    h = (h ^ (h >> uint64(33))) * uint64(0xC4CEB9FE1A85EC53)
    h = h ^ (h >> uint64(33))
    return h


@njit(parallel=True, cache=True)
def _hash_all_tables(sketches, all_band_indices, all_seeds, num_tables):
    """Hash all sequences across all tables at once.

    Returns (num_tables, N) array of bucket IDs.
    """
    n = sketches.shape[0]
    num_bands = all_band_indices.shape[1]
    result = np.empty((num_tables, n), dtype=np.uint64)

    for ti in prange(num_tables):
        band_indices = all_band_indices[ti]
        seed = uint64(all_seeds[ti])
        for i in range(n):
            result[ti, i] = _hash_band_numba(sketches[i], band_indices, seed)

    return result


@njit(cache=True)
def _extract_pairs_from_sorted(order, sorted_ids, n, max_bucket):
    """Extract candidate pairs from sorted bucket IDs.

    Returns pairs packed as int64: pair = i * N + j (where i < j).
    """
    # First pass: count pairs to pre-allocate
    total_pairs = int64(0)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_ids[j] == sorted_ids[i]:
            j += 1
        bucket_size = min(j - i, max_bucket)
        total_pairs += int64(bucket_size) * int64(bucket_size - 1) // int64(2)
        i = j

    if total_pairs == 0:
        return np.empty(0, dtype=np.int64)

    packed = np.empty(total_pairs, dtype=np.int64)
    write_pos = 0

    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_ids[j] == sorted_ids[i]:
            j += 1
        end = min(j, i + max_bucket)
        for a in range(i, end):
            for b in range(a + 1, end):
                x = int64(order[a])
                y = int64(order[b])
                if x > y:
                    x, y = y, x
                packed[write_pos] = x * int64(n) + y
                write_pos += 1
        i = j

    return packed[:write_pos]


def lsh_candidates(
    sketches: np.ndarray,
    num_tables: int,
    num_bands: int,
    seed: int = 42,
    device: str = "cpu",
) -> np.ndarray:
    """Find candidate pairs using multi-probe LSH on sketch arrays.

    Args:
        sketches: (N, sketch_size) uint64 array of MinHash sketches.
        num_tables: Number of independent hash tables (L).
        num_bands: Number of sketch positions per band (b).
        seed: Base random seed for band selection.
        device: "cpu" or GPU device ID (e.g., "0").

    Returns:
        (M, 2) int32 array of deduplicated candidate pairs (i, j) where i < j.
    """
    n, sketch_size = sketches.shape
    if n < 2:
        return np.empty((0, 2), dtype=np.int32)

    rng = np.random.RandomState(seed)

    # Pre-generate all band indices and seeds
    all_band_indices = np.empty((num_tables, num_bands), dtype=np.int32)
    all_seeds = np.empty(num_tables, dtype=np.int64)
    for t in range(num_tables):
        all_band_indices[t] = rng.choice(sketch_size, size=num_bands, replace=False).astype(np.int32)
        all_seeds[t] = int(rng.randint(0, 2**31))

    if device != "cpu" and _CUPY_AVAILABLE:
        return _lsh_candidates_gpu(
            sketches, all_band_indices, all_seeds, num_tables, n,
            int(device),
        )

    return _lsh_candidates_cpu(
        sketches, all_band_indices, all_seeds, num_tables, n,
    )


def _lsh_candidates_cpu(
    sketches: np.ndarray,
    all_band_indices: np.ndarray,
    all_seeds: np.ndarray,
    num_tables: int,
    n: int,
) -> np.ndarray:
    """CPU path for LSH candidate generation."""
    # Batch hash all sequences across all tables (parallel over tables)
    all_bucket_ids = _hash_all_tables(sketches, all_band_indices, all_seeds, num_tables)

    # Extract pairs table by table (argsort + Numba pair extraction)
    all_packed = []
    for t in range(num_tables):
        bucket_ids = all_bucket_ids[t]
        order = np.argsort(bucket_ids)
        sorted_ids = bucket_ids[order]

        packed = _extract_pairs_from_sorted(order, sorted_ids, n, 1000)
        if len(packed) > 0:
            all_packed.append(packed)

    if not all_packed:
        return np.empty((0, 2), dtype=np.int32)

    # Concatenate, sort, deduplicate
    all_packed = np.concatenate(all_packed)
    all_packed.sort()

    # Unique via consecutive-diff mask
    mask = np.empty(len(all_packed), dtype=np.bool_)
    mask[0] = True
    mask[1:] = all_packed[1:] != all_packed[:-1]
    unique_packed = all_packed[mask]

    # Unpack to (i, j) pairs
    pairs = np.empty((len(unique_packed), 2), dtype=np.int32)
    pairs[:, 0] = (unique_packed // n).astype(np.int32)
    pairs[:, 1] = (unique_packed % n).astype(np.int32)

    return pairs


# ──────────────────────────────────────────────────────────────────────
# GPU path (CuPy)
# ──────────────────────────────────────────────────────────────────────

# Raw CUDA kernel: each thread hashes one (table, sequence) pair.
# Grid dimensions: (num_tables, ceil(N / block_size)).
_LSH_HASH_KERNEL_CODE = r"""
extern "C" __global__
void lsh_hash_kernel(
    const unsigned long long* sketches,      // (N, sketch_size) row-major
    const int*                band_indices,   // (num_tables, num_bands) row-major
    const long long*          seeds,          // (num_tables,)
    unsigned long long*       bucket_ids,     // (num_tables, N) output
    int N,
    int sketch_size,
    int num_bands,
    int num_tables
) {
    int ti = blockIdx.y;                                     // table index
    int si = blockIdx.x * blockDim.x + threadIdx.x;         // sequence index
    if (ti >= num_tables || si >= N) return;

    const int* indices = band_indices + ti * num_bands;
    unsigned long long seed = (unsigned long long)seeds[ti];
    const unsigned long long* sk = sketches + (long long)si * sketch_size;

    unsigned long long h = seed;
    for (int b = 0; b < num_bands; b++) {
        unsigned long long val = sk[indices[b]];
        h = h * 0x517CC1B727220A95ULL + val;
    }
    h = (h ^ (h >> 33)) * 0xC4CEB9FE1A85EC53ULL;
    h = h ^ (h >> 33);

    bucket_ids[(long long)ti * N + si] = h;
}
"""


def _lsh_candidates_gpu(
    sketches: np.ndarray,
    all_band_indices: np.ndarray,
    all_seeds: np.ndarray,
    num_tables: int,
    n: int,
    device_id: int,
) -> np.ndarray:
    """GPU-accelerated LSH candidate generation using CuPy.

    Hashing is done on GPU. Pair extraction uses GPU sort + vectorised
    bucket-boundary detection, then falls back to CPU for the combinatorial
    pair enumeration within buckets (which is inherently variable-length).
    """
    with cp.cuda.Device(device_id):
        sketch_size = sketches.shape[1]
        num_bands = all_band_indices.shape[1]

        d_sketches = cp.asarray(sketches)
        d_band_indices = cp.asarray(all_band_indices)
        d_seeds = cp.asarray(all_seeds)
        d_bucket_ids = cp.empty((num_tables, n), dtype=cp.uint64)

        kernel = cp.RawKernel(_LSH_HASH_KERNEL_CODE, "lsh_hash_kernel")
        threads_per_block = 256
        blocks_x = (n + threads_per_block - 1) // threads_per_block

        kernel(
            (blocks_x, num_tables), (threads_per_block,),
            (d_sketches, d_band_indices, d_seeds, d_bucket_ids,
             np.int32(n), np.int32(sketch_size), np.int32(num_bands),
             np.int32(num_tables)),
        )

        # For each table: GPU argsort + CPU pair extraction
        # (pair extraction is inherently serial due to variable bucket sizes)
        all_packed = []
        for t in range(num_tables):
            d_ids = d_bucket_ids[t]
            d_order = cp.argsort(d_ids)
            d_sorted = d_ids[d_order]

            # Transfer to CPU for pair extraction
            order = cp.asnumpy(d_order).astype(np.int64)
            sorted_ids = cp.asnumpy(d_sorted)

            packed = _extract_pairs_from_sorted(order, sorted_ids, n, 1000)
            if len(packed) > 0:
                all_packed.append(packed)

        if not all_packed:
            return np.empty((0, 2), dtype=np.int32)

        # Deduplicate on GPU: concatenate, sort, unique
        d_all_packed = cp.asarray(np.concatenate(all_packed))
        d_all_packed = cp.sort(d_all_packed)
        # Unique via consecutive-diff mask
        d_mask = cp.empty(len(d_all_packed), dtype=cp.bool_)
        d_mask[0] = True
        d_mask[1:] = d_all_packed[1:] != d_all_packed[:-1]
        d_unique = d_all_packed[d_mask]

        # Unpack to (i, j) pairs on GPU
        d_pairs_i = (d_unique // n).astype(cp.int32)
        d_pairs_j = (d_unique % n).astype(cp.int32)
        d_pairs = cp.stack([d_pairs_i, d_pairs_j], axis=1)

        return cp.asnumpy(d_pairs)
