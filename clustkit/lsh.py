"""Phase 2: LSH Bucketing — Find candidate pairs via locality-sensitive hashing.

Uses Numba JIT for hashing and pair extraction. Deduplication via sort-based
approach on packed int64 pairs instead of Python sets.
"""

import numpy as np
from numba import njit, prange, uint64, int32, int64


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
) -> np.ndarray:
    """Find candidate pairs using multi-probe LSH on sketch arrays.

    Args:
        sketches: (N, sketch_size) uint64 array of MinHash sketches.
        num_tables: Number of independent hash tables (L).
        num_bands: Number of sketch positions per band (b).
        seed: Base random seed for band selection.

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
