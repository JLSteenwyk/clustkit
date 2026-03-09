"""Phase 2: LSH Bucketing — Find candidate pairs via locality-sensitive hashing (CPU reference)."""

import warnings

import numpy as np
from collections import defaultdict

# Integer overflow is expected and intentional in hash functions (modular arithmetic)
warnings.filterwarnings("ignore", message="overflow encountered", category=RuntimeWarning)


def _hash_band(sketch: np.ndarray, band_indices: np.ndarray, seed: int = 0) -> int:
    """Hash a band of sketch values into a single bucket ID.

    Uses a simple polynomial rolling hash on the selected sketch entries.
    """
    h = np.uint64(seed)
    for idx in band_indices:
        val = np.uint64(sketch[idx])
        h = np.uint64(h * np.uint64(0x517CC1B727220A95) + val)
    # Finalize
    h = np.uint64((h ^ (h >> np.uint64(13))) * np.uint64(0xC2B2AE35))
    h = np.uint64(h ^ (h >> np.uint64(16)))
    return int(h)


def lsh_candidates(
    sketches: np.ndarray,
    num_tables: int,
    num_bands: int,
    seed: int = 42,
) -> np.ndarray:
    """Find candidate pairs using multi-probe LSH on sketch arrays.

    For each of `num_tables` hash tables, selects `num_bands` positions from the
    sketch, hashes them into a bucket, and records co-occurring sequences as
    candidate pairs.

    Args:
        sketches: (N, sketch_size) uint64 array of MinHash sketches.
        num_tables: Number of independent hash tables (L).
        num_bands: Number of sketch positions per band (b).
        seed: Base random seed for band selection.

    Returns:
        (M, 2) int32 array of deduplicated candidate pairs (i, j) where i < j.
    """
    n, sketch_size = sketches.shape
    rng = np.random.RandomState(seed)
    candidate_set: set[tuple[int, int]] = set()

    for t in range(num_tables):
        # Select which sketch positions form this band
        band_indices = rng.choice(sketch_size, size=num_bands, replace=False).astype(np.int32)
        table_seed = int(rng.randint(0, 2**31))

        # Build buckets for this hash table
        buckets: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            bucket_id = _hash_band(sketches[i], band_indices, seed=table_seed)
            buckets[bucket_id].append(i)

        # All pairs within each bucket are candidates
        for members in buckets.values():
            if len(members) < 2:
                continue
            # Cap bucket size to avoid quadratic blowup from degenerate buckets
            if len(members) > 1000:
                members = members[:1000]
            for a_idx in range(len(members)):
                for b_idx in range(a_idx + 1, len(members)):
                    i, j = members[a_idx], members[b_idx]
                    if i > j:
                        i, j = j, i
                    candidate_set.add((i, j))

    if not candidate_set:
        return np.empty((0, 2), dtype=np.int32)

    pairs = np.array(sorted(candidate_set), dtype=np.int32)
    return pairs
