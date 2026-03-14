"""Sequence search: query-vs-database search using MinHash + LSH + banded NW.

Multi-stage filtering pipeline inspired by MMseqs2:
  Stage 1: LSH prefilter -- hash query and db sketches into the same tables,
           find query->db pairs sharing at least one bucket.
  Stage 2: Jaccard estimate from MinHash sketches -- discard pairs with
           near-zero sketch overlap that cannot reach the identity threshold.
  Stage 3: Banded Needleman-Wunsch alignment -- accurate identity computation
           with early termination.

Results are returned as top-k hits per query, sorted by identity descending.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit, prange, int32, int64, uint64, float32

from clustkit.io import SequenceDataset
from clustkit.sketch import compute_sketches
from clustkit.lsh import _hash_all_tables
from clustkit.pairwise import (
    _batch_align,
    _batch_jaccard,
    _nw_identity,
)
from clustkit.utils import auto_kmer_for_lsh, auto_lsh_params, logger, timer

_logger = logging.getLogger("clustkit")


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SearchHit:
    """A single search hit."""

    query_id: str
    target_id: str
    identity: float
    query_length: int
    target_length: int
    score: float = 0.0  # raw alignment score (for ranking)


@dataclass
class SearchResults:
    """Results for all queries."""

    hits: list[list[SearchHit]]  # hits[i] = list of hits for query i, sorted by identity desc
    num_queries: int
    num_targets: int
    num_candidates: int  # total LSH candidates before filtering
    num_aligned: int  # pairs that entered alignment
    runtime_seconds: float


# ──────────────────────────────────────────────────────────────────────
# LSH query-vs-database candidate generation
# ──────────────────────────────────────────────────────────────────────

def _generate_lsh_params(sketch_size: int, num_tables: int, num_bands: int, seed: int = 42):
    """Generate band indices and seeds for LSH tables.

    Uses the same RNG procedure as ``lsh_candidates`` in ``clustkit.lsh``
    so that pre-built LSH indices are compatible.

    Returns:
        (all_band_indices, all_seeds) -- arrays ready for ``_hash_all_tables``.
    """
    rng = np.random.RandomState(seed)
    all_band_indices = np.empty((num_tables, num_bands), dtype=np.int32)
    all_seeds = np.empty(num_tables, dtype=np.int64)
    for t in range(num_tables):
        all_band_indices[t] = rng.choice(
            sketch_size, size=num_bands, replace=False
        ).astype(np.int32)
        all_seeds[t] = int(rng.randint(0, 2**31))
    return all_band_indices, all_seeds


def _lsh_query_candidates(
    query_sketches: np.ndarray,
    db_sketches: np.ndarray,
    num_tables: int,
    num_bands: int,
    seed: int = 42,
    max_bucket: int = 1000,
) -> np.ndarray:
    """Find candidate pairs between query and database sequences using LSH.

    Hashes both query and db sketches into the same LSH tables (same band
    indices, same seeds), then finds cross-set pairs: for each table, any
    query and db sequence that share a bucket become a candidate pair.

    Only query->db pairs are returned (no query-query or db-db pairs).

    Args:
        query_sketches: (Nq, sketch_size) uint64 array.
        db_sketches: (Nd, sketch_size) uint64 array.
        num_tables: Number of independent hash tables (L).
        num_bands: Number of sketch positions per band (b).
        seed: Base random seed for band selection (must match db index if reusing).
        max_bucket: Maximum bucket size to enumerate (caps combinatorial explosion).

    Returns:
        (M, 2) int32 array where col 0 = query index, col 1 = db index.
        Pairs are deduplicated.
    """
    nq = query_sketches.shape[0]
    nd = db_sketches.shape[0]
    sketch_size = query_sketches.shape[1]

    if nq == 0 or nd == 0:
        return np.empty((0, 2), dtype=np.int32)

    # Generate LSH parameters (deterministic, matches lsh.py convention)
    all_band_indices, all_seeds = _generate_lsh_params(
        sketch_size, num_tables, num_bands, seed
    )

    # Hash both query and db sketches into the same tables
    query_bucket_ids = _hash_all_tables(
        query_sketches, all_band_indices, all_seeds, num_tables
    )  # (num_tables, nq)
    db_bucket_ids = _hash_all_tables(
        db_sketches, all_band_indices, all_seeds, num_tables
    )  # (num_tables, nd)

    # For each table, find cross-set pairs with incremental dedup
    unique_set: set[int] = set()
    for t in range(num_tables):
        packed = _extract_cross_pairs_for_table(
            query_bucket_ids[t], db_bucket_ids[t], nq, nd, max_bucket
        )
        if len(packed) > 0:
            unique_set.update(packed.tolist())

    if not unique_set:
        return np.empty((0, 2), dtype=np.int32)

    unique_packed = np.fromiter(unique_set, dtype=np.int64, count=len(unique_set))

    # Unpack: pair = query_idx * nd + db_idx
    pairs = np.empty((len(unique_packed), 2), dtype=np.int32)
    pairs[:, 0] = (unique_packed // nd).astype(np.int32)
    pairs[:, 1] = (unique_packed % nd).astype(np.int32)

    return pairs


@njit(cache=True)
def _extract_cross_pairs_for_table_numba(
    query_buckets,
    db_buckets,
    db_order,
    db_sorted_buckets,
    db_bucket_starts,
    db_bucket_ends,
    db_unique_buckets,
    nd,
    max_bucket,
):
    """Numba-accelerated cross-pair extraction for one LSH table.

    Uses pre-sorted db arrays and binary search for O(nq * log(B) + output)
    instead of Python dict lookups.

    Returns packed int64 array: pair = query_idx * nd + db_idx.
    """
    n_unique = len(db_unique_buckets)
    nq = len(query_buckets)

    # First pass: count pairs to pre-allocate output
    total = int64(0)
    for qi in range(nq):
        q_bucket = query_buckets[qi]
        # Binary search for q_bucket in db_unique_buckets
        lo = int64(0)
        hi = int64(n_unique)
        found = int64(-1)
        while lo < hi:
            mid = (lo + hi) >> 1
            if db_unique_buckets[mid] < q_bucket:
                lo = mid + 1
            elif db_unique_buckets[mid] > q_bucket:
                hi = mid
            else:
                found = mid
                break
        if found >= 0:
            bucket_size = min(
                db_bucket_ends[found] - db_bucket_starts[found],
                int64(max_bucket),
            )
            total += bucket_size

    if total == 0:
        return np.empty(0, dtype=np.int64)

    # Second pass: fill output
    out = np.empty(total, dtype=np.int64)
    pos = int64(0)
    for qi in range(nq):
        q_bucket = query_buckets[qi]
        lo = int64(0)
        hi = int64(n_unique)
        found = int64(-1)
        while lo < hi:
            mid = (lo + hi) >> 1
            if db_unique_buckets[mid] < q_bucket:
                lo = mid + 1
            elif db_unique_buckets[mid] > q_bucket:
                hi = mid
            else:
                found = mid
                break
        if found >= 0:
            start = db_bucket_starts[found]
            end = min(db_bucket_ends[found], start + int64(max_bucket))
            for j in range(start, end):
                di = db_order[j]
                out[pos] = int64(qi) * int64(nd) + int64(di)
                pos += 1

    return out[:pos]


def _preprocess_db_buckets(db_buckets):
    """Sort db bucket IDs and compute bucket boundaries.

    Returns (db_order, db_sorted, unique_buckets, starts, ends) — all numpy
    arrays ready for the Numba kernel.  Reusable across tables when db bucket
    IDs change per table but format stays the same.
    """
    db_order = np.argsort(db_buckets)
    db_sorted = db_buckets[db_order]
    nd = len(db_sorted)

    # Find bucket boundaries
    if nd == 0:
        return (
            db_order,
            db_sorted,
            np.empty(0, dtype=db_sorted.dtype),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    # Indices where value changes
    change = np.empty(nd, dtype=np.bool_)
    change[0] = True
    change[1:] = db_sorted[1:] != db_sorted[:-1]
    starts = np.nonzero(change)[0].astype(np.int64)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = int64(nd)
    unique_buckets = db_sorted[starts]

    return db_order, db_sorted, unique_buckets, starts, ends


def _extract_cross_pairs_for_table(
    query_buckets: np.ndarray,
    db_buckets: np.ndarray,
    nq: int,
    nd: int,
    max_bucket: int,
) -> np.ndarray:
    """Extract query->db pairs sharing a bucket in one LSH table.

    Uses Numba-accelerated sort-merge with binary search.
    Returns packed int64 array: pair = query_idx * nd + db_idx.
    """
    if nq == 0 or nd == 0:
        return np.empty(0, dtype=np.int64)

    db_order, db_sorted, unique_buckets, starts, ends = _preprocess_db_buckets(
        db_buckets
    )

    return _extract_cross_pairs_for_table_numba(
        query_buckets, db_buckets, db_order, db_sorted,
        starts, ends, unique_buckets, int64(nd), int64(max_bucket),
    )


# ──────────────────────────────────────────────────────────────────────
# Jaccard pre-filter for search (Stage 2)
# ──────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True)
def _batch_jaccard_cross(
    pairs: np.ndarray,
    query_sketches: np.ndarray,
    db_sketches: np.ndarray,
    threshold: float,
):
    """Compute Jaccard similarity for cross-set (query, db) pairs.

    Unlike ``_batch_jaccard`` which indexes both columns from the same sketch
    matrix, this function indexes col 0 from query_sketches and col 1 from
    db_sketches.

    Args:
        pairs: (M, 2) int32 -- col 0 = query idx, col 1 = db idx.
        query_sketches: (Nq, sketch_size) uint64.
        db_sketches: (Nd, sketch_size) uint64.
        threshold: Jaccard threshold for the mask.

    Returns:
        (sims, mask) -- float32 and bool arrays of length M.
    """
    m = pairs.shape[0]
    sketch_size = query_sketches.shape[1]
    max_val = uint64(0xFFFFFFFFFFFFFFFF)
    sims = np.empty(m, dtype=np.float32)
    mask = np.empty(m, dtype=np.bool_)

    for idx in prange(m):
        qi = pairs[idx, 0]
        di = pairs[idx, 1]
        sketch_a = query_sketches[qi]
        sketch_b = db_sketches[di]

        # Merge-based Jaccard on sorted sketches
        shared = int32(0)
        ia = int32(0)
        ib = int32(0)
        union_count = int32(0)
        s = int32(sketch_size)

        while union_count < s and ia < s and ib < s:
            a_val = sketch_a[ia]
            b_val = sketch_b[ib]
            if a_val == max_val and b_val == max_val:
                break
            if a_val == b_val:
                shared += int32(1)
                ia += int32(1)
                ib += int32(1)
            elif a_val < b_val:
                ia += int32(1)
            else:
                ib += int32(1)
            union_count += int32(1)

        if union_count == 0:
            sim = float32(0.0)
        else:
            sim = float32(float32(shared) / float32(union_count))

        sims[idx] = sim
        mask[idx] = sim >= float32(threshold)

    return sims, mask


def _jaccard_prefilter_cross(
    pairs: np.ndarray,
    query_sketches: np.ndarray,
    db_sketches: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Pre-filter candidate pairs using Jaccard similarity (cross-set variant).

    Uses a conservative Jaccard floor well below the expected value for true
    positives at the given identity threshold.

    Args:
        pairs: (M, 2) int32 -- col 0 = query idx, col 1 = db idx.
        query_sketches: (Nq, sketch_size) uint64.
        db_sketches: (Nd, sketch_size) uint64.
        threshold: Identity threshold.

    Returns:
        Boolean mask of pairs that pass the Jaccard pre-filter.
    """
    m = len(pairs)
    if m == 0:
        return np.ones(0, dtype=np.bool_)

    # Conservative Jaccard floor: well below minimum expected for true pairs
    jaccard_floor = max(0.001, 0.3 * threshold**5)

    sims, _ = _batch_jaccard_cross(
        pairs, query_sketches, db_sketches, np.float32(jaccard_floor)
    )

    return sims >= jaccard_floor


# ──────────────────────────────────────────────────────────────────────
# Alignment on merged sequence matrix (Stage 3)
# ──────────────────────────────────────────────────────────────────────

def _merge_sequences_for_alignment(
    query_dataset: SequenceDataset,
    db_dataset: SequenceDataset,
):
    """Merge query and db encoded sequences into a single compact representation.

    Query sequences come first (indices 0..Nq-1), then db sequences
    (indices Nq..Nq+Nd-1).

    Returns compact format when both datasets have it, otherwise padded matrix.

    Returns:
        dict with keys:
        - 'lengths': merged (N,) int32 array
        - 'nq', 'nd': counts
        - 'flat_sequences', 'offsets': compact format (or None)
        - 'encoded_sequences': padded matrix (or None)
    """
    nq = query_dataset.num_sequences
    nd = db_dataset.num_sequences
    q_lens = query_dataset.lengths
    d_lens = db_dataset.lengths
    merged_lengths = np.concatenate([q_lens, d_lens]).astype(np.int32)

    # Prefer compact format
    q_flat = query_dataset.flat_sequences
    q_off = query_dataset.offsets
    d_flat = db_dataset.flat_sequences
    d_off = db_dataset.offsets

    if q_flat is not None and q_off is not None and d_flat is not None and d_off is not None:
        # Concatenate flat arrays, shift db offsets
        merged_flat = np.concatenate([q_flat, d_flat])
        db_offsets_shifted = d_off + len(q_flat)
        merged_offsets = np.concatenate([q_off, db_offsets_shifted]).astype(np.int64)
        return {
            'lengths': merged_lengths,
            'nq': nq, 'nd': nd,
            'flat_sequences': merged_flat,
            'offsets': merged_offsets,
            'encoded_sequences': None,
        }

    # Fallback: padded matrix
    q_seqs = query_dataset.encoded_sequences
    d_seqs = db_dataset.encoded_sequences
    max_len = max(
        q_seqs.shape[1] if q_seqs.shape[1] > 0 else 0,
        d_seqs.shape[1] if d_seqs.shape[1] > 0 else 0,
    )
    total = nq + nd
    merged = np.zeros((total, max_len), dtype=np.uint8)
    merged[:nq, : q_seqs.shape[1]] = q_seqs
    merged[nq:, : d_seqs.shape[1]] = d_seqs

    return {
        'lengths': merged_lengths,
        'nq': nq, 'nd': nd,
        'flat_sequences': None,
        'offsets': None,
        'encoded_sequences': merged,
    }


def _remap_pairs_to_merged(pairs: np.ndarray, nq_offset: int) -> np.ndarray:
    """Remap (query_idx, db_idx) pairs to merged-matrix indices.

    In the merged matrix, query indices are [0, nq) and db indices are
    [nq, nq+nd). The input pairs have col 0 = query local idx, col 1 = db
    local idx.

    Returns:
        (M, 2) int32 array with merged-matrix indices.
    """
    remapped = np.empty_like(pairs)
    remapped[:, 0] = pairs[:, 0]  # query indices stay as-is
    remapped[:, 1] = pairs[:, 1] + np.int32(nq_offset)  # shift db indices
    return remapped


# ──────────────────────────────────────────────────────────────────────
# Main search function
# ──────────────────────────────────────────────────────────────────────

def search_sequences(
    query_dataset: SequenceDataset,
    db_dataset: SequenceDataset,
    threshold: float = 0.5,
    top_k: int = 10,
    mode: str = "protein",
    kmer_size: int = 5,
    sketch_size: int = 128,
    sensitivity: str = "high",
    band_width: int | None = None,
    device: str = "cpu",
    db_sketches: np.ndarray | None = None,
    db_lsh_tables: dict | None = None,
) -> SearchResults:
    """Search query sequences against a database using MinHash + LSH + banded NW.

    Multi-stage pipeline:
      1. Sketch query and database sequences (MinHash).
      2. LSH candidate generation (query->db pairs only).
      3. Jaccard pre-filter on candidate pairs.
      4. Banded Needleman-Wunsch alignment on surviving candidates.
      5. Collect top-k hits per query, sorted by identity descending.

    Args:
        query_dataset: Query sequences as a SequenceDataset.
        db_dataset: Database sequences as a SequenceDataset.
        threshold: Minimum sequence identity to report a hit (0.0-1.0).
        top_k: Maximum number of hits to return per query.
        mode: "protein" or "nucleotide".
        kmer_size: K-mer size for sketching.
        sketch_size: Number of MinHash values per sketch.
        sensitivity: LSH sensitivity: "low", "medium", or "high".
        band_width: Half-width of the alignment band. If None, auto-computed
            from the 95th percentile of sequence lengths.
        device: "cpu" or GPU device ID (e.g., "0").
        db_sketches: Pre-computed database sketches. If None, computed here.
        db_lsh_tables: Reserved for future pre-built LSH index support.

    Returns:
        SearchResults with top-k hits per query.
    """
    start_time = time.perf_counter()

    nq = query_dataset.num_sequences
    nd = db_dataset.num_sequences

    _logger.info(f"Search: {nq} queries against {nd} database sequences")

    if nq == 0 or nd == 0:
        _logger.warning("Empty query or database set. No search results.")
        return SearchResults(
            hits=[[] for _ in range(nq)],
            num_queries=nq,
            num_targets=nd,
            num_candidates=0,
            num_aligned=0,
            runtime_seconds=time.perf_counter() - start_time,
        )

    # ── Adaptive k-mer size for LSH recall ──
    k_lsh = auto_kmer_for_lsh(threshold, mode, kmer_size)
    if k_lsh != kmer_size:
        _logger.info(
            f"  Adaptive k: using k={k_lsh} for LSH (threshold={threshold})"
        )

    # ── Stage 0: Sketch ──
    with timer("Search Stage 0: Sketching"):
        if db_sketches is None:
            _logger.info(f"  Sketching {nd} database sequences (k={k_lsh}, s={sketch_size})")
            db_sketches_computed = compute_sketches(
                db_dataset.encoded_sequences,
                db_dataset.lengths,
                k_lsh,
                sketch_size,
                mode,
                device=device,
                flat_sequences=db_dataset.flat_sequences,
                offsets=db_dataset.offsets,
            )
        else:
            db_sketches_computed = db_sketches
            _logger.info(f"  Using pre-computed database sketches ({db_sketches_computed.shape})")

        _logger.info(f"  Sketching {nq} query sequences (k={k_lsh}, s={sketch_size})")
        query_sketches = compute_sketches(
            query_dataset.encoded_sequences,
            query_dataset.lengths,
            k_lsh,
            sketch_size,
            mode,
            device=device,
            flat_sequences=query_dataset.flat_sequences,
            offsets=query_dataset.offsets,
        )

    # ── Stage 1: LSH candidate generation ──
    lsh_params = auto_lsh_params(threshold, sensitivity, k=k_lsh)
    _logger.info(
        f"  LSH params: {lsh_params['num_tables']} tables, "
        f"{lsh_params['num_bands']} bands/table"
    )

    with timer("Search Stage 1: LSH candidate generation"):
        candidate_pairs = _lsh_query_candidates(
            query_sketches,
            db_sketches_computed,
            num_tables=lsh_params["num_tables"],
            num_bands=lsh_params["num_bands"],
        )

    num_candidates = len(candidate_pairs)
    _logger.info(f"  Found {num_candidates} query->db candidate pairs")

    if num_candidates == 0:
        _logger.info("  No candidates found. Returning empty results.")
        return SearchResults(
            hits=[[] for _ in range(nq)],
            num_queries=nq,
            num_targets=nd,
            num_candidates=0,
            num_aligned=0,
            runtime_seconds=time.perf_counter() - start_time,
        )

    # ── Stage 2: Jaccard pre-filter ──
    with timer("Search Stage 2: Jaccard pre-filter"):
        jaccard_mask = _jaccard_prefilter_cross(
            candidate_pairs, query_sketches, db_sketches_computed, threshold
        )
        pairs_before = len(candidate_pairs)
        candidate_pairs = candidate_pairs[jaccard_mask]
        pairs_after = len(candidate_pairs)

        if pairs_before > pairs_after:
            _logger.info(
                f"  Jaccard pre-filter: {pairs_before} -> {pairs_after} pairs "
                f"({pairs_before - pairs_after} removed, "
                f"{100 * (pairs_before - pairs_after) / pairs_before:.1f}%)"
            )

    num_aligned = len(candidate_pairs)

    if num_aligned == 0:
        _logger.info("  All candidates filtered by Jaccard. Returning empty results.")
        return SearchResults(
            hits=[[] for _ in range(nq)],
            num_queries=nq,
            num_targets=nd,
            num_candidates=num_candidates,
            num_aligned=0,
            runtime_seconds=time.perf_counter() - start_time,
        )

    # ── Stage 3: Banded NW alignment ──
    # Auto-compute band_width if not specified
    if band_width is None:
        all_lengths = np.concatenate(
            [query_dataset.lengths, db_dataset.lengths]
        )
        p95_len = int(np.percentile(all_lengths, 95))
        band_width = max(20, int(p95_len * 0.3))

    _logger.info(
        f"  Alignment: {num_aligned} pairs, band_width={band_width}, "
        f"threshold={threshold}"
    )

    with timer("Search Stage 3: Banded NW alignment"):
        # Merge query + db sequences into a single representation
        merged = _merge_sequences_for_alignment(query_dataset, db_dataset)
        merged_lengths = merged['lengths']
        nq_offset = merged['nq']

        # Remap pairs to merged-matrix indices
        merged_pairs = _remap_pairs_to_merged(candidate_pairs, nq_offset)

        # Sort pairs by min index for cache-friendly access
        sort_key = np.minimum(merged_pairs[:, 0], merged_pairs[:, 1])
        sort_order = np.argsort(sort_key, kind="mergesort")
        merged_pairs = merged_pairs[sort_order]
        # Keep track of original pair order to map back to candidate_pairs
        candidate_pairs = candidate_pairs[sort_order]

        # Run alignment — prefer compact format
        if merged['flat_sequences'] is not None:
            from clustkit.pairwise import _batch_align_compact
            sims, mask = _batch_align_compact(
                merged_pairs,
                merged['flat_sequences'],
                merged['offsets'],
                merged_lengths,
                np.float32(threshold),
                int32(band_width),
            )
        else:
            sims, mask = _batch_align(
                merged_pairs,
                merged['encoded_sequences'],
                merged_lengths,
                np.float32(threshold),
                int32(band_width),
            )

    # Filter to passing pairs
    passing_pairs = candidate_pairs[mask]  # (query_idx, db_idx) in original space
    passing_sims = sims[mask]

    _logger.info(
        f"  {len(passing_pairs)} pairs above threshold {threshold}"
    )

    # ── Collect top-k hits per query ──
    hits = _collect_top_k_hits(
        passing_pairs,
        passing_sims,
        nq,
        top_k,
        query_dataset,
        db_dataset,
    )

    total_hits = sum(len(h) for h in hits)
    queries_with_hits = sum(1 for h in hits if len(h) > 0)
    _logger.info(
        f"  {total_hits} total hits for {queries_with_hits}/{nq} queries"
    )

    elapsed = time.perf_counter() - start_time
    _logger.info(f"Search completed in {elapsed:.2f}s")

    return SearchResults(
        hits=hits,
        num_queries=nq,
        num_targets=nd,
        num_candidates=num_candidates,
        num_aligned=num_aligned,
        runtime_seconds=elapsed,
    )


def _collect_top_k_hits(
    passing_pairs: np.ndarray,
    passing_sims: np.ndarray,
    nq: int,
    top_k: int,
    query_dataset: SequenceDataset,
    db_dataset: SequenceDataset,
    passing_scores: np.ndarray | None = None,
) -> list[list[SearchHit]]:
    """Collect top-k hits per query, sorted by score (or identity) descending.

    When ``passing_scores`` is provided, hits are ranked by score descending
    (analogous to E-value ranking in BLAST/MMseqs2). Otherwise falls back to
    ranking by identity descending.

    Args:
        passing_pairs: (K, 2) int32 -- col 0 = query idx, col 1 = db idx.
        passing_sims: (K,) float32 -- identity values.
        nq: Number of queries.
        top_k: Max hits per query.
        query_dataset: For IDs and lengths.
        db_dataset: For IDs and lengths.
        passing_scores: (K,) float32 -- alignment scores for ranking (optional).

    Returns:
        List of length nq, each element a list of SearchHit sorted by
        score (or identity) descending.
    """
    hits: list[list[SearchHit]] = [[] for _ in range(nq)]

    if len(passing_pairs) == 0:
        return hits

    query_indices = passing_pairs[:, 0]
    db_indices = passing_pairs[:, 1]
    identities = passing_sims

    # Rank by score when available, otherwise by identity
    rank_values = passing_scores if passing_scores is not None else identities

    sort_order = np.lexsort((-rank_values, query_indices))

    sorted_qi = query_indices[sort_order]
    sorted_di = db_indices[sort_order]
    sorted_id = identities[sort_order]
    sorted_sc = rank_values[sort_order]

    # Walk through sorted results
    i = 0
    n = len(sort_order)
    while i < n:
        qi = int(sorted_qi[i])
        count = 0
        while i < n and int(sorted_qi[i]) == qi and count < top_k:
            di = int(sorted_di[i])
            identity = float(sorted_id[i])
            score = float(sorted_sc[i])
            hits[qi].append(
                SearchHit(
                    query_id=query_dataset.ids[qi],
                    target_id=db_dataset.ids[di],
                    identity=identity,
                    query_length=int(query_dataset.lengths[qi]),
                    target_length=int(db_dataset.lengths[di]),
                    score=score,
                )
            )
            count += 1
            i += 1
        # Skip remaining hits for this query beyond top_k
        while i < n and int(sorted_qi[i]) == qi:
            i += 1

    return hits


# ──────────────────────────────────────────────────────────────────────
# Search with pre-built database index
# ──────────────────────────────────────────────────────────────────────

def search_with_index(
    db_index,
    query_dataset: SequenceDataset,
    threshold: float = 0.5,
    top_k: int = 10,
    band_width: int | None = None,
    device: str = "cpu",
) -> SearchResults:
    """Search query sequences against a pre-built database index.

    Uses pre-computed sketches and LSH bucket IDs from the database index,
    avoiding the expensive database sketching and hashing steps.

    When a k-mer inverted index is available in the database, uses the
    faster k-mer index path instead of LSH-based candidate generation.

    Args:
        db_index: Pre-built DatabaseIndex from ``clustkit.database``.
        query_dataset: Query sequences as a SequenceDataset.
        threshold: Minimum sequence identity to report a hit (0.0-1.0).
        top_k: Maximum number of hits to return per query.
        band_width: Half-width of the alignment band. If None, auto-computed.
        device: Compute device for sketching queries.

    Returns:
        SearchResults with top-k hits per query.
    """
    # Dispatch to k-mer index path if available
    if db_index.kmer_offsets is not None:
        from clustkit.kmer_index import search_kmer_index
        return search_kmer_index(
            db_index, query_dataset,
            threshold=threshold, top_k=top_k,
            band_width=band_width, device=device,
        )

    start_time = time.perf_counter()

    nq = query_dataset.num_sequences
    nd = db_index.dataset.num_sequences
    params = db_index.params
    mode = params["mode"]
    k_lsh = params["kmer_size"]
    sketch_size = params["sketch_size"]

    _logger.info(
        f"Search: {nq} queries against {nd} database sequences (pre-indexed)"
    )

    if nq == 0 or nd == 0:
        return SearchResults(
            hits=[[] for _ in range(nq)],
            num_queries=nq, num_targets=nd,
            num_candidates=0, num_aligned=0,
            runtime_seconds=time.perf_counter() - start_time,
        )

    # ── Stage 0: Sketch queries (db sketches are pre-computed) ──
    with timer("Search Stage 0: Sketching queries"):
        query_sketches = compute_sketches(
            query_dataset.encoded_sequences,
            query_dataset.lengths,
            k_lsh, sketch_size, mode,
            device=device,
            flat_sequences=query_dataset.flat_sequences,
            offsets=query_dataset.offsets,
        )

    _logger.info(
        f"  LSH params (from index): {params['num_tables']} tables, "
        f"{params['num_bands']} bands/table"
    )

    # ── Stage 1: LSH candidate generation using pre-built db bucket IDs ──
    with timer("Search Stage 1: LSH candidate generation (pre-indexed)"):
        # Hash queries with same band indices and seeds as the database index
        query_bucket_ids = _hash_all_tables(
            query_sketches,
            db_index.lsh_band_indices,
            db_index.lsh_seeds,
            params["num_tables"],
        )

        # Preprocess db bucket IDs and extract pairs with incremental dedup.
        # With many tables and large db, total raw pairs can reach hundreds of
        # millions. Using a set for dedup avoids materialising the full list.
        num_tables = params["num_tables"]
        unique_set: set[int] = set()
        for t in range(num_tables):
            db_order, db_sorted, unique_bkts, starts, ends = (
                _preprocess_db_buckets(db_index.lsh_bucket_ids[t])
            )
            packed = _extract_cross_pairs_for_table_numba(
                query_bucket_ids[t],
                db_index.lsh_bucket_ids[t],
                db_order, db_sorted,
                starts, ends, unique_bkts,
                int64(nd), int64(1000),
            )
            if len(packed) > 0:
                unique_set.update(packed.tolist())

        if not unique_set:
            candidate_pairs = np.empty((0, 2), dtype=np.int32)
        else:
            unique_packed = np.fromiter(unique_set, dtype=np.int64, count=len(unique_set))
            candidate_pairs = np.empty((len(unique_packed), 2), dtype=np.int32)
            candidate_pairs[:, 0] = (unique_packed // nd).astype(np.int32)
            candidate_pairs[:, 1] = (unique_packed % nd).astype(np.int32)

    num_candidates = len(candidate_pairs)
    _logger.info(f"  Found {num_candidates} query->db candidate pairs")

    if num_candidates == 0:
        return SearchResults(
            hits=[[] for _ in range(nq)],
            num_queries=nq, num_targets=nd,
            num_candidates=0, num_aligned=0,
            runtime_seconds=time.perf_counter() - start_time,
        )

    # ── Stage 2: Jaccard pre-filter ──
    with timer("Search Stage 2: Jaccard pre-filter"):
        jaccard_mask = _jaccard_prefilter_cross(
            candidate_pairs, query_sketches, db_index.sketches, threshold
        )
        pairs_before = len(candidate_pairs)
        candidate_pairs = candidate_pairs[jaccard_mask]
        pairs_after = len(candidate_pairs)
        if pairs_before > pairs_after:
            _logger.info(
                f"  Jaccard pre-filter: {pairs_before} -> {pairs_after} pairs "
                f"({pairs_before - pairs_after} removed, "
                f"{100 * (pairs_before - pairs_after) / pairs_before:.1f}%)"
            )

    num_aligned = len(candidate_pairs)
    if num_aligned == 0:
        return SearchResults(
            hits=[[] for _ in range(nq)],
            num_queries=nq, num_targets=nd,
            num_candidates=num_candidates, num_aligned=0,
            runtime_seconds=time.perf_counter() - start_time,
        )

    # ── Stage 3: Banded NW alignment ──
    if band_width is None:
        all_lengths = np.concatenate(
            [query_dataset.lengths, db_index.dataset.lengths]
        )
        p95_len = int(np.percentile(all_lengths, 95))
        band_width = max(20, int(p95_len * 0.3))

    _logger.info(
        f"  Alignment: {num_aligned} pairs, band_width={band_width}, "
        f"threshold={threshold}"
    )

    with timer("Search Stage 3: Banded NW alignment"):
        merged = _merge_sequences_for_alignment(query_dataset, db_index.dataset)
        merged_lengths = merged["lengths"]
        nq_offset = merged["nq"]

        merged_pairs = _remap_pairs_to_merged(candidate_pairs, nq_offset)

        sort_key = np.minimum(merged_pairs[:, 0], merged_pairs[:, 1])
        sort_order = np.argsort(sort_key, kind="mergesort")
        merged_pairs = merged_pairs[sort_order]
        candidate_pairs = candidate_pairs[sort_order]

        if merged["flat_sequences"] is not None:
            from clustkit.pairwise import _batch_align_compact
            sims, aln_mask = _batch_align_compact(
                merged_pairs,
                merged["flat_sequences"],
                merged["offsets"],
                merged_lengths,
                np.float32(threshold),
                int32(band_width),
            )
        else:
            sims, aln_mask = _batch_align(
                merged_pairs,
                merged["encoded_sequences"],
                merged_lengths,
                np.float32(threshold),
                int32(band_width),
            )

    passing_pairs = candidate_pairs[aln_mask]
    passing_sims = sims[aln_mask]
    _logger.info(f"  {len(passing_pairs)} pairs above threshold {threshold}")

    hits = _collect_top_k_hits(
        passing_pairs, passing_sims, nq, top_k,
        query_dataset, db_index.dataset,
    )

    total_hits = sum(len(h) for h in hits)
    queries_with_hits = sum(1 for h in hits if len(h) > 0)
    _logger.info(f"  {total_hits} total hits for {queries_with_hits}/{nq} queries")

    elapsed = time.perf_counter() - start_time
    _logger.info(f"Search completed in {elapsed:.2f}s")

    return SearchResults(
        hits=hits,
        num_queries=nq,
        num_targets=nd,
        num_candidates=num_candidates,
        num_aligned=num_aligned,
        runtime_seconds=elapsed,
    )


# ──────────────────────────────────────────────────────────────────────
# TSV output
# ──────────────────────────────────────────────────────────────────────

def write_search_results_tsv(filepath, results: SearchResults):
    """Write search results to a tab-separated file.

    Output columns:
        query_id, target_id, identity, query_length, target_length

    Hits are written in order: queries by index, hits by identity descending.

    Args:
        filepath: Output file path (string or Path).
        results: SearchResults from ``search_sequences``.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        f.write("query_id\ttarget_id\tidentity\tquery_length\ttarget_length\n")
        for query_hits in results.hits:
            for hit in query_hits:
                f.write(
                    f"{hit.query_id}\t{hit.target_id}\t"
                    f"{hit.identity:.6f}\t{hit.query_length}\t"
                    f"{hit.target_length}\n"
                )

    _logger.info(
        f"Wrote search results to {filepath} "
        f"({sum(len(h) for h in results.hits)} hits)"
    )
