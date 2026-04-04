"""Candidate generation helpers for clustering."""

import numpy as np

from clustkit.kmer_index import (
    REDUCED_ALPHA,
    REDUCED_ALPHA_SIZE,
    _batch_score_queries,
    _remap_flat,
    _score_query_two_stage_spaced,
    build_kmer_index,
    build_kmer_index_spaced,
    compute_freq_threshold,
)
from clustkit.lsh import lsh_candidates
from clustkit.utils import auto_lsh_params, logger, timer


def _pack_upper_triangle(pairs: np.ndarray, num_sequences: int) -> np.ndarray:
    """Pack undirected pairs as min(i,j) * N + max(i,j), excluding self-pairs."""
    if len(pairs) == 0:
        return np.empty(0, dtype=np.int64)

    a = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    b = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    mask = a != b
    if not np.any(mask):
        return np.empty(0, dtype=np.int64)
    return a[mask] * int(num_sequences) + b[mask]


def _unpack_upper_triangle(packed: np.ndarray, num_sequences: int) -> np.ndarray:
    """Inverse of _pack_upper_triangle()."""
    if len(packed) == 0:
        return np.empty((0, 2), dtype=np.int32)
    pairs = np.empty((len(packed), 2), dtype=np.int32)
    pairs[:, 0] = (packed // int(num_sequences)).astype(np.int32)
    pairs[:, 1] = (packed % int(num_sequences)).astype(np.int32)
    return pairs


def _score_self_index_subset(
    dataset,
    query_ids: np.ndarray,
    kmer_offsets: np.ndarray,
    kmer_entries: np.ndarray,
    kmer_freqs: np.ndarray,
    *,
    k: int,
    alpha_size: int,
    max_cands_per_query: int,
    phase_a_topk: int,
    freq_percentile: float,
    min_total_hits: int,
    min_diag_hits: int,
    diag_bin_width: int,
) -> np.ndarray:
    """Score a subset of sequences against a self index and return packed pairs."""
    if len(query_ids) == 0:
        return np.empty(0, dtype=np.int64)

    q_offsets = dataset.offsets[query_ids].astype(np.int64)
    q_lengths = dataset.lengths[query_ids].astype(np.int32)
    out_targets = np.empty((len(query_ids), max_cands_per_query), dtype=np.int32)
    out_counts = np.zeros(len(query_ids), dtype=np.int32)

    freq_thresh = compute_freq_threshold(
        kmer_freqs, dataset.num_sequences, percentile=freq_percentile
    )

    _batch_score_queries(
        dataset.flat_sequences,
        q_offsets,
        q_lengths,
        np.int32(k),
        np.int32(alpha_size),
        kmer_offsets,
        kmer_entries,
        kmer_freqs,
        freq_thresh,
        np.int32(dataset.num_sequences),
        np.int32(min_total_hits),
        np.int32(min_diag_hits),
        np.int32(diag_bin_width),
        np.int32(max_cands_per_query),
        np.int32(phase_a_topk),
        out_targets,
        out_counts,
    )

    total = int(out_counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)

    pairs = np.empty((total, 2), dtype=np.int32)
    pos = 0
    for qi, seq_id in enumerate(query_ids):
        nc = int(out_counts[qi])
        if nc == 0:
            continue
        pairs[pos:pos + nc, 0] = np.int32(seq_id)
        pairs[pos:pos + nc, 1] = out_targets[qi, :nc]
        pos += nc

    return _pack_upper_triangle(pairs[:pos], dataset.num_sequences)


def _score_self_spaced_subset(
    dataset,
    query_ids: np.ndarray,
    red_flat: np.ndarray,
    seed_pattern: str,
    *,
    max_cands_per_query: int,
    phase_a_topk: int,
    freq_percentile: float,
    min_total_hits: int,
    min_diag_hits: int,
    diag_bin_width: int,
    chunk_size: int = 8,
    max_query_length: int = 2048,
) -> np.ndarray:
    """Score a subset of sequences against a spaced-seed self index."""
    if len(query_ids) == 0:
        return np.empty(0, dtype=np.int64)

    (sp_offsets, sp_entries, sp_freqs,
     seed_offsets, weight, span) = build_kmer_index_spaced(
        red_flat,
        dataset.offsets,
        dataset.lengths,
        seed_pattern,
        "protein",
        alpha_size=REDUCED_ALPHA_SIZE,
    )

    freq_thresh = compute_freq_threshold(
        sp_freqs, dataset.num_sequences, percentile=freq_percentile
    )
    packed_chunks = []
    for start in range(0, len(query_ids), chunk_size):
        qids = query_ids[start:start + chunk_size]
        chunk_pairs = []
        for seq_id in qids:
            q_start = int(dataset.offsets[seq_id])
            q_len = int(dataset.lengths[seq_id])
            if q_len > max_query_length:
                continue
            q_seq = red_flat[q_start:q_start + q_len]

            cand_ids, cand_scores = _score_query_two_stage_spaced(
                q_seq,
                np.int32(q_len),
                seed_offsets,
                np.int32(weight),
                np.int32(span),
                np.int32(REDUCED_ALPHA_SIZE),
                sp_offsets,
                sp_entries,
                sp_freqs,
                freq_thresh,
                np.int32(dataset.num_sequences),
                np.int32(min_total_hits),
                np.int32(min_diag_hits),
                np.int32(diag_bin_width),
                np.int32(phase_a_topk),
            )

            nc = len(cand_ids)
            if nc == 0:
                continue
            if nc > max_cands_per_query:
                order = np.argsort(-cand_scores)[:max_cands_per_query]
                cand_ids = cand_ids[order]
                nc = len(cand_ids)

            pairs = np.empty((nc, 2), dtype=np.int32)
            pairs[:, 0] = np.int32(seq_id)
            pairs[:, 1] = cand_ids[:nc]
            chunk_pairs.append(pairs)

        if chunk_pairs:
            packed = _pack_upper_triangle(
                np.concatenate(chunk_pairs, axis=0),
                dataset.num_sequences,
            )
            if len(packed) > 0:
                packed_chunks.append(packed)

    if len(packed_chunks) == 0:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(packed_chunks)


def clustering_candidates(
    dataset,
    sketches: np.ndarray,
    *,
    threshold: float,
    sensitivity: str,
    mode: str,
    k_lsh: int,
    device: str = "cpu",
    strategy: str = "lsh",
    augment_min_candidates: int = 64,
    augment_max_sequences: int = 4096,
    augment_max_cands_per_query: int = 256,
    augment_phase_a_topk: int = 4000,
    augment_freq_percentile: float = 99.0,
    augment_min_total_hits: int = 2,
    augment_min_diag_hits: int = 2,
    augment_diag_bin_width: int = 10,
    reduced_k: int = 5,
    spaced_seed: str = "110011",
    spaced_max_query_length: int = 2048,
) -> np.ndarray:
    """Generate clustering candidates using LSH or an LSH+k-mer hybrid."""
    lsh_params = auto_lsh_params(threshold, sensitivity, k=k_lsh)
    logger.info(
        f"  LSH params: {lsh_params['num_tables']} tables, "
        f"{lsh_params['num_bands']} bands/table"
    )

    with timer("Phase 2: LSH candidate generation"):
        base_pairs = lsh_candidates(
            sketches,
            num_tables=lsh_params["num_tables"],
            num_bands=lsh_params["num_bands"],
            device=device,
        )

    if strategy != "hybrid" or mode != "protein":
        return base_pairs

    if len(base_pairs) == 0:
        candidate_counts = np.zeros(dataset.num_sequences, dtype=np.int32)
    else:
        candidate_counts = np.bincount(
            np.concatenate([base_pairs[:, 0], base_pairs[:, 1]]),
            minlength=dataset.num_sequences,
        ).astype(np.int32)

    augment_ids = np.where(candidate_counts < augment_min_candidates)[0]
    if len(augment_ids) == 0:
        logger.info("  Hybrid candidate augmentation skipped: no low-support sequences")
        return base_pairs

    if len(augment_ids) > augment_max_sequences:
        order = np.argsort(candidate_counts[augment_ids], kind="mergesort")
        augment_ids = augment_ids[order[:augment_max_sequences]]

    logger.info(
        f"  Hybrid augmentation: {len(augment_ids)} low-support sequences "
        f"(< {augment_min_candidates} base candidates)"
    )

    packed_arrays = [_pack_upper_triangle(base_pairs, dataset.num_sequences)]
    red_flat = _remap_flat(dataset.flat_sequences, REDUCED_ALPHA, len(dataset.flat_sequences))

    with timer("Phase 2.1: Reduced-alphabet candidate augmentation"):
        red_offsets, red_entries, red_freqs = build_kmer_index(
            red_flat,
            dataset.offsets,
            dataset.lengths,
            reduced_k,
            mode,
            alpha_size=REDUCED_ALPHA_SIZE,
        )
        packed_arrays.append(
            _score_self_index_subset(
                dataset,
                augment_ids,
                red_offsets,
                red_entries,
                red_freqs,
                k=reduced_k,
                alpha_size=REDUCED_ALPHA_SIZE,
                max_cands_per_query=augment_max_cands_per_query,
                phase_a_topk=augment_phase_a_topk,
                freq_percentile=augment_freq_percentile,
                min_total_hits=augment_min_total_hits,
                min_diag_hits=augment_min_diag_hits,
                diag_bin_width=augment_diag_bin_width,
            )
        )

    if spaced_seed:
        with timer("Phase 2.2: Spaced-seed candidate augmentation"):
            packed_arrays.append(
                _score_self_spaced_subset(
                    dataset,
                    augment_ids,
                    red_flat,
                    spaced_seed,
                    max_cands_per_query=augment_max_cands_per_query,
                    phase_a_topk=augment_phase_a_topk,
                    freq_percentile=augment_freq_percentile,
                    min_total_hits=augment_min_total_hits,
                    min_diag_hits=augment_min_diag_hits,
                    diag_bin_width=augment_diag_bin_width,
                    max_query_length=spaced_max_query_length,
                )
            )

    union_packed = np.unique(np.concatenate([arr for arr in packed_arrays if len(arr) > 0]))
    augmented_pairs = _unpack_upper_triangle(union_packed, dataset.num_sequences)
    logger.info(
        f"  Hybrid candidate union: {len(base_pairs)} -> {len(augmented_pairs)} pairs"
    )
    return augmented_pairs
