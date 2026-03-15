"""K-mer inverted index for fast sequence search.

Pre-computes a CSR-format inverted index mapping each k-mer to the database
positions where it occurs.  At search time, queries are scored against the
index in two phases:

  Phase A  –  Per-target k-mer count (fast, coarse filter).
  Phase B  –  Diagonal scoring for Phase-A survivors (selective, fine filter).

Only the surviving (query, target) pairs are passed to banded NW alignment.
"""

import numpy as np
from numba import njit, prange, int32, int64, int16

from clustkit.utils import logger, timer


# ──────────────────────────────────────────────────────────────────────────
# Reduced alphabet for sensitive distant homology detection
# ──────────────────────────────────────────────────────────────────────────

# Murphy-10-like reduced alphabet (9 groups)
# ClustKIT encoding: A=0 C=1 D=2 E=3 F=4 G=5 H=6 I=7 K=8 L=9
#                    M=10 N=11 P=12 Q=13 R=14 S=15 T=16 V=17 W=18 Y=19
# Groups: {A,G}=0 {C}=1 {D,E,N,Q}=2 {F,W,Y}=3 {H}=4
#         {I,L,M,V}=5 {K,R}=6 {P}=7 {S,T}=8
REDUCED_ALPHA = np.array(
    [0, 1, 2, 2, 3, 0, 4, 5, 6, 5, 5, 2, 7, 2, 6, 8, 8, 5, 3, 3],
    dtype=np.uint8,
)
REDUCED_ALPHA_SIZE = 9

# Dayhoff-6: classic evolutionary grouping (coarser, broader homology)
# Groups: {A,G,P,S,T}=0 {D,E,N,Q}=1 {H,K,R}=2 {F,W,Y}=3 {I,L,M,V}=4 {C}=5
DAYHOFF6_ALPHA = np.array(
    [0, 5, 1, 1, 3, 0, 2, 4, 2, 4, 4, 1, 0, 1, 2, 0, 0, 4, 3, 3],
    dtype=np.uint8,
)
DAYHOFF6_ALPHA_SIZE = 6

# Hydrophobicity-8: groups by physical property conservation
# Groups: {A,G,I,L,M,V}=0 {F,W,Y}=1 {D,E}=2 {H,K,R}=3 {N,Q}=4 {S,T}=5 {C}=6 {P}=7
HYDRO8_ALPHA = np.array(
    [0, 6, 2, 2, 1, 0, 3, 0, 3, 0, 0, 4, 7, 4, 3, 5, 5, 0, 1, 1],
    dtype=np.uint8,
)
HYDRO8_ALPHA_SIZE = 8


@njit(cache=True)
def _remap_flat(src, mapping, n):
    """Remap a flat uint8 array through a lookup table."""
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        v = src[i]
        if v < 20:
            out[i] = mapping[v]
        else:
            out[i] = 20  # sentinel for unknown
    return out


# ──────────────────────────────────────────────────────────────────────────
# Index building
# ──────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _count_kmers_nb(flat_seqs, offsets, lengths, k, alpha_size, counts):
    """Pass 1: count k-mer occurrences across all database sequences."""
    n = len(lengths)
    for i in range(n):
        start = int64(offsets[i])
        length = int32(lengths[i])
        if length < k:
            continue
        num_kmers = length - k + 1
        for pos in range(num_kmers):
            kmer_val = int64(0)
            valid = True
            for j in range(k):
                r = flat_seqs[start + pos + j]
                if r >= alpha_size:
                    valid = False
                    break
                kmer_val = kmer_val * int64(alpha_size) + int64(r)
            if valid:
                counts[kmer_val] += int64(1)


@njit(cache=True)
def _fill_entries_nb(flat_seqs, offsets, lengths, k, alpha_size,
                     kmer_offsets, entries, cursors):
    """Pass 2: fill CSR entries array with packed (seq_id << 32 | position)."""
    n = len(lengths)
    for i in range(n):
        start = int64(offsets[i])
        length = int32(lengths[i])
        if length < k:
            continue
        num_kmers = length - k + 1
        for pos in range(num_kmers):
            kmer_val = int64(0)
            valid = True
            for j in range(k):
                r = flat_seqs[start + pos + j]
                if r >= alpha_size:
                    valid = False
                    break
                kmer_val = kmer_val * int64(alpha_size) + int64(r)
            if valid:
                idx = cursors[kmer_val]
                entries[idx] = (int64(i) << 32) | int64(pos)
                cursors[kmer_val] = idx + int64(1)


def build_kmer_index(flat_sequences, offsets, lengths, k, mode,
                     alpha_size=None):
    """Build a CSR-format k-mer inverted index.

    Args:
        flat_sequences: Concatenated uint8 encoded sequences.
        offsets: int64 start offset per sequence in *flat_sequences*.
        lengths: int32 length per sequence.
        k: K-mer size.
        mode: ``"protein"`` or ``"nucleotide"``.
        alpha_size: Alphabet size override (default: 20 for protein, 4 for nt).

    Returns:
        ``(kmer_offsets, kmer_entries, kmer_freqs)``

        * ``kmer_offsets`` – ``int64[num_possible_kmers + 1]`` CSR row pointers.
        * ``kmer_entries`` – ``int64[total_occurrences]`` packed entries.
        * ``kmer_freqs``   – ``int32[num_possible_kmers]`` per-k-mer counts.
    """
    if alpha_size is None:
        alpha_size = 20 if mode == "protein" else 4
    num_possible = alpha_size ** k

    # Pass 1 – count
    counts = np.zeros(num_possible, dtype=np.int64)
    _count_kmers_nb(flat_sequences, offsets, lengths, int32(k),
                    int32(alpha_size), counts)

    kmer_freqs = counts.astype(np.int32)

    # Prefix sum → CSR offsets
    kmer_offsets = np.zeros(num_possible + 1, dtype=np.int64)
    np.cumsum(counts, out=kmer_offsets[1:])
    total = int(kmer_offsets[-1])

    # Pass 2 – fill entries
    kmer_entries = np.empty(total, dtype=np.int64)
    cursors = kmer_offsets[:-1].copy()
    _fill_entries_nb(flat_sequences, offsets, lengths, int32(k),
                     int32(alpha_size), kmer_offsets, kmer_entries, cursors)

    logger.info(
        f"K-mer index built: {num_possible} k-mers, "
        f"{total:,} total entries, k={k}"
    )
    return kmer_offsets, kmer_entries, kmer_freqs


# ──────────────────────────────────────────────────────────────────────────
# Spaced seed index building
# ──────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _count_kmers_spaced(flat_seqs, offsets, lengths, seed_offsets, weight,
                        span, alpha_size, counts):
    """Count k-mers using a spaced seed pattern."""
    n = len(lengths)
    for i in range(n):
        start = int64(offsets[i])
        length = int32(lengths[i])
        if length < span:
            continue
        num_seeds = length - span + 1
        for pos in range(num_seeds):
            kmer_val = int64(0)
            valid = True
            for j in range(weight):
                r = flat_seqs[start + pos + seed_offsets[j]]
                if r >= alpha_size:
                    valid = False
                    break
                kmer_val = kmer_val * int64(alpha_size) + int64(r)
            if valid:
                counts[kmer_val] += int64(1)


@njit(cache=True)
def _fill_entries_spaced(flat_seqs, offsets, lengths, seed_offsets, weight,
                         span, alpha_size, kmer_offsets, entries, cursors):
    """Fill CSR entries for a spaced seed index."""
    n = len(lengths)
    for i in range(n):
        start = int64(offsets[i])
        length = int32(lengths[i])
        if length < span:
            continue
        num_seeds = length - span + 1
        for pos in range(num_seeds):
            kmer_val = int64(0)
            valid = True
            for j in range(weight):
                r = flat_seqs[start + pos + seed_offsets[j]]
                if r >= alpha_size:
                    valid = False
                    break
                kmer_val = kmer_val * int64(alpha_size) + int64(r)
            if valid:
                idx = cursors[kmer_val]
                entries[idx] = (int64(i) << 32) | int64(pos)
                cursors[kmer_val] = idx + int64(1)


def build_kmer_index_spaced(flat_sequences, offsets, lengths, seed_pattern,
                            mode, alpha_size=None):
    """Build a CSR-format k-mer index using a spaced seed pattern.

    Args:
        seed_pattern: string like ``"11011"`` where 1=match, 0=don't-care.
        Other args same as :func:`build_kmer_index`.

    Returns:
        ``(kmer_offsets, kmer_entries, kmer_freqs, seed_offsets, weight, span)``
    """
    seed_offsets = np.array(
        [i for i, c in enumerate(seed_pattern) if c == '1'], dtype=np.int32
    )
    weight = len(seed_offsets)
    span = len(seed_pattern)

    if alpha_size is None:
        alpha_size = 20 if mode == "protein" else 4
    num_possible = alpha_size ** weight

    counts = np.zeros(num_possible, dtype=np.int64)
    _count_kmers_spaced(flat_sequences, offsets, lengths, seed_offsets,
                        int32(weight), int32(span), int32(alpha_size), counts)

    kmer_freqs = counts.astype(np.int32)
    kmer_offsets = np.zeros(num_possible + 1, dtype=np.int64)
    np.cumsum(counts, out=kmer_offsets[1:])
    total = int(kmer_offsets[-1])

    kmer_entries = np.empty(total, dtype=np.int64)
    cursors = kmer_offsets[:-1].copy()
    _fill_entries_spaced(flat_sequences, offsets, lengths, seed_offsets,
                         int32(weight), int32(span), int32(alpha_size),
                         kmer_offsets, kmer_entries, cursors)

    logger.info(
        f"Spaced seed index: pattern={seed_pattern}, "
        f"{num_possible} k-mers, {total:,} entries"
    )
    return kmer_offsets, kmer_entries, kmer_freqs, seed_offsets, weight, span


# ──────────────────────────────────────────────────────────────────────────
# Spaced seed scoring (Phase A + B)
# ──────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _score_query_two_stage_spaced(
    q_seq, q_len, seed_offsets, weight, span, alpha_size,
    kmer_offsets, kmer_entries, kmer_freqs, freq_thresh,
    num_db, min_total_hits, min_diag_hits, diag_bin_width,
    phase_a_topk,
):
    """Two-stage scoring using a spaced seed pattern."""
    num_seeds = q_len - span + 1
    if num_seeds <= 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    # ── Phase A ──────────────────────────────────────────────────────
    target_counts = np.zeros(num_db, dtype=np.int16)

    for qpos in range(num_seeds):
        kmer_val = int64(0)
        valid = True
        for j in range(weight):
            r = q_seq[qpos + seed_offsets[j]]
            if r >= alpha_size:
                valid = False
                break
            kmer_val = kmer_val * int64(alpha_size) + int64(r)
        if not valid:
            continue
        if kmer_freqs[kmer_val] > freq_thresh:
            continue
        s = kmer_offsets[kmer_val]
        e = kmer_offsets[kmer_val + 1]
        for h in range(s, e):
            tid = int32(kmer_entries[h] >> 32)
            if target_counts[tid] < 32767:
                target_counts[tid] += int16(1)

    # ── Top-K selection ──────────────────────────────────────────────
    num_passing = int32(0)
    for i in range(num_db):
        if target_counts[i] >= min_total_hits:
            num_passing += int32(1)
    if num_passing == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    passing_ids = np.empty(num_passing, dtype=np.int32)
    passing_scores = np.empty(num_passing, dtype=np.int16)
    pos = int32(0)
    for i in range(num_db):
        if target_counts[i] >= min_total_hits:
            passing_ids[pos] = int32(i)
            passing_scores[pos] = target_counts[i]
            pos += int32(1)

    if min_diag_hits <= 1:
        return passing_ids, passing_scores.astype(np.int32)

    topk = min(int32(phase_a_topk), num_passing)
    if topk < num_passing:
        order = np.argsort(-passing_scores)
        topk_ids = np.empty(topk, dtype=np.int32)
        for i in range(topk):
            topk_ids[i] = passing_ids[order[i]]
    else:
        topk_ids = passing_ids

    # ── Phase B: diagonal scoring ────────────────────────────────────
    survivor_mask = np.zeros(num_db, dtype=np.bool_)
    for i in range(len(topk_ids)):
        survivor_mask[topk_ids[i]] = True

    n_surv_hits = int64(0)
    for i in range(len(topk_ids)):
        n_surv_hits += int64(target_counts[topk_ids[i]])

    DIAG_MULT = int64(1000000)
    max_diag_shift = int32(q_len)
    surv_keys = np.empty(n_surv_hits, dtype=np.int64)
    sw = int64(0)

    for qpos in range(num_seeds):
        kmer_val = int64(0)
        valid = True
        for j in range(weight):
            r = q_seq[qpos + seed_offsets[j]]
            if r >= alpha_size:
                valid = False
                break
            kmer_val = kmer_val * int64(alpha_size) + int64(r)
        if not valid:
            continue
        if kmer_freqs[kmer_val] > freq_thresh:
            continue
        s = kmer_offsets[kmer_val]
        e = kmer_offsets[kmer_val + 1]
        for h in range(s, e):
            entry = kmer_entries[h]
            tid = int32(entry >> 32)
            if survivor_mask[tid]:
                tpos = int32(entry & 0xFFFFFFFF)
                diag = int32(tpos) - int32(qpos) + max_diag_shift
                dbin = int32(diag // diag_bin_width)
                surv_keys[sw] = int64(tid) * DIAG_MULT + int64(dbin)
                sw += int64(1)

    surv_keys = surv_keys[:sw]
    surv_keys.sort()

    final_ids = np.empty(len(topk_ids), dtype=np.int32)
    final_scores = np.empty(len(topk_ids), dtype=np.int32)
    num_final = int32(0)
    prev_tid = int32(-1)
    best_count = int32(0)
    i = int64(0)
    n = int64(len(surv_keys))

    while i < n:
        key = surv_keys[i]
        tid = int32(key // DIAG_MULT)
        dbin = int32(key % DIAG_MULT)
        run = int32(0)
        while i < n:
            k2 = surv_keys[i]
            if int32(k2 // DIAG_MULT) != tid or int32(k2 % DIAG_MULT) != dbin:
                break
            run += int32(1)
            i += int64(1)
        if tid != prev_tid:
            if prev_tid >= 0 and best_count >= min_diag_hits:
                final_ids[num_final] = prev_tid
                final_scores[num_final] = best_count
                num_final += int32(1)
            prev_tid = tid
            best_count = run
        else:
            if run > best_count:
                best_count = run

    if prev_tid >= 0 and best_count >= min_diag_hits:
        final_ids[num_final] = prev_tid
        final_scores[num_final] = best_count
        num_final += int32(1)

    return final_ids[:num_final], final_scores[:num_final]


@njit(parallel=True, cache=True)
def _batch_score_queries_spaced(
    q_flat, q_offsets, q_lengths,
    seed_offsets, weight, span, alpha_size,
    kmer_offsets, kmer_entries, kmer_freqs,
    freq_thresh, num_db, min_total_hits,
    min_diag_hits, diag_bin_width,
    max_cands, phase_a_topk,
    out_targets, out_counts,
):
    """Score all queries using a spaced seed pattern (parallel)."""
    nq = len(q_lengths)
    for qi in prange(nq):
        qs = int64(q_offsets[qi])
        ql = int32(q_lengths[qi])
        q_seq = q_flat[qs:qs + ql]

        cand_ids, cand_scores = _score_query_two_stage_spaced(
            q_seq, ql,
            seed_offsets, weight, span, alpha_size,
            kmer_offsets, kmer_entries, kmer_freqs, freq_thresh,
            num_db, min_total_hits, min_diag_hits, diag_bin_width,
            phase_a_topk,
        )

        nc = len(cand_ids)
        if nc > max_cands:
            order = np.argsort(-cand_scores)
            for j in range(max_cands):
                out_targets[qi, j] = cand_ids[order[j]]
            out_counts[qi] = max_cands
        else:
            for j in range(nc):
                out_targets[qi, j] = cand_ids[j]
            out_counts[qi] = int32(nc)


# ──────────────────────────────────────────────────────────────────────────
# Frequency threshold
# ──────────────────────────────────────────────────────────────────────────

def compute_freq_threshold(kmer_freqs, num_sequences, percentile=99.5):
    """Compute a frequency cap for overly common k-mers.

    K-mers above this threshold are skipped during query scoring because
    they generate too many low-quality candidates (analogous to stop-words
    in text search).
    """
    nonzero = kmer_freqs[kmer_freqs > 0]
    if len(nonzero) == 0:
        return int32(num_sequences)
    pctl = int(np.percentile(nonzero, percentile))
    floor = max(100, num_sequences // 200)
    return int32(max(pctl, floor))


# ──────────────────────────────────────────────────────────────────────────
# Per-query scoring (Numba)
# ──────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _score_query_phase_a(
    q_seq,            # uint8[q_len] – query sequence (slice of flat array)
    q_len,            # int32
    k,                # int32
    alpha_size,       # int32
    kmer_offsets,     # int64[num_possible + 1]
    kmer_entries,     # int64[total]
    kmer_freqs,       # int32[num_possible]
    freq_thresh,      # int32
    num_db,           # int32 – number of database sequences
    min_total_hits,   # int32 – Phase A threshold
):
    """Phase A only: count k-mer hits per target, return survivors.

    Fast path used when ``min_diag_hits <= 1``.  Single pass through entries,
    no diagonal buffer, no sorting.

    Returns ``(target_ids, hit_counts)`` – int32 arrays.
    """
    num_kmers = q_len - k + 1
    if num_kmers <= 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    target_counts = np.zeros(num_db, dtype=np.int16)

    for qpos in range(num_kmers):
        kmer_val = int64(0)
        valid = True
        for j in range(k):
            r = q_seq[qpos + j]
            if r >= alpha_size:
                valid = False
                break
            kmer_val = kmer_val * int64(alpha_size) + int64(r)
        if not valid:
            continue
        if kmer_freqs[kmer_val] > freq_thresh:
            continue

        s = kmer_offsets[kmer_val]
        e = kmer_offsets[kmer_val + 1]
        for h in range(s, e):
            tid = int32(kmer_entries[h] >> 32)
            if target_counts[tid] < 32767:
                target_counts[tid] += int16(1)

    # Collect survivors
    num_survivors = int32(0)
    for i in range(num_db):
        if target_counts[i] >= min_total_hits:
            num_survivors += int32(1)

    if num_survivors == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    result_ids = np.empty(num_survivors, dtype=np.int32)
    result_scores = np.empty(num_survivors, dtype=np.int32)
    pos = int32(0)
    for i in range(num_db):
        if target_counts[i] >= min_total_hits:
            result_ids[pos] = int32(i)
            result_scores[pos] = int32(target_counts[i])
            pos += int32(1)
    return result_ids, result_scores


@njit(cache=True)
def _score_query_with_diag(
    q_seq, q_len, k, alpha_size,
    kmer_offsets, kmer_entries, kmer_freqs, freq_thresh,
    num_db, min_total_hits, min_diag_hits, diag_bin_width,
):
    """Phase A + B: k-mer counting then diagonal scoring for survivors.

    Two passes through entries:
      Pass 1 – increment per-target counts (Phase A).
      Pass 2 – collect (tid, diag_bin) for survivors, sort, count runs (Phase B).

    Returns ``(target_ids, best_diag_counts)`` – int32 arrays.
    """
    num_kmers = q_len - k + 1
    if num_kmers <= 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    # ── Phase A ──────────────────────────────────────────────────────
    target_counts = np.zeros(num_db, dtype=np.int16)

    for qpos in range(num_kmers):
        kmer_val = int64(0)
        valid = True
        for j in range(k):
            r = q_seq[qpos + j]
            if r >= alpha_size:
                valid = False
                break
            kmer_val = kmer_val * int64(alpha_size) + int64(r)
        if not valid:
            continue
        if kmer_freqs[kmer_val] > freq_thresh:
            continue
        s = kmer_offsets[kmer_val]
        e = kmer_offsets[kmer_val + 1]
        for h in range(s, e):
            tid = int32(kmer_entries[h] >> 32)
            if target_counts[tid] < 32767:
                target_counts[tid] += int16(1)

    # Build survivor mask and count survivor hits
    survivor_mask = np.zeros(num_db, dtype=np.bool_)
    num_survivors = int32(0)
    n_surv_hits = int64(0)
    for i in range(num_db):
        if target_counts[i] >= min_total_hits:
            survivor_mask[i] = True
            num_survivors += int32(1)
            n_surv_hits += int64(target_counts[i])

    if num_survivors == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    # ── Phase B: second pass for diagonal keys ───────────────────────
    DIAG_MULT = int64(1000000)
    max_diag_shift = int32(q_len)

    surv_keys = np.empty(n_surv_hits, dtype=np.int64)
    sw = int64(0)

    for qpos in range(num_kmers):
        kmer_val = int64(0)
        valid = True
        for j in range(k):
            r = q_seq[qpos + j]
            if r >= alpha_size:
                valid = False
                break
            kmer_val = kmer_val * int64(alpha_size) + int64(r)
        if not valid:
            continue
        if kmer_freqs[kmer_val] > freq_thresh:
            continue
        s = kmer_offsets[kmer_val]
        e = kmer_offsets[kmer_val + 1]
        for h in range(s, e):
            entry = kmer_entries[h]
            tid = int32(entry >> 32)
            if survivor_mask[tid]:
                tpos = int32(entry & 0xFFFFFFFF)
                diag = int32(tpos) - int32(qpos) + max_diag_shift
                dbin = int32(diag // diag_bin_width)
                surv_keys[sw] = int64(tid) * DIAG_MULT + int64(dbin)
                sw += int64(1)

    surv_keys = surv_keys[:sw]
    surv_keys.sort()

    # Count runs
    final_ids = np.empty(num_survivors, dtype=np.int32)
    final_scores = np.empty(num_survivors, dtype=np.int32)
    num_final = int32(0)
    prev_tid = int32(-1)
    best_count = int32(0)
    i = int64(0)
    n = int64(len(surv_keys))

    while i < n:
        key = surv_keys[i]
        tid = int32(key // DIAG_MULT)
        dbin = int32(key % DIAG_MULT)
        run = int32(0)
        while i < n:
            k2 = surv_keys[i]
            if int32(k2 // DIAG_MULT) != tid or int32(k2 % DIAG_MULT) != dbin:
                break
            run += int32(1)
            i += int64(1)
        if tid != prev_tid:
            if prev_tid >= 0 and best_count >= min_diag_hits:
                final_ids[num_final] = prev_tid
                final_scores[num_final] = best_count
                num_final += int32(1)
            prev_tid = tid
            best_count = run
        else:
            if run > best_count:
                best_count = run

    if prev_tid >= 0 and best_count >= min_diag_hits:
        final_ids[num_final] = prev_tid
        final_scores[num_final] = best_count
        num_final += int32(1)

    return final_ids[:num_final], final_scores[:num_final]


@njit(cache=True)
def _score_query_two_stage(
    q_seq, q_len, k, alpha_size,
    kmer_offsets, kmer_entries, kmer_freqs, freq_thresh,
    num_db, min_total_hits, min_diag_hits, diag_bin_width,
    phase_a_topk,
):
    """Two-stage scoring: Phase A → top-K selection → selective Phase B.

    Much faster than full Phase A+B because Phase B only processes the
    top-K targets from Phase A (sparse survivor mask), not all targets
    that pass min_total_hits.

    Args:
        phase_a_topk: Number of Phase A top candidates to keep for Phase B.
            Typically 3000-10000.  If 0 or >= num_db, falls back to
            ``_score_query_with_diag`` (full Phase B on all survivors).

    Returns:
        ``(target_ids, best_diag_counts)`` – int32 arrays.
    """
    num_kmers = q_len - k + 1
    if num_kmers <= 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    # ── Phase A: k-mer count per target ─────────────────────────────
    target_counts = np.zeros(num_db, dtype=np.int16)

    for qpos in range(num_kmers):
        kmer_val = int64(0)
        valid = True
        for j in range(k):
            r = q_seq[qpos + j]
            if r >= alpha_size:
                valid = False
                break
            kmer_val = kmer_val * int64(alpha_size) + int64(r)
        if not valid:
            continue
        if kmer_freqs[kmer_val] > freq_thresh:
            continue
        s = kmer_offsets[kmer_val]
        e = kmer_offsets[kmer_val + 1]
        for h in range(s, e):
            tid = int32(kmer_entries[h] >> 32)
            if target_counts[tid] < 32767:
                target_counts[tid] += int16(1)

    # ── Select top-K from Phase A ───────────────────────────────────
    # Collect all targets passing min_total_hits
    num_passing = int32(0)
    for i in range(num_db):
        if target_counts[i] >= min_total_hits:
            num_passing += int32(1)

    if num_passing == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    passing_ids = np.empty(num_passing, dtype=np.int32)
    passing_scores = np.empty(num_passing, dtype=np.int16)
    pos = int32(0)
    for i in range(num_db):
        if target_counts[i] >= min_total_hits:
            passing_ids[pos] = int32(i)
            passing_scores[pos] = target_counts[i]
            pos += int32(1)

    # If Phase B not needed or few enough survivors, shortcut
    if min_diag_hits <= 1:
        return passing_ids, passing_scores.astype(np.int32)

    # Select top-K by Phase A count for Phase B
    topk = min(int32(phase_a_topk), num_passing)
    if topk < num_passing:
        # Partial argsort: take top-K indices
        order = np.argsort(-passing_scores)
        topk_ids = np.empty(topk, dtype=np.int32)
        for i in range(topk):
            topk_ids[i] = passing_ids[order[i]]
    else:
        topk_ids = passing_ids

    # Build sparse survivor mask for top-K only
    survivor_mask = np.zeros(num_db, dtype=np.bool_)
    for i in range(len(topk_ids)):
        survivor_mask[topk_ids[i]] = True

    # Count expected Phase B entries (for pre-allocation)
    n_surv_hits = int64(0)
    for i in range(len(topk_ids)):
        n_surv_hits += int64(target_counts[topk_ids[i]])

    # ── Phase B: diagonal scoring for top-K targets ─────────────────
    DIAG_MULT = int64(1000000)
    max_diag_shift = int32(q_len)

    surv_keys = np.empty(n_surv_hits, dtype=np.int64)
    sw = int64(0)

    for qpos in range(num_kmers):
        kmer_val = int64(0)
        valid = True
        for j in range(k):
            r = q_seq[qpos + j]
            if r >= alpha_size:
                valid = False
                break
            kmer_val = kmer_val * int64(alpha_size) + int64(r)
        if not valid:
            continue
        if kmer_freqs[kmer_val] > freq_thresh:
            continue
        s = kmer_offsets[kmer_val]
        e = kmer_offsets[kmer_val + 1]
        for h in range(s, e):
            entry = kmer_entries[h]
            tid = int32(entry >> 32)
            if survivor_mask[tid]:
                tpos = int32(entry & 0xFFFFFFFF)
                diag = int32(tpos) - int32(qpos) + max_diag_shift
                dbin = int32(diag // diag_bin_width)
                surv_keys[sw] = int64(tid) * DIAG_MULT + int64(dbin)
                sw += int64(1)

    surv_keys = surv_keys[:sw]
    surv_keys.sort()

    # Count runs
    final_ids = np.empty(len(topk_ids), dtype=np.int32)
    final_scores = np.empty(len(topk_ids), dtype=np.int32)
    num_final = int32(0)
    prev_tid = int32(-1)
    best_count = int32(0)
    i = int64(0)
    n = int64(len(surv_keys))

    while i < n:
        key = surv_keys[i]
        tid = int32(key // DIAG_MULT)
        dbin = int32(key % DIAG_MULT)
        run = int32(0)
        while i < n:
            k2 = surv_keys[i]
            if int32(k2 // DIAG_MULT) != tid or int32(k2 % DIAG_MULT) != dbin:
                break
            run += int32(1)
            i += int64(1)
        if tid != prev_tid:
            if prev_tid >= 0 and best_count >= min_diag_hits:
                final_ids[num_final] = prev_tid
                final_scores[num_final] = best_count
                num_final += int32(1)
            prev_tid = tid
            best_count = run
        else:
            if run > best_count:
                best_count = run

    if prev_tid >= 0 and best_count >= min_diag_hits:
        final_ids[num_final] = prev_tid
        final_scores[num_final] = best_count
        num_final += int32(1)

    return final_ids[:num_final], final_scores[:num_final]


@njit(cache=True)
def _score_single_query(
    q_seq, q_len, k, alpha_size,
    kmer_offsets, kmer_entries, kmer_freqs, freq_thresh,
    num_db, min_total_hits, min_diag_hits, diag_bin_width,
    phase_a_topk,
):
    """Score one query – dispatches to Phase A only or two-stage A+B."""
    if min_diag_hits <= 1:
        return _score_query_phase_a(
            q_seq, q_len, k, alpha_size,
            kmer_offsets, kmer_entries, kmer_freqs, freq_thresh,
            num_db, min_total_hits,
        )
    return _score_query_two_stage(
        q_seq, q_len, k, alpha_size,
        kmer_offsets, kmer_entries, kmer_freqs, freq_thresh,
        num_db, min_total_hits, min_diag_hits, diag_bin_width,
        phase_a_topk,
    )


# ──────────────────────────────────────────────────────────────────────────
# Batch query scoring (parallel over queries)
# ──────────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True)
def _batch_score_queries(
    q_flat,           # uint8[] – concatenated query sequences
    q_offsets,        # int64[] – per-query start offset in q_flat
    q_lengths,        # int32[] – per-query length
    k,                # int32
    alpha_size,       # int32
    kmer_offsets,     # int64[]
    kmer_entries,     # int64[]
    kmer_freqs,       # int32[]
    freq_thresh,      # int32
    num_db,           # int32
    min_total_hits,   # int32
    min_diag_hits,    # int32
    diag_bin_width,   # int32
    max_cands,        # int32 – cap per query
    phase_a_topk,     # int32 – Phase A top-K for selective Phase B
    out_targets,      # int32[nq, max_cands] – pre-allocated
    out_counts,       # int32[nq]            – pre-allocated
):
    """Score all queries in parallel, writing results to *out_targets*."""
    nq = len(q_lengths)
    for qi in prange(nq):
        qs = int64(q_offsets[qi])
        ql = int32(q_lengths[qi])
        q_seq = q_flat[qs:qs + ql]

        cand_ids, cand_scores = _score_single_query(
            q_seq, ql,
            k, alpha_size,
            kmer_offsets, kmer_entries, kmer_freqs, freq_thresh,
            num_db, min_total_hits, min_diag_hits, diag_bin_width,
            phase_a_topk,
        )

        nc = len(cand_ids)
        if nc > max_cands:
            # Take top-scoring candidates
            order = np.argsort(-cand_scores)
            for j in range(max_cands):
                out_targets[qi, j] = cand_ids[order[j]]
            out_counts[qi] = max_cands
        else:
            for j in range(nc):
                out_targets[qi, j] = cand_ids[j]
            out_counts[qi] = int32(nc)


# ──────────────────────────────────────────────────────────────────────────
# IDF-weighted Phase A scoring
# ──────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _score_query_two_stage_idf(
    q_seq, q_len, k, alpha_size,
    kmer_offsets, kmer_entries, kmer_freqs, freq_thresh,
    num_db, min_total_hits, min_diag_hits, diag_bin_width,
    phase_a_topk, idf_weights,
):
    """Two-stage scoring with IDF-weighted Phase A.

    Phase A accumulates IDF scores (log2(N/freq)) instead of raw counts.
    This upweights rare k-mers, making candidate ranking more informative
    for distant homology detection.  Phase B remains count-based.
    """
    num_kmers = q_len - k + 1
    if num_kmers <= 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    # ── Phase A: IDF-weighted scoring ────────────────────────────────
    target_scores = np.zeros(num_db, dtype=np.float32)
    target_counts = np.zeros(num_db, dtype=np.int16)  # for Phase B prealloc

    for qpos in range(num_kmers):
        kmer_val = int64(0)
        valid = True
        for j in range(k):
            r = q_seq[qpos + j]
            if r >= alpha_size:
                valid = False
                break
            kmer_val = kmer_val * int64(alpha_size) + int64(r)
        if not valid:
            continue
        if kmer_freqs[kmer_val] > freq_thresh:
            continue
        w = idf_weights[kmer_val]
        s = kmer_offsets[kmer_val]
        e = kmer_offsets[kmer_val + 1]
        for h in range(s, e):
            tid = int32(kmer_entries[h] >> 32)
            target_scores[tid] += w
            if target_counts[tid] < 32767:
                target_counts[tid] += int16(1)

    # ── Select top-K from Phase A (by IDF score) ────────────────────
    min_score_f = np.float32(min_total_hits)
    num_passing = int32(0)
    for i in range(num_db):
        if target_scores[i] >= min_score_f:
            num_passing += int32(1)

    if num_passing == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    passing_ids = np.empty(num_passing, dtype=np.int32)
    passing_idf = np.empty(num_passing, dtype=np.float32)
    pos = int32(0)
    for i in range(num_db):
        if target_scores[i] >= min_score_f:
            passing_ids[pos] = int32(i)
            passing_idf[pos] = target_scores[i]
            pos += int32(1)

    if min_diag_hits <= 1:
        result_scores = np.empty(num_passing, dtype=np.int32)
        for i in range(num_passing):
            result_scores[i] = int32(passing_idf[i])
        return passing_ids, result_scores

    # Select top-K by IDF score for Phase B
    topk = min(int32(phase_a_topk), num_passing)
    if topk < num_passing:
        order = np.argsort(-passing_idf)
        topk_ids = np.empty(topk, dtype=np.int32)
        for i in range(topk):
            topk_ids[i] = passing_ids[order[i]]
    else:
        topk_ids = passing_ids

    # ── Phase B: diagonal scoring (count-based, same as standard) ───
    survivor_mask = np.zeros(num_db, dtype=np.bool_)
    for i in range(len(topk_ids)):
        survivor_mask[topk_ids[i]] = True

    n_surv_hits = int64(0)
    for i in range(len(topk_ids)):
        n_surv_hits += int64(target_counts[topk_ids[i]])

    DIAG_MULT = int64(1000000)
    max_diag_shift = int32(q_len)
    surv_keys = np.empty(n_surv_hits, dtype=np.int64)
    sw = int64(0)

    for qpos in range(num_kmers):
        kmer_val = int64(0)
        valid = True
        for j in range(k):
            r = q_seq[qpos + j]
            if r >= alpha_size:
                valid = False
                break
            kmer_val = kmer_val * int64(alpha_size) + int64(r)
        if not valid:
            continue
        if kmer_freqs[kmer_val] > freq_thresh:
            continue
        s = kmer_offsets[kmer_val]
        e = kmer_offsets[kmer_val + 1]
        for h in range(s, e):
            entry = kmer_entries[h]
            tid = int32(entry >> 32)
            if survivor_mask[tid]:
                tpos = int32(entry & 0xFFFFFFFF)
                diag = int32(tpos) - int32(qpos) + max_diag_shift
                dbin = int32(diag // diag_bin_width)
                surv_keys[sw] = int64(tid) * DIAG_MULT + int64(dbin)
                sw += int64(1)

    surv_keys = surv_keys[:sw]
    surv_keys.sort()

    final_ids = np.empty(len(topk_ids), dtype=np.int32)
    final_scores = np.empty(len(topk_ids), dtype=np.int32)
    num_final = int32(0)
    prev_tid = int32(-1)
    best_count = int32(0)
    i = int64(0)
    n = int64(len(surv_keys))

    while i < n:
        key = surv_keys[i]
        tid = int32(key // DIAG_MULT)
        dbin = int32(key % DIAG_MULT)
        run = int32(0)
        while i < n:
            k2 = surv_keys[i]
            if int32(k2 // DIAG_MULT) != tid or int32(k2 % DIAG_MULT) != dbin:
                break
            run += int32(1)
            i += int64(1)
        if tid != prev_tid:
            if prev_tid >= 0 and best_count >= min_diag_hits:
                final_ids[num_final] = prev_tid
                final_scores[num_final] = best_count
                num_final += int32(1)
            prev_tid = tid
            best_count = run
        else:
            if run > best_count:
                best_count = run

    if prev_tid >= 0 and best_count >= min_diag_hits:
        final_ids[num_final] = prev_tid
        final_scores[num_final] = best_count
        num_final += int32(1)

    return final_ids[:num_final], final_scores[:num_final]


@njit(parallel=True, cache=True)
def _batch_score_queries_idf(
    q_flat, q_offsets, q_lengths,
    k, alpha_size,
    kmer_offsets, kmer_entries, kmer_freqs,
    freq_thresh, num_db, min_total_hits,
    min_diag_hits, diag_bin_width,
    max_cands, phase_a_topk,
    idf_weights,
    out_targets, out_counts,
):
    """Score all queries with IDF-weighted Phase A (parallel)."""
    nq = len(q_lengths)
    for qi in prange(nq):
        qs = int64(q_offsets[qi])
        ql = int32(q_lengths[qi])
        q_seq = q_flat[qs:qs + ql]

        cand_ids, cand_scores = _score_query_two_stage_idf(
            q_seq, ql,
            k, alpha_size,
            kmer_offsets, kmer_entries, kmer_freqs, freq_thresh,
            num_db, min_total_hits, min_diag_hits, diag_bin_width,
            phase_a_topk, idf_weights,
        )

        nc = len(cand_ids)
        if nc > max_cands:
            order = np.argsort(-cand_scores)
            for j in range(max_cands):
                out_targets[qi, j] = cand_ids[order[j]]
            out_counts[qi] = max_cands
        else:
            for j in range(nc):
                out_targets[qi, j] = cand_ids[j]
            out_counts[qi] = int32(nc)


# ──────────────────────────────────────────────────────────────────────────
# Similar k-mer matching (BLOSUM62-aware, k=5)
# ──────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _score_query_similar_k5(
    q_seq, q_len,
    kmer_offsets, kmer_entries, kmer_freqs, freq_thresh,
    num_db, min_total_hits, sub_matrix, kmer_score_thresh,
):
    """Phase A with BLOSUM62 similar k-mer matching for k=5.

    For each query 5-mer, enumerates all 5-mers whose BLOSUM62 column-sum
    score is >= *kmer_score_thresh*, using branch-and-bound pruning (5
    nested loops with early exit when partial score + remaining max <
    threshold).  Each matching 5-mer's posting list is traversed, and
    per-target *score-weighted* counts are accumulated (weight = BLOSUM62
    sum score of the matching k-mer).  This ensures exact matches (~25)
    outweigh marginal neighbors (~11), preventing noise from drowning signal.

    Returns ``(target_ids, scores)`` for targets with scores >= *min_total_hits*.
    """
    k = int32(5)
    alpha = int32(20)
    num_kmers = q_len - k + 1
    if num_kmers <= 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    # Max BLOSUM62 score per amino acid (diagonal is always max)
    max_sc = np.empty(20, dtype=np.int32)
    for i in range(20):
        mx = int32(-100)
        for j in range(20):
            v = int32(sub_matrix[i, j])
            if v > mx:
                mx = v
        max_sc[i] = mx

    target_counts = np.zeros(num_db, dtype=np.int32)

    for qpos in range(num_kmers):
        q0 = int32(q_seq[qpos])
        q1 = int32(q_seq[qpos + 1])
        q2 = int32(q_seq[qpos + 2])
        q3 = int32(q_seq[qpos + 3])
        q4 = int32(q_seq[qpos + 4])

        if q0 >= 20 or q1 >= 20 or q2 >= 20 or q3 >= 20 or q4 >= 20:
            continue

        T = kmer_score_thresh
        mrem1234 = max_sc[q1] + max_sc[q2] + max_sc[q3] + max_sc[q4]
        mrem234 = max_sc[q2] + max_sc[q3] + max_sc[q4]
        mrem34 = max_sc[q3] + max_sc[q4]
        mrem4 = max_sc[q4]

        for a0 in range(alpha):
            s0 = int32(sub_matrix[q0, a0])
            if s0 + mrem1234 < T:
                continue
            v0 = int64(a0) * int64(160000)  # a0 * 20^4

            for a1 in range(alpha):
                s1 = s0 + int32(sub_matrix[q1, a1])
                if s1 + mrem234 < T:
                    continue
                v1 = v0 + int64(a1) * int64(8000)  # + a1 * 20^3

                for a2 in range(alpha):
                    s2 = s1 + int32(sub_matrix[q2, a2])
                    if s2 + mrem34 < T:
                        continue
                    v2 = v1 + int64(a2) * int64(400)  # + a2 * 20^2

                    for a3 in range(alpha):
                        s3 = s2 + int32(sub_matrix[q3, a3])
                        if s3 + mrem4 < T:
                            continue
                        v3 = v2 + int64(a3) * int64(20)

                        for a4 in range(alpha):
                            s4 = s3 + int32(sub_matrix[q4, a4])
                            if s4 < T:
                                continue
                            kmer_val = v3 + int64(a4)

                            freq = kmer_freqs[kmer_val]
                            if freq == 0 or freq > freq_thresh:
                                continue

                            start = kmer_offsets[kmer_val]
                            end = kmer_offsets[kmer_val + 1]
                            for h in range(start, end):
                                tid = int32(kmer_entries[h] >> 32)
                                target_counts[tid] += s4

    # Collect survivors
    num_survivors = int32(0)
    for i in range(num_db):
        if target_counts[i] >= min_total_hits:
            num_survivors += int32(1)

    if num_survivors == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    result_ids = np.empty(num_survivors, dtype=np.int32)
    result_scores = np.empty(num_survivors, dtype=np.int32)
    pos = int32(0)
    for i in range(num_db):
        if target_counts[i] >= min_total_hits:
            result_ids[pos] = int32(i)
            result_scores[pos] = target_counts[i]
            pos += int32(1)
    return result_ids, result_scores


@njit(parallel=True, cache=True)
def _batch_score_queries_similar(
    q_flat, q_offsets, q_lengths,
    kmer_offsets, kmer_entries, kmer_freqs,
    freq_thresh, num_db, min_total_hits,
    max_cands,
    sub_matrix, kmer_score_thresh,
    out_targets, out_counts,
):
    """Score all queries using BLOSUM62 similar k-mer matching (k=5, parallel)."""
    nq = len(q_lengths)
    for qi in prange(nq):
        qs = int64(q_offsets[qi])
        ql = int32(q_lengths[qi])
        q_seq = q_flat[qs:qs + ql]

        cand_ids, cand_scores = _score_query_similar_k5(
            q_seq, ql,
            kmer_offsets, kmer_entries, kmer_freqs, freq_thresh,
            num_db, min_total_hits, sub_matrix, kmer_score_thresh,
        )

        nc = len(cand_ids)
        if nc > max_cands:
            order = np.argsort(-cand_scores)
            for j in range(max_cands):
                out_targets[qi, j] = cand_ids[order[j]]
            out_counts[qi] = max_cands
        else:
            for j in range(nc):
                out_targets[qi, j] = cand_ids[j]
            out_counts[qi] = int32(nc)


# ──────────────────────────────────────────────────────────────────────────
# High-level search function
# ──────────────────────────────────────────────────────────────────────────

def search_kmer_index(
    db_index,
    query_dataset,
    threshold: float = 0.5,
    top_k: int = 10,
    band_width: int | None = None,
    device: str = "cpu",
    min_total_hits: int = 2,
    min_diag_hits: int = 2,
    diag_bin_width: int = 10,
    max_cands_per_query: int = 2000,
    phase_a_topk: int = 10000,
    freq_percentile: float = 95.0,
    min_ungapped_score: int = 0,
    kmer_score_thresh: int = 13,
    local_alignment: bool = True,
    evalue_normalize: bool = True,
    reduced_alphabet: bool = False,
    reduced_k: int | list[int] = 5,
    use_idf: bool = False,
    spaced_seeds: list[str] | None = None,
    extra_alphabets: list[tuple] | None = None,
):
    """Search queries against a pre-built k-mer inverted index.

    Stages:
        1. K-mer index lookup + diagonal scoring  →  candidate pairs
        2. Banded NW alignment on candidates
        3. Collect top-k hits per query

    Args:
        db_index: :class:`DatabaseIndex` with ``kmer_offsets`` / ``kmer_entries``.
        query_dataset: Query :class:`SequenceDataset`.
        threshold: Minimum identity to report.
        top_k: Max hits per query.
        band_width: Half-width for banded NW (auto if *None*).
        device: ``"cpu"`` or GPU id.
        min_total_hits: Phase-A k-mer count threshold.
        min_diag_hits: Phase-B diagonal-bin threshold.
        diag_bin_width: Diagonal bin width in residues.
        max_cands_per_query: Cap on candidates entering alignment per query.
        freq_percentile: Percentile for k-mer frequency filtering.

    Returns:
        :class:`SearchResults`.
    """
    import time
    from clustkit.search import (
        SearchResults,
        _merge_sequences_for_alignment,
        _remap_pairs_to_merged,
        _collect_top_k_hits,
    )
    from clustkit.pairwise import (
        _batch_align, _batch_align_compact,
        _batch_align_compact_scored, _batch_sw_compact_scored,
        _batch_ungapped_prefilter, BLOSUM62,
    )

    start_time = time.perf_counter()

    nq = query_dataset.num_sequences
    nd = db_index.dataset.num_sequences
    params = db_index.params
    mode = params["mode"]
    k = int32(params.get("kmer_index_k", params["kmer_size"]))
    alpha_size = int32(20 if mode == "protein" else 4)

    logger.info(
        f"Search (k-mer index): {nq} queries against {nd} database "
        f"sequences, k={k}"
    )

    if nq == 0 or nd == 0:
        return SearchResults(
            hits=[[] for _ in range(nq)],
            num_queries=nq, num_targets=nd,
            num_candidates=0, num_aligned=0,
            runtime_seconds=time.perf_counter() - start_time,
        )

    # ── Stage 1: k-mer index scoring ─────────────────────────────────
    freq_thresh = compute_freq_threshold(
        db_index.kmer_freqs, nd, freq_percentile,
    )
    logger.info(f"  K-mer freq threshold: {freq_thresh}")

    # Ensure query compact format
    q_flat = query_dataset.flat_sequences
    q_off = query_dataset.offsets
    q_lens = query_dataset.lengths
    if q_flat is None or q_off is None:
        raise ValueError("Query dataset must have compact (flat) storage")

    out_targets = np.empty((nq, max_cands_per_query), dtype=np.int32)
    out_counts = np.zeros(nq, dtype=np.int32)

    use_similar = (k >= 5 and mode == "protein" and kmer_score_thresh > 0)
    if use_similar:
        logger.info(
            f"  Using BLOSUM62 similar k-mer matching "
            f"(k={k}, score_thresh={kmer_score_thresh})"
        )

    # Pre-compute IDF weights if enabled
    if use_idf and not use_similar:
        std_idf = np.log2(
            np.maximum(
                np.float32(nd)
                / np.maximum(db_index.kmer_freqs.astype(np.float32), 1.0),
                1.0,
            )
        ).astype(np.float32)
        logger.info("  Using IDF-weighted Phase A scoring")

    with timer("Search Stage 1: K-mer index scoring"):
        if use_similar:
            _batch_score_queries_similar(
                q_flat,
                q_off.astype(np.int64),
                q_lens.astype(np.int32),
                db_index.kmer_offsets,
                db_index.kmer_entries,
                db_index.kmer_freqs,
                freq_thresh,
                int32(nd),
                int32(min_total_hits),
                int32(max_cands_per_query),
                BLOSUM62,
                int32(kmer_score_thresh),
                out_targets,
                out_counts,
            )
        elif use_idf:
            _batch_score_queries_idf(
                q_flat,
                q_off.astype(np.int64),
                q_lens.astype(np.int32),
                k, alpha_size,
                db_index.kmer_offsets,
                db_index.kmer_entries,
                db_index.kmer_freqs,
                freq_thresh,
                int32(nd),
                int32(min_total_hits),
                int32(min_diag_hits),
                int32(diag_bin_width),
                int32(max_cands_per_query),
                int32(phase_a_topk),
                std_idf,
                out_targets,
                out_counts,
            )
        else:
            _batch_score_queries(
                q_flat,
                q_off.astype(np.int64),
                q_lens.astype(np.int32),
                k, alpha_size,
                db_index.kmer_offsets,
                db_index.kmer_entries,
                db_index.kmer_freqs,
                freq_thresh,
                int32(nd),
                int32(min_total_hits),
                int32(min_diag_hits),
                int32(diag_bin_width),
                int32(max_cands_per_query),
                int32(phase_a_topk),
                out_targets,
                out_counts,
            )

    # Flatten results into (M, 2) candidate pairs
    total_cands = int(out_counts.sum())
    logger.info(
        f"  K-mer scoring: {total_cands} candidate pairs "
        f"(avg {total_cands / max(nq, 1):.1f}/query)"
    )

    if total_cands == 0:
        return SearchResults(
            hits=[[] for _ in range(nq)],
            num_queries=nq, num_targets=nd,
            num_candidates=0, num_aligned=0,
            runtime_seconds=time.perf_counter() - start_time,
        )

    candidate_pairs = np.empty((total_cands, 2), dtype=np.int32)
    pos = 0
    for qi in range(nq):
        nc = int(out_counts[qi])
        if nc > 0:
            candidate_pairs[pos:pos + nc, 0] = qi
            candidate_pairs[pos:pos + nc, 1] = out_targets[qi, :nc]
            pos += nc

    # ── Stage 1.5: Reduced alphabet candidates (optional) ────────────
    if reduced_alphabet and mode == "protein":
        # Support single k or list of k values (triple/multi index)
        rk_values = (
            reduced_k if isinstance(reduced_k, list) else [reduced_k]
        )

        # Remap query sequences once (shared across all reduced indices)
        red_q_flat = _remap_flat(q_flat, REDUCED_ALPHA, len(q_flat))
        all_red_packed = []

        for rk in rk_values:
            with timer(f"Search Stage 1.5: Reduced alphabet k={rk}"):
                # Use cached reduced index if available
                _cache_key = f'_red_idx_k{rk}'
                if hasattr(db_index, _cache_key):
                    red_offsets, red_entries, red_freqs = getattr(
                        db_index, _cache_key
                    )
                    logger.info(
                        f"  Using cached reduced alphabet index (k={rk})"
                    )
                else:
                    db_flat = db_index.dataset.flat_sequences
                    red_db_flat = _remap_flat(
                        db_flat, REDUCED_ALPHA, len(db_flat)
                    )
                    red_offsets, red_entries, red_freqs = build_kmer_index(
                        red_db_flat,
                        db_index.dataset.offsets,
                        db_index.dataset.lengths,
                        rk, mode,
                        alpha_size=REDUCED_ALPHA_SIZE,
                    )
                    setattr(db_index, _cache_key, (
                        red_offsets, red_entries, red_freqs,
                    ))

                red_freq_thresh = compute_freq_threshold(
                    red_freqs, nd, freq_percentile,
                )

                red_mc = max_cands_per_query
                red_out_targets = np.empty((nq, red_mc), dtype=np.int32)
                red_out_counts = np.zeros(nq, dtype=np.int32)

                # Use IDF scoring for reduced index too
                if use_idf:
                    red_idf = np.log2(
                        np.maximum(
                            np.float32(nd)
                            / np.maximum(
                                red_freqs.astype(np.float32), 1.0
                            ),
                            1.0,
                        )
                    ).astype(np.float32)
                    _batch_score_queries_idf(
                        red_q_flat,
                        q_off.astype(np.int64),
                        q_lens.astype(np.int32),
                        int32(rk), int32(REDUCED_ALPHA_SIZE),
                        red_offsets, red_entries, red_freqs,
                        red_freq_thresh,
                        int32(nd), int32(min_total_hits),
                        int32(min_diag_hits), int32(diag_bin_width),
                        int32(red_mc), int32(phase_a_topk),
                        red_idf,
                        red_out_targets, red_out_counts,
                    )
                else:
                    _batch_score_queries(
                        red_q_flat,
                        q_off.astype(np.int64),
                        q_lens.astype(np.int32),
                        int32(rk), int32(REDUCED_ALPHA_SIZE),
                        red_offsets, red_entries, red_freqs,
                        red_freq_thresh,
                        int32(nd), int32(min_total_hits),
                        int32(min_diag_hits), int32(diag_bin_width),
                        int32(red_mc), int32(phase_a_topk),
                        red_out_targets, red_out_counts,
                    )

                # Flatten into packed pairs
                red_total = int(red_out_counts.sum())
                if red_total > 0:
                    red_pairs = np.empty((red_total, 2), dtype=np.int32)
                    rpos = 0
                    for qi in range(nq):
                        nc = int(red_out_counts[qi])
                        if nc > 0:
                            red_pairs[rpos:rpos + nc, 0] = qi
                            red_pairs[rpos:rpos + nc, 1] = (
                                red_out_targets[qi, :nc]
                            )
                            rpos += nc
                    packed = (
                        red_pairs[:, 0].astype(np.int64) * nd
                        + red_pairs[:, 1].astype(np.int64)
                    )
                    all_red_packed.append(packed)
                    logger.info(
                        f"  Reduced k={rk}: {red_total} candidates"
                    )

        # Union all candidate sets (standard + all reduced)
        if all_red_packed:
            std_packed = (
                candidate_pairs[:, 0].astype(np.int64) * nd
                + candidate_pairs[:, 1].astype(np.int64)
            )
            all_packed_arrays = [std_packed] + all_red_packed
            union_packed = np.unique(np.concatenate(all_packed_arrays))
            n_before = len(candidate_pairs)
            candidate_pairs = np.empty(
                (len(union_packed), 2), dtype=np.int32
            )
            candidate_pairs[:, 0] = (union_packed // nd).astype(np.int32)
            candidate_pairs[:, 1] = (union_packed % nd).astype(np.int32)
            total_cands = len(candidate_pairs)
            logger.info(
                f"  Reduced alphabet union: {total_cands} total "
                f"(was {n_before} from standard index)"
            )

    # ── Stage 1.55: Extra alphabet candidates (optional) ────────────
    # Each entry: (name, alpha_map, alpha_size, k_values)
    if extra_alphabets and mode == "protein":
        extra_packed = []
        for alpha_name, alpha_map, alpha_sz, alpha_k_vals in extra_alphabets:
            ext_q_flat = _remap_flat(q_flat, alpha_map, len(q_flat))

            for rk in alpha_k_vals:
                with timer(
                    f"Search Stage 1.55: {alpha_name} k={rk}"
                ):
                    _ecache = f'_ext_{alpha_name}_k{rk}'
                    if hasattr(db_index, _ecache):
                        ext_off, ext_ent, ext_freq = getattr(
                            db_index, _ecache
                        )
                        logger.info(
                            f"  Using cached {alpha_name} index (k={rk})"
                        )
                    else:
                        db_flat = db_index.dataset.flat_sequences
                        ext_db_flat = _remap_flat(
                            db_flat, alpha_map, len(db_flat)
                        )
                        ext_off, ext_ent, ext_freq = build_kmer_index(
                            ext_db_flat,
                            db_index.dataset.offsets,
                            db_index.dataset.lengths,
                            rk, mode, alpha_size=alpha_sz,
                        )
                        setattr(db_index, _ecache, (
                            ext_off, ext_ent, ext_freq,
                        ))

                    ext_freq_thresh = compute_freq_threshold(
                        ext_freq, nd, freq_percentile,
                    )
                    ext_mc = max_cands_per_query
                    ext_out_t = np.empty((nq, ext_mc), dtype=np.int32)
                    ext_out_c = np.zeros(nq, dtype=np.int32)

                    _batch_score_queries(
                        ext_q_flat,
                        q_off.astype(np.int64),
                        q_lens.astype(np.int32),
                        int32(rk), int32(alpha_sz),
                        ext_off, ext_ent, ext_freq,
                        ext_freq_thresh,
                        int32(nd), int32(min_total_hits),
                        int32(min_diag_hits), int32(diag_bin_width),
                        int32(ext_mc), int32(phase_a_topk),
                        ext_out_t, ext_out_c,
                    )

                    ext_total = int(ext_out_c.sum())
                    if ext_total > 0:
                        ext_pairs = np.empty(
                            (ext_total, 2), dtype=np.int32
                        )
                        rpos = 0
                        for qi in range(nq):
                            nc = int(ext_out_c[qi])
                            if nc > 0:
                                ext_pairs[rpos:rpos + nc, 0] = qi
                                ext_pairs[rpos:rpos + nc, 1] = (
                                    ext_out_t[qi, :nc]
                                )
                                rpos += nc
                        packed = (
                            ext_pairs[:, 0].astype(np.int64) * nd
                            + ext_pairs[:, 1].astype(np.int64)
                        )
                        extra_packed.append(packed)
                        logger.info(
                            f"  {alpha_name} k={rk}: "
                            f"{ext_total} candidates"
                        )

        if extra_packed:
            std_packed = (
                candidate_pairs[:, 0].astype(np.int64) * nd
                + candidate_pairs[:, 1].astype(np.int64)
            )
            all_arrays = [std_packed] + extra_packed
            union_packed = np.unique(np.concatenate(all_arrays))
            n_before = len(candidate_pairs)
            candidate_pairs = np.empty(
                (len(union_packed), 2), dtype=np.int32
            )
            candidate_pairs[:, 0] = (
                union_packed // nd
            ).astype(np.int32)
            candidate_pairs[:, 1] = (
                union_packed % nd
            ).astype(np.int32)
            total_cands = len(candidate_pairs)
            logger.info(
                f"  Extra alphabet union: {total_cands} total "
                f"(was {n_before})"
            )

    # ── Stage 1.6: Spaced seed candidates (optional) ────────────────
    if spaced_seeds and mode == "protein":
        # Remap to reduced alphabet (reuse if already done above)
        if not (reduced_alphabet and mode == "protein"):
            red_q_flat = _remap_flat(q_flat, REDUCED_ALPHA, len(q_flat))

        spaced_packed = []
        for pattern in spaced_seeds:
            with timer(f"Search Stage 1.6: Spaced seed {pattern}"):
                _sp_cache = f'_spaced_{pattern}'
                if hasattr(db_index, _sp_cache):
                    sp_data = getattr(db_index, _sp_cache)
                    sp_off, sp_ent, sp_freq = sp_data[:3]
                    sp_seed_off, sp_weight, sp_span = sp_data[3:]
                    logger.info(
                        f"  Using cached spaced seed index ({pattern})"
                    )
                else:
                    db_flat = db_index.dataset.flat_sequences
                    red_db_flat = _remap_flat(
                        db_flat, REDUCED_ALPHA, len(db_flat)
                    )
                    (sp_off, sp_ent, sp_freq,
                     sp_seed_off, sp_weight, sp_span) = (
                        build_kmer_index_spaced(
                            red_db_flat,
                            db_index.dataset.offsets,
                            db_index.dataset.lengths,
                            pattern, mode,
                            alpha_size=REDUCED_ALPHA_SIZE,
                        )
                    )
                    setattr(db_index, _sp_cache, (
                        sp_off, sp_ent, sp_freq,
                        sp_seed_off, sp_weight, sp_span,
                    ))

                sp_freq_thresh = compute_freq_threshold(
                    sp_freq, nd, freq_percentile,
                )

                sp_mc = max_cands_per_query
                sp_out_targets = np.empty((nq, sp_mc), dtype=np.int32)
                sp_out_counts = np.zeros(nq, dtype=np.int32)

                _batch_score_queries_spaced(
                    red_q_flat,
                    q_off.astype(np.int64),
                    q_lens.astype(np.int32),
                    sp_seed_off, int32(sp_weight), int32(sp_span),
                    int32(REDUCED_ALPHA_SIZE),
                    sp_off, sp_ent, sp_freq,
                    sp_freq_thresh,
                    int32(nd), int32(min_total_hits),
                    int32(min_diag_hits), int32(diag_bin_width),
                    int32(sp_mc), int32(phase_a_topk),
                    sp_out_targets, sp_out_counts,
                )

                sp_total = int(sp_out_counts.sum())
                if sp_total > 0:
                    sp_pairs = np.empty((sp_total, 2), dtype=np.int32)
                    rpos = 0
                    for qi in range(nq):
                        nc = int(sp_out_counts[qi])
                        if nc > 0:
                            sp_pairs[rpos:rpos + nc, 0] = qi
                            sp_pairs[rpos:rpos + nc, 1] = (
                                sp_out_targets[qi, :nc]
                            )
                            rpos += nc
                    packed = (
                        sp_pairs[:, 0].astype(np.int64) * nd
                        + sp_pairs[:, 1].astype(np.int64)
                    )
                    spaced_packed.append(packed)
                    logger.info(
                        f"  Spaced {pattern}: {sp_total} candidates"
                    )

        # Union spaced seed candidates with existing
        if spaced_packed:
            std_packed = (
                candidate_pairs[:, 0].astype(np.int64) * nd
                + candidate_pairs[:, 1].astype(np.int64)
            )
            all_arrays = [std_packed] + spaced_packed
            union_packed = np.unique(np.concatenate(all_arrays))
            n_before = len(candidate_pairs)
            candidate_pairs = np.empty(
                (len(union_packed), 2), dtype=np.int32
            )
            candidate_pairs[:, 0] = (union_packed // nd).astype(np.int32)
            candidate_pairs[:, 1] = (union_packed % nd).astype(np.int32)
            total_cands = len(candidate_pairs)
            logger.info(
                f"  Spaced seed union: {total_cands} total "
                f"(was {n_before})"
            )

    # ── Merge sequences for alignment stages ─────────────────────────
    if band_width is None:
        all_lengths = np.concatenate(
            [query_dataset.lengths, db_index.dataset.lengths]
        )
        p95_len = int(np.percentile(all_lengths, 95))
        band_width = max(20, int(p95_len * 0.3))

    merged = _merge_sequences_for_alignment(query_dataset, db_index.dataset)
    merged_lengths = merged["lengths"]
    nq_offset = merged["nq"]

    merged_pairs = _remap_pairs_to_merged(candidate_pairs, nq_offset)

    # Sort for cache locality
    sort_key = np.minimum(merged_pairs[:, 0], merged_pairs[:, 1])
    sort_order = np.argsort(sort_key, kind="mergesort")
    merged_pairs = merged_pairs[sort_order]
    candidate_pairs = candidate_pairs[sort_order]

    use_blosum = (mode == "protein") and (merged["flat_sequences"] is not None)

    # ── Stage 1.5: Ungapped extension pre-filter ──────────────────
    if use_blosum and min_ungapped_score > 0:
        with timer("Search Stage 1.5: Ungapped pre-filter"):
            ug_mask = _batch_ungapped_prefilter(
                merged_pairs,
                merged["flat_sequences"],
                merged["offsets"],
                merged_lengths,
                BLOSUM62,
                int32(min_ungapped_score),
            )
            n_before = len(merged_pairs)
            merged_pairs = merged_pairs[ug_mask]
            candidate_pairs = candidate_pairs[ug_mask]
            n_after = len(merged_pairs)
            logger.info(
                f"  Ungapped pre-filter: {n_before} -> {n_after} pairs "
                f"({n_before - n_after} removed, "
                f"{100 * (n_before - n_after) / max(n_before, 1):.1f}%)"
            )

    # ── Stage 2: Alignment ──────────────────────────────────────────
    num_aligned = len(candidate_pairs)
    use_sw = local_alignment and use_blosum
    aln_label = "SW local alignment" if use_sw else "Banded NW alignment"
    logger.info(
        f"  Alignment: {num_aligned} pairs, band_width={band_width}, "
        f"threshold={threshold}, mode={aln_label}"
    )

    with timer(f"Search Stage 2: {aln_label}"):
        if use_sw:
            sims, raw_scores, aln_mask = _batch_sw_compact_scored(
                merged_pairs,
                merged["flat_sequences"],
                merged["offsets"],
                merged_lengths,
                np.float32(threshold),
                int32(band_width),
                BLOSUM62,
            )
        elif use_blosum:
            sims, raw_scores, aln_mask = _batch_align_compact_scored(
                merged_pairs,
                merged["flat_sequences"],
                merged["offsets"],
                merged_lengths,
                np.float32(threshold),
                int32(band_width),
                BLOSUM62,
            )
        elif merged["flat_sequences"] is not None:
            sims, aln_mask = _batch_align_compact(
                merged_pairs,
                merged["flat_sequences"],
                merged["offsets"],
                merged_lengths,
                np.float32(threshold),
                int32(band_width),
            )
            raw_scores = None
        else:
            sims, aln_mask = _batch_align(
                merged_pairs,
                merged["encoded_sequences"],
                merged_lengths,
                np.float32(threshold),
                int32(band_width),
            )
            raw_scores = None

    passing_pairs = candidate_pairs[aln_mask]
    passing_sims = sims[aln_mask]
    passing_scores = raw_scores[aln_mask].astype(np.float32) if raw_scores is not None else None

    # E-value normalization for SW scores: rank_score = lambda*S - log(m*n)
    # Normalises for sequence length so short true homologs aren't dwarfed
    # by long random alignments.  Only valid for local alignment scores.
    if use_sw and evalue_normalize and passing_scores is not None and len(passing_pairs) > 0:
        LAMBDA = np.float32(0.267)
        q_lens = query_dataset.lengths[passing_pairs[:, 0]].astype(np.float32)
        t_lens = db_index.dataset.lengths[passing_pairs[:, 1]].astype(np.float32)
        passing_scores = LAMBDA * passing_scores - np.log(q_lens * t_lens)

    logger.info(f"  {len(passing_pairs)} pairs above threshold {threshold}")

    # ── Collect top-k hits ───────────────────────────────────────────
    hits = _collect_top_k_hits(
        passing_pairs, passing_sims, nq, top_k,
        query_dataset, db_index.dataset,
        passing_scores=passing_scores,
    )

    total_hits = sum(len(h) for h in hits)
    queries_with_hits = sum(1 for h in hits if len(h) > 0)
    logger.info(
        f"  {total_hits} total hits for {queries_with_hits}/{nq} queries"
    )

    elapsed = time.perf_counter() - start_time
    logger.info(f"Search completed in {elapsed:.2f}s")

    return SearchResults(
        hits=hits,
        num_queries=nq,
        num_targets=nd,
        num_candidates=total_cands,
        num_aligned=num_aligned,
        runtime_seconds=elapsed,
    )
