#!/usr/bin/env python3
"""Benchmark: C/OpenMP SW alignment vs Numba.

Tests correctness (score/identity match) and speed on real SCOPe data.
Also tests reduced band width (126 vs 50) for additional speedup.
"""

import ctypes
import random
import sys
import time
from pathlib import Path

import numpy as np
from numba import int32 as nb_int32, float32 as nb_float32

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numba
numba.set_num_threads(8)

from clustkit.database import load_database
from clustkit.io import read_sequences
from clustkit.kmer_index import (
    _batch_score_queries, compute_freq_threshold,
    build_kmer_index, REDUCED_ALPHA, REDUCED_ALPHA_SIZE, _remap_flat,
)
from clustkit.pairwise import _batch_sw_compact_scored, BLOSUM62
from clustkit.search import _merge_sequences_for_alignment, _remap_pairs_to_merged
from benchmarks.benchmark_scop_search import read_fasta, write_fasta

# ── Load C SW extension ──────────────────────────────────────────────

LIB_PATH = Path(__file__).resolve().parent.parent / "clustkit" / "csrc" / "sw_align.so"
_clib = ctypes.cdll.LoadLibrary(str(LIB_PATH))

_clib.batch_sw_align_c.restype = None
_clib.batch_sw_align_c.argtypes = [
    ctypes.c_void_p,   # pairs
    ctypes.c_void_p,   # flat_sequences
    ctypes.c_void_p,   # offsets
    ctypes.c_void_p,   # lengths
    ctypes.c_int32,    # M
    ctypes.c_int32,    # band_width
    ctypes.c_void_p,   # sub_matrix
    ctypes.c_float,    # threshold
    ctypes.c_void_p,   # out_sims
    ctypes.c_void_p,   # out_scores
    ctypes.c_void_p,   # out_mask
]


def run_c_sw(merged_pairs, flat_seqs, offsets, lengths, band_width, sub_matrix):
    M = len(merged_pairs)
    pairs_flat = np.ascontiguousarray(merged_pairs.flatten())
    out_sims = np.empty(M, dtype=np.float32)
    out_scores = np.empty(M, dtype=np.int32)
    out_mask = np.empty(M, dtype=np.uint8)

    _clib.batch_sw_align_c(
        pairs_flat.ctypes.data,
        flat_seqs.ctypes.data,
        offsets.ctypes.data,
        lengths.ctypes.data,
        M, band_width,
        sub_matrix.ctypes.data,
        0.1,  # threshold
        out_sims.ctypes.data,
        out_scores.ctypes.data,
        out_mask.ctypes.data,
    )
    return out_sims, out_scores, out_mask.astype(np.bool_)


def main():
    out_dir = Path("benchmarks/data/speed_sensitivity_results")
    scop_dir = Path("benchmarks/data/scop_search_results")

    import json
    with open(scop_dir / "metadata.json") as f:
        meta = json.load(f)
    random.seed(42)
    qsids = sorted(random.sample(meta["query_sids"], min(2000, len(meta["query_sids"]))))
    all_seqs = dict(read_fasta(scop_dir / "queries.fasta"))
    sub = [(s, all_seqs[s]) for s in qsids if s in all_seqs]
    query_fasta = str(out_dir / "queries_subset.fasta")
    write_fasta(sub, query_fasta)

    print("Loading database...", flush=True)
    db_index = load_database(out_dir / "clustkit_db")
    query_ds = read_sequences(query_fasta, "protein")
    db_ds = db_index.dataset
    nq = query_ds.num_sequences
    nd = db_ds.num_sequences
    print(f"Loaded {nq} queries, {nd} database\n", flush=True)

    # ── Generate candidates (dual k=5, mc=8K) ───────────────────────
    print("Generating candidates...", flush=True)
    q_flat = query_ds.flat_sequences
    q_off = query_ds.offsets.astype(np.int64)
    q_lens = query_ds.lengths.astype(np.int32)
    mc = 8000

    k = nb_int32(db_index.params.get("kmer_index_k", db_index.params["kmer_size"]))
    freq_thresh = compute_freq_threshold(db_index.kmer_freqs, nd, 99.5)
    out_t = np.empty((nq, mc), dtype=np.int32)
    out_c = np.zeros(nq, dtype=np.int32)
    _batch_score_queries(
        q_flat, q_off, q_lens, k, nb_int32(20),
        db_index.kmer_offsets, db_index.kmer_entries, db_index.kmer_freqs,
        nb_int32(int(freq_thresh)), nb_int32(nd), nb_int32(2), nb_int32(2), nb_int32(10),
        nb_int32(mc), nb_int32(200000), out_t, out_c)

    # Flatten + reduced k=5
    total_std = int(out_c.sum())
    cp = np.empty((total_std, 2), dtype=np.int32)
    p = 0
    for qi in range(nq):
        nc = int(out_c[qi])
        if nc > 0:
            cp[p:p+nc, 0] = qi
            cp[p:p+nc, 1] = out_t[qi, :nc]
            p += nc
    all_packed = [cp[:, 0].astype(np.int64) * nd + cp[:, 1].astype(np.int64)]

    red_q = _remap_flat(q_flat, REDUCED_ALPHA, len(q_flat))
    red_db = _remap_flat(db_ds.flat_sequences, REDUCED_ALPHA, len(db_ds.flat_sequences))
    r5o, r5e, r5f = build_kmer_index(red_db, db_ds.offsets, db_ds.lengths, 5, "protein",
                                      alpha_size=REDUCED_ALPHA_SIZE)
    r5ft = compute_freq_threshold(r5f, nd, 99.5)
    rt = np.empty((nq, mc), dtype=np.int32)
    rc = np.zeros(nq, dtype=np.int32)
    _batch_score_queries(red_q, q_off, q_lens, nb_int32(5), nb_int32(REDUCED_ALPHA_SIZE),
                         r5o, r5e, r5f, nb_int32(int(r5ft)), nb_int32(nd), nb_int32(2),
                         nb_int32(2), nb_int32(10), nb_int32(mc), nb_int32(200000), rt, rc)
    tot = int(rc.sum())
    if tot > 0:
        rp = np.empty((tot, 2), dtype=np.int32)
        pp = 0
        for qi in range(nq):
            nc = int(rc[qi])
            if nc > 0:
                rp[pp:pp+nc, 0] = qi; rp[pp:pp+nc, 1] = rt[qi, :nc]; pp += nc
        all_packed.append(rp[:, 0].astype(np.int64) * nd + rp[:, 1].astype(np.int64))

    union = np.unique(np.concatenate(all_packed))
    candidate_pairs = np.empty((len(union), 2), dtype=np.int32)
    candidate_pairs[:, 0] = (union // nd).astype(np.int32)
    candidate_pairs[:, 1] = (union % nd).astype(np.int32)
    print(f"  {len(candidate_pairs)} candidate pairs\n", flush=True)

    # ── Prepare alignment ────────────────────────────────────────────
    merged = _merge_sequences_for_alignment(query_ds, db_ds)
    merged_pairs = _remap_pairs_to_merged(candidate_pairs, merged["nq"])
    sort_key = np.minimum(merged_pairs[:, 0], merged_pairs[:, 1])
    sort_order = np.argsort(sort_key, kind="mergesort")
    merged_pairs = merged_pairs[sort_order]
    candidate_pairs = candidate_pairs[sort_order]

    flat_seqs = merged["flat_sequences"]
    m_offsets = merged["offsets"].astype(np.int64)
    m_lengths = merged["lengths"].astype(np.int32)
    sub_mat = BLOSUM62.astype(np.int8)

    all_lengths = np.concatenate([query_ds.lengths, db_ds.lengths])
    bw_default = max(20, int(np.percentile(all_lengths, 95) * 0.3))

    M = len(merged_pairs)
    print(f"Aligning {M} pairs, band_width={bw_default}\n", flush=True)

    # ── Benchmark 1: Numba SW (warmup + timed) ──────────────────────
    print("=" * 80)
    print("Numba SW alignment")
    print("=" * 80, flush=True)

    # Warmup
    _ = _batch_sw_compact_scored(
        merged_pairs[:1000], flat_seqs, m_offsets, m_lengths,
        nb_float32(0.1), nb_int32(bw_default), BLOSUM62)

    t0 = time.perf_counter()
    nb_sims, nb_scores, nb_mask = _batch_sw_compact_scored(
        merged_pairs, flat_seqs, m_offsets, m_lengths,
        nb_float32(0.1), nb_int32(bw_default), BLOSUM62)
    nb_time = time.perf_counter() - t0
    nb_passing = int(nb_mask.sum())
    print(f"  Time: {nb_time:.1f}s, passing: {nb_passing}/{M}", flush=True)

    # ── Benchmark 2: C SW (band=126) ────────────────────────────────
    print("\n" + "=" * 80)
    print("C/OpenMP SW alignment (band=126)")
    print("=" * 80, flush=True)

    t0 = time.perf_counter()
    c_sims, c_scores, c_mask = run_c_sw(
        merged_pairs, flat_seqs, m_offsets, m_lengths, bw_default, sub_mat)
    c_time = time.perf_counter() - t0
    c_passing = int(c_mask.sum())
    speedup = nb_time / c_time
    print(f"  Time: {c_time:.1f}s, passing: {c_passing}/{M}", flush=True)
    print(f"  Speedup: {speedup:.2f}x", flush=True)

    # Correctness: compare scores
    score_corr = np.corrcoef(nb_scores.astype(np.float64), c_scores.astype(np.float64))[0, 1]
    score_diff = np.abs(nb_scores - c_scores)
    print(f"  Score correlation: {score_corr:.6f}", flush=True)
    print(f"  Score exact match: {np.mean(score_diff == 0)*100:.1f}%", flush=True)
    print(f"  Score mean diff: {score_diff.mean():.2f}, max diff: {score_diff.max()}", flush=True)

    # ── Benchmark 3: C SW (band=50) ─────────────────────────────────
    print("\n" + "=" * 80)
    print("C/OpenMP SW alignment (band=50)")
    print("=" * 80, flush=True)

    t0 = time.perf_counter()
    c50_sims, c50_scores, c50_mask = run_c_sw(
        merged_pairs, flat_seqs, m_offsets, m_lengths, 50, sub_mat)
    c50_time = time.perf_counter() - t0
    speedup50 = nb_time / c50_time
    print(f"  Time: {c50_time:.1f}s", flush=True)
    print(f"  Speedup vs Numba bw=126: {speedup50:.2f}x", flush=True)

    # Compare scores: bw=50 vs bw=126
    s50_corr = np.corrcoef(c_scores.astype(np.float64), c50_scores.astype(np.float64))[0, 1]
    s50_diff = np.abs(c_scores - c50_scores)
    print(f"  Score corr (bw50 vs bw126): {s50_corr:.6f}", flush=True)
    print(f"  Score exact match: {np.mean(s50_diff == 0)*100:.1f}%", flush=True)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80, flush=True)
    print(f"  Numba SW (bw=126):   {nb_time:.1f}s  (baseline)")
    print(f"  C SW (bw=126):       {c_time:.1f}s  ({speedup:.2f}x)")
    print(f"  C SW (bw=50):        {c50_time:.1f}s  ({speedup50:.2f}x)")
    print(f"\n  Score correlation (C vs Numba): {score_corr:.6f}")
    print(f"  Score correlation (bw50 vs 126): {s50_corr:.6f}")
    print(f"\n  {M} pairs aligned, {nq} queries, {nd} database sequences")


if __name__ == "__main__":
    main()
