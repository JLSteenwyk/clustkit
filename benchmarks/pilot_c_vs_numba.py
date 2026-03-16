#!/usr/bin/env python3
"""Benchmark: C/OpenMP extension vs Numba for Phase A+B scoring.

Compares wall-clock time and verifies correctness (same candidates produced).
"""

import ctypes
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from numba import int32, int64

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numba
numba.set_num_threads(8)

from clustkit.database import load_database
from clustkit.io import read_sequences
from clustkit.kmer_index import (
    _batch_score_queries, compute_freq_threshold,
    build_kmer_index, REDUCED_ALPHA, REDUCED_ALPHA_SIZE, _remap_flat,
)

# ── Load C extension ─────────────────────────────────────────────────

LIB_PATH = Path(__file__).resolve().parent.parent / "clustkit" / "csrc" / "kmer_score.so"
_clib = ctypes.cdll.LoadLibrary(str(LIB_PATH))

_clib.batch_score_queries_c.restype = None
_clib.batch_score_queries_c.argtypes = [
    ctypes.c_void_p,   # q_flat
    ctypes.c_void_p,   # q_offsets
    ctypes.c_void_p,   # q_lengths
    ctypes.c_int32,    # nq
    ctypes.c_int32,    # k
    ctypes.c_int32,    # alpha_size
    ctypes.c_void_p,   # kmer_offsets
    ctypes.c_void_p,   # kmer_entries
    ctypes.c_void_p,   # kmer_freqs
    ctypes.c_int32,    # freq_thresh
    ctypes.c_int32,    # num_db
    ctypes.c_int32,    # min_total_hits
    ctypes.c_int32,    # min_diag_hits
    ctypes.c_int32,    # diag_bin_width
    ctypes.c_int32,    # max_cands
    ctypes.c_int32,    # phase_a_topk
    ctypes.c_void_p,   # out_targets
    ctypes.c_void_p,   # out_counts
    ctypes.c_void_p,   # out_scores (can be NULL)
]


def run_c_scoring(q_flat, q_off, q_lens, nq, k, alpha_size,
                  kmer_offsets, kmer_entries, kmer_freqs,
                  freq_thresh, num_db, min_total_hits,
                  min_diag_hits, diag_bin_width, max_cands, phase_a_topk):
    out_targets = np.empty((nq, max_cands), dtype=np.int32)
    out_counts = np.zeros(nq, dtype=np.int32)
    out_scores = np.zeros((nq, max_cands), dtype=np.int32)

    _clib.batch_score_queries_c(
        q_flat.ctypes.data,
        q_off.ctypes.data,
        q_lens.ctypes.data,
        ctypes.c_int32(nq),
        ctypes.c_int32(k),
        ctypes.c_int32(alpha_size),
        kmer_offsets.ctypes.data,
        kmer_entries.ctypes.data,
        kmer_freqs.ctypes.data,
        ctypes.c_int32(freq_thresh),
        ctypes.c_int32(num_db),
        ctypes.c_int32(min_total_hits),
        ctypes.c_int32(min_diag_hits),
        ctypes.c_int32(diag_bin_width),
        ctypes.c_int32(max_cands),
        ctypes.c_int32(phase_a_topk),
        out_targets.ctypes.data,
        out_counts.ctypes.data,
        out_scores.ctypes.data,
    )
    return out_targets, out_counts, out_scores


def main():
    out_dir = Path("benchmarks/data/speed_sensitivity_results")

    print("Loading database...", flush=True)
    db_index = load_database(out_dir / "clustkit_db")

    scop_dir = Path("benchmarks/data/scop_search_results")
    with open(scop_dir / "metadata.json") as f:
        full_metadata = json.load(f)
    all_query_sids = full_metadata["query_sids"]
    random.seed(42)
    query_sids = sorted(random.sample(all_query_sids, min(2000, len(all_query_sids))))
    from benchmarks.benchmark_scop_search import read_fasta, write_fasta
    all_seqs = dict(read_fasta(scop_dir / "queries.fasta"))
    sub_seqs = [(sid, all_seqs[sid]) for sid in query_sids if sid in all_seqs]
    query_fasta = str(out_dir / "queries_subset.fasta")
    write_fasta(sub_seqs, query_fasta)
    query_ds = read_sequences(query_fasta, "protein")

    nq = query_ds.num_sequences
    nd = db_index.dataset.num_sequences
    q_flat = query_ds.flat_sequences
    q_off = query_ds.offsets.astype(np.int64)
    q_lens = query_ds.lengths.astype(np.int32)
    mc = 8000
    topk = 200000

    k_val = int(db_index.params.get("kmer_index_k", db_index.params["kmer_size"]))
    freq_thresh = int(compute_freq_threshold(db_index.kmer_freqs, nd, 99.5))

    print(f"Loaded {nq} queries, {nd} database sequences")
    print(f"k={k_val}, freq_thresh={freq_thresh}, mc={mc}, topk={topk}\n", flush=True)

    # ── Benchmark 1: Standard k=3 index ──────────────────────────────
    print("=" * 80)
    print("Standard k=3 index: Numba vs C")
    print("=" * 80, flush=True)

    # Warmup Numba
    out_t = np.empty((nq, mc), dtype=np.int32)
    out_c = np.zeros(nq, dtype=np.int32)
    _batch_score_queries(
        q_flat, q_off, q_lens, int32(k_val), int32(20),
        db_index.kmer_offsets, db_index.kmer_entries, db_index.kmer_freqs,
        int32(freq_thresh), int32(nd), int32(2), int32(2), int32(10),
        int32(mc), int32(topk), out_t, out_c,
    )
    numba_total = int(out_c.sum())
    print(f"  Numba warmup done: {numba_total} candidates", flush=True)

    # Numba timed run
    out_t_nb = np.empty((nq, mc), dtype=np.int32)
    out_c_nb = np.zeros(nq, dtype=np.int32)
    t0 = time.perf_counter()
    _batch_score_queries(
        q_flat, q_off, q_lens, int32(k_val), int32(20),
        db_index.kmer_offsets, db_index.kmer_entries, db_index.kmer_freqs,
        int32(freq_thresh), int32(nd), int32(2), int32(2), int32(10),
        int32(mc), int32(topk), out_t_nb, out_c_nb,
    )
    numba_time = time.perf_counter() - t0
    numba_total = int(out_c_nb.sum())
    print(f"  Numba: {numba_total} candidates in {numba_time:.2f}s", flush=True)

    # C timed run (warmup included — first call)
    t0 = time.perf_counter()
    out_t_c, out_c_c, out_s_c = run_c_scoring(
        q_flat, q_off, q_lens, nq, k_val, 20,
        db_index.kmer_offsets, db_index.kmer_entries, db_index.kmer_freqs,
        freq_thresh, nd, 2, 2, 10, mc, topk,
    )
    c_time = time.perf_counter() - t0
    c_total = int(out_c_c.sum())
    print(f"  C:     {c_total} candidates in {c_time:.2f}s", flush=True)

    speedup = numba_time / c_time if c_time > 0 else 0
    print(f"  Speedup: {speedup:.2f}x\n", flush=True)

    # Correctness check: compare candidate sets
    nb_set = set()
    for qi in range(nq):
        for j in range(int(out_c_nb[qi])):
            nb_set.add((qi, int(out_t_nb[qi, j])))
    c_set = set()
    for qi in range(nq):
        for j in range(int(out_c_c[qi])):
            c_set.add((qi, int(out_t_c[qi, j])))
    overlap = len(nb_set & c_set)
    print(f"  Correctness: Numba={len(nb_set)}, C={len(c_set)}, "
          f"overlap={overlap} ({100*overlap/max(len(nb_set),1):.1f}%)\n", flush=True)

    # ── Benchmark 2: Reduced k=5 index ───────────────────────────────
    print("=" * 80)
    print("Reduced k=5 index: Numba vs C")
    print("=" * 80, flush=True)

    red_q_flat = _remap_flat(q_flat, REDUCED_ALPHA, len(q_flat))
    red_db_flat = _remap_flat(db_index.dataset.flat_sequences, REDUCED_ALPHA,
                              len(db_index.dataset.flat_sequences))
    red_off, red_ent, red_freq = build_kmer_index(
        red_db_flat, db_index.dataset.offsets, db_index.dataset.lengths,
        5, "protein", alpha_size=REDUCED_ALPHA_SIZE,
    )
    red_ft = int(compute_freq_threshold(red_freq, nd, 99.5))

    # Numba
    out_t_nb2 = np.empty((nq, mc), dtype=np.int32)
    out_c_nb2 = np.zeros(nq, dtype=np.int32)
    t0 = time.perf_counter()
    _batch_score_queries(
        red_q_flat, q_off, q_lens, int32(5), int32(REDUCED_ALPHA_SIZE),
        red_off, red_ent, red_freq, int32(red_ft),
        int32(nd), int32(2), int32(2), int32(10),
        int32(mc), int32(topk), out_t_nb2, out_c_nb2,
    )
    nb2_time = time.perf_counter() - t0
    print(f"  Numba: {int(out_c_nb2.sum())} candidates in {nb2_time:.2f}s", flush=True)

    # C
    t0 = time.perf_counter()
    out_t_c2, out_c_c2, _ = run_c_scoring(
        red_q_flat, q_off, q_lens, nq, 5, REDUCED_ALPHA_SIZE,
        red_off, red_ent, red_freq, red_ft, nd, 2, 2, 10, mc, topk,
    )
    c2_time = time.perf_counter() - t0
    speedup2 = nb2_time / c2_time if c2_time > 0 else 0
    print(f"  C:     {int(out_c_c2.sum())} candidates in {c2_time:.2f}s", flush=True)
    print(f"  Speedup: {speedup2:.2f}x\n", flush=True)

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  Standard k=3:  Numba {numba_time:.1f}s → C {c_time:.1f}s  ({speedup:.2f}x)")
    print(f"  Reduced k=5:   Numba {nb2_time:.1f}s → C {c2_time:.1f}s  ({speedup2:.2f}x)")
    total_numba = numba_time + nb2_time
    total_c = c_time + c2_time
    print(f"  Combined:      Numba {total_numba:.1f}s → C {total_c:.1f}s  ({total_numba/total_c:.2f}x)")
    print(f"\n  If all 5 indices use C (estimated):")
    est_c_total = total_c / 2 * 5  # rough estimate for 5 indices
    est_numba_total = 184  # from our measurements
    print(f"    Numba ~{est_numba_total}s → C ~{est_c_total:.0f}s")


if __name__ == "__main__":
    main()
