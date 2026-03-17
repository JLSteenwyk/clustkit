#!/usr/bin/env python3
"""Diagonal-hint alignment: use Phase B diagonal to center SW band.

Tests whether passing the best diagonal from Phase B to SW alignment:
1. Improves ROC1 (better alignment of off-diagonal homologs)
2. Allows narrower bands (20-30 instead of 50) without quality loss
"""

import ctypes
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.benchmark_scop_search import (
    classify_hit, compute_roc_n, read_fasta, write_fasta, RankedHit,
)

import numba; numba.set_num_threads(8)

from clustkit.database import load_database
from clustkit.io import read_sequences
from clustkit.kmer_index import (
    build_kmer_index, compute_freq_threshold, build_kmer_index_spaced,
    REDUCED_ALPHA, REDUCED_ALPHA_SIZE, _remap_flat,
)
from clustkit.pairwise import BLOSUM62
from clustkit.search import (
    _merge_sequences_for_alignment, _remap_pairs_to_merged, _collect_top_k_hits,
)

# ── C extensions (updated with diagonal hint support) ────────────────
_BASE = Path(__file__).resolve().parent.parent / "clustkit" / "csrc"

_klib = ctypes.cdll.LoadLibrary(str(_BASE / "kmer_score.so"))
_klib.batch_score_queries_c.restype = None
_klib.batch_score_queries_c.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # +out_diags
]

_swlib = ctypes.cdll.LoadLibrary(str(_BASE / "sw_align.so"))
_swlib.batch_sw_align_c.restype = None
_swlib.batch_sw_align_c.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_float,
    ctypes.c_void_p,  # diag_hints (can be NULL)
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]


def c_score_with_diags(q, qo, ql, nq, k, alpha, ko, ke, kf, ft, nd, mc, topk):
    ot = np.empty((nq, mc), dtype=np.int32)
    oc = np.zeros(nq, dtype=np.int32)
    os_ = np.zeros((nq, mc), dtype=np.int32)
    od = np.zeros((nq, mc), dtype=np.int32)
    _klib.batch_score_queries_c(
        q.ctypes.data, qo.ctypes.data, ql.ctypes.data, nq, k, alpha,
        ko.ctypes.data, ke.ctypes.data, kf.ctypes.data, ft, nd,
        2, 2, 10, mc, topk,
        ot.ctypes.data, oc.ctypes.data, os_.ctypes.data, od.ctypes.data)
    return ot, oc, os_, od


def c_sw(mp, fs, off, lens, bw, diag_hints=None):
    M = len(mp)
    pf = np.ascontiguousarray(mp.flatten())
    si = np.empty(M, dtype=np.float32)
    sc = np.empty(M, dtype=np.int32)
    mk = np.empty(M, dtype=np.uint8)
    sm = BLOSUM62.astype(np.int8)
    dh_ptr = diag_hints.ctypes.data if diag_hints is not None else None
    _swlib.batch_sw_align_c(
        pf.ctypes.data, fs.ctypes.data, off.ctypes.data, lens.ctypes.data,
        M, bw, sm.ctypes.data, 0.1,
        dh_ptr,
        si.ctypes.data, sc.ctypes.data, mk.ctypes.data)
    return si, sc, mk.astype(np.bool_)


def evaluate_roc1(hits_list, metadata):
    di = metadata["domain_info"]; fs = metadata["family_sizes_in_db"]
    qs = set(metadata["query_sids"])
    hbq = defaultdict(list)
    for qh in hits_list:
        for h in qh:
            hbq[h.query_id].append((h.target_id, h.score if h.score != 0 else h.identity))
    vals = []
    for qid in qs:
        qi = di.get(qid)
        if qi is None: continue
        tp = fs.get(str(qi["family"]), 1) - 1
        if tp <= 0: continue
        qh = sorted(hbq.get(qid, []), key=lambda x: -x[1])
        ranked = [RankedHit(target_id=t, score=s, label=classify_hit(qid, t, di))
                  for t, s in qh if classify_hit(qid, t, di) != "IGNORE"]
        vals.append(compute_roc_n(ranked, 1, tp))
    return float(np.mean(vals)) if vals else 0.0


def main():
    scop_dir = Path("benchmarks/data/scop_search_results")
    out_dir = Path("benchmarks/data/speed_sensitivity_results")
    with open(scop_dir / "metadata.json") as f:
        fm = json.load(f)
    random.seed(42)
    qsids = sorted(random.sample(fm["query_sids"], min(2000, len(fm["query_sids"]))))
    all_seqs = dict(read_fasta(scop_dir / "queries.fasta"))
    sub = [(s, all_seqs[s]) for s in qsids if s in all_seqs]
    qf = str(out_dir / "queries_subset.fasta")
    write_fasta(sub, qf)
    metadata = dict(fm); metadata["query_sids"] = qsids

    print("Loading database...", flush=True)
    db_index = load_database(out_dir / "clustkit_db")
    query_ds = read_sequences(qf, "protein")
    db_ds = db_index.dataset
    nq, nd = query_ds.num_sequences, db_ds.num_sequences
    q_flat = query_ds.flat_sequences
    q_off = query_ds.offsets.astype(np.int64)
    q_lens = query_ds.lengths.astype(np.int32)
    mc = 8000; topk = 200000
    k = int(db_index.params.get("kmer_index_k", db_index.params["kmer_size"]))
    ft = int(compute_freq_threshold(db_index.kmer_freqs, nd, 99.5))

    print(f"Loaded {nq} queries, {nd} db\n", flush=True)

    # ── Score with diagonal hints ────────────────────────────────────
    print("=" * 80)
    print("C scoring with diagonal hint output (std k=3)")
    print("=" * 80, flush=True)

    t0 = time.perf_counter()
    ot, oc, os_, od = c_score_with_diags(
        q_flat, q_off, q_lens, nq, k, 20,
        db_index.kmer_offsets, db_index.kmer_entries, db_index.kmer_freqs,
        ft, nd, mc, topk)
    t_score = time.perf_counter() - t0
    total_cands = int(oc.sum())
    print(f"  {total_cands} candidates in {t_score:.1f}s", flush=True)

    # Flatten
    pairs = np.empty((total_cands, 2), dtype=np.int32)
    diags = np.empty(total_cands, dtype=np.int32)
    p = 0
    for qi in range(nq):
        nc = int(oc[qi])
        if nc > 0:
            pairs[p:p+nc, 0] = qi
            pairs[p:p+nc, 1] = ot[qi, :nc]
            diags[p:p+nc] = od[qi, :nc]
            p += nc

    print(f"  Diagonal range: [{diags.min()}, {diags.max()}], median={np.median(diags):.0f}", flush=True)
    print(f"  Zero diagonals: {np.sum(diags == 0)} ({100*np.sum(diags==0)/len(diags):.1f}%)\n", flush=True)

    # ── Alignment comparison ─────────────────────────────────────────
    merged = _merge_sequences_for_alignment(query_ds, db_ds)
    m_off = merged["offsets"].astype(np.int64)
    m_lens = merged["lengths"].astype(np.int32)

    merged_pairs = _remap_pairs_to_merged(pairs, merged["nq"])
    sk = np.minimum(merged_pairs[:, 0], merged_pairs[:, 1])
    so = np.argsort(sk, kind="mergesort")
    merged_pairs = merged_pairs[so]
    pairs = pairs[so]
    diags = diags[so]

    print("=" * 80)
    print("SW alignment: no hint vs diagonal hint at various band widths")
    print("=" * 80, flush=True)

    configs = [
        ("No hint, bw=126", 126, None),
        ("No hint, bw=50", 50, None),
        ("No hint, bw=30", 30, None),
        ("No hint, bw=20", 20, None),
        ("Diag hint, bw=50", 50, diags),
        ("Diag hint, bw=30", 30, diags),
        ("Diag hint, bw=20", 20, diags),
        ("Diag hint, bw=15", 15, diags),
    ]

    print(f"\n  {'Config':30s} {'Time':>7s} {'ROC1':>7s} {'vs baseline':>12s}", flush=True)
    print("  " + "-" * 60, flush=True)

    baseline_roc1 = None
    for label, bw, hints in configs:
        dh = hints.astype(np.int32) if hints is not None else None

        t0 = time.perf_counter()
        sims, scores, mask = c_sw(merged_pairs, merged["flat_sequences"],
                                   m_off, m_lens, bw, dh)
        t_sw = time.perf_counter() - t0

        passing = scores > 0
        hits = _collect_top_k_hits(pairs[passing], sims[passing], nq, 500,
                                   query_ds, db_ds,
                                   passing_scores=scores[passing].astype(np.float32))
        roc1 = evaluate_roc1(hits, metadata)

        if baseline_roc1 is None:
            baseline_roc1 = roc1

        diff = roc1 - baseline_roc1
        print(f"  {label:30s} {t_sw:6.1f}s {roc1:7.4f} {diff:+12.4f}", flush=True)

    print(f"\n  Reference: MMseqs2=0.7942  DIAMOND=0.7963")
    print(f"  Candidates: {total_cands} from std k=3 only (single index for this test)")


if __name__ == "__main__":
    main()
