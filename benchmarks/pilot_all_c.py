#!/usr/bin/env python3
"""All-C pipeline benchmark: C scoring for ALL indices + C SW alignment.

Measures the full pipeline time when every compute-intensive stage uses
the C/OpenMP extension. Also uses optimized numpy union (lexsort instead
of per-query loop).
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
    build_kmer_index, compute_freq_threshold,
    build_kmer_index_spaced,
    REDUCED_ALPHA, REDUCED_ALPHA_SIZE, _remap_flat,
)
from clustkit.pairwise import BLOSUM62
from clustkit.search import (
    _merge_sequences_for_alignment, _remap_pairs_to_merged, _collect_top_k_hits,
)

# ── Load C extensions ────────────────────────────────────────────────
_BASE = Path(__file__).resolve().parent.parent / "clustkit" / "csrc"

_kmer_lib = ctypes.cdll.LoadLibrary(str(_BASE / "kmer_score.so"))
_kmer_lib.batch_score_queries_c.restype = None
_kmer_lib.batch_score_queries_c.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]

_kmer_lib.batch_score_queries_spaced_c.restype = None
_kmer_lib.batch_score_queries_spaced_c.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]

_sw_lib = ctypes.cdll.LoadLibrary(str(_BASE / "sw_align.so"))
_sw_lib.batch_sw_align_c.restype = None
_sw_lib.batch_sw_align_c.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_float,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]


def c_score(q_flat, q_off, q_lens, nq, k, alpha, kmer_off, kmer_ent, kmer_freq, ft, nd, mc, topk):
    out_t = np.empty((nq, mc), dtype=np.int32)
    out_c = np.zeros(nq, dtype=np.int32)
    out_s = np.zeros((nq, mc), dtype=np.int32)
    _kmer_lib.batch_score_queries_c(
        q_flat.ctypes.data, q_off.ctypes.data, q_lens.ctypes.data,
        nq, k, alpha,
        kmer_off.ctypes.data, kmer_ent.ctypes.data, kmer_freq.ctypes.data,
        ft, nd, 2, 2, 10, mc, topk,
        out_t.ctypes.data, out_c.ctypes.data, out_s.ctypes.data)
    return out_t, out_c, out_s


def c_score_spaced(q_flat, q_off, q_lens, nq, seed_off, weight, span, alpha,
                   kmer_off, kmer_ent, kmer_freq, ft, nd, mc, topk):
    out_t = np.empty((nq, mc), dtype=np.int32)
    out_c = np.zeros(nq, dtype=np.int32)
    out_s = np.zeros((nq, mc), dtype=np.int32)
    _kmer_lib.batch_score_queries_spaced_c(
        q_flat.ctypes.data, q_off.ctypes.data, q_lens.ctypes.data,
        nq,
        seed_off.ctypes.data, weight, span, alpha,
        kmer_off.ctypes.data, kmer_ent.ctypes.data, kmer_freq.ctypes.data,
        ft, nd, 2, 2, 10, mc, topk,
        out_t.ctypes.data, out_c.ctypes.data, out_s.ctypes.data)
    return out_t, out_c, out_s


def c_sw(merged_pairs, flat_seqs, offsets, lengths, bw):
    M = len(merged_pairs)
    pf = np.ascontiguousarray(merged_pairs.flatten())
    sims = np.empty(M, dtype=np.float32)
    scores = np.empty(M, dtype=np.int32)
    mask = np.empty(M, dtype=np.uint8)
    sm = BLOSUM62.astype(np.int8)
    _sw_lib.batch_sw_align_c(
        pf.ctypes.data, flat_seqs.ctypes.data,
        offsets.ctypes.data, lengths.ctypes.data,
        M, bw, sm.ctypes.data, 0.1,
        sims.ctypes.data, scores.ctypes.data, mask.ctypes.data)
    return sims, scores, mask.astype(np.bool_)


def flatten_index(out_t, out_c, nq, nd):
    total = int(out_c.sum())
    packed = np.empty(total, dtype=np.int64)
    p = 0
    for qi in range(nq):
        nc = int(out_c[qi])
        if nc > 0:
            packed[p:p+nc] = np.int64(qi) * nd + out_t[qi, :nc].astype(np.int64)
            p += nc
    return packed


def fast_topn_select(packed_list, nd, nq, N):
    """Fast union + top-N per query using lexsort (no Python per-query loop)."""
    t0 = time.perf_counter()

    all_pk = np.concatenate(packed_list)
    order = np.argsort(all_pk, kind='mergesort')
    all_pk = all_pk[order]

    # Deduplicate
    changes = np.empty(len(all_pk), dtype=np.bool_)
    changes[0] = True
    changes[1:] = all_pk[1:] != all_pk[:-1]
    upos = np.nonzero(changes)[0]
    unique_pk = all_pk[upos]

    # Count n_indices per pair
    pair_ids = np.cumsum(changes) - 1
    n_idx = np.zeros(len(upos), dtype=np.int32)
    np.add.at(n_idx, pair_ids, 1)

    qi_arr = (unique_pk // nd).astype(np.int32)

    # Lexsort: primary by query_id ascending, secondary by n_indices descending
    sort_key = np.lexsort((-n_idx, qi_arr))
    sorted_qi = qi_arr[sort_key]
    sorted_pk = unique_pk[sort_key]

    # Single pass: keep first N per query
    keep_pk = np.empty(min(nq * N, len(sorted_pk)), dtype=np.int64)
    out_pos = 0
    prev_qi = -1
    count = 0
    for i in range(len(sorted_pk)):
        qi = int(sorted_qi[i])
        if qi != prev_qi:
            prev_qi = qi
            count = 0
        if count < N:
            keep_pk[out_pos] = sorted_pk[i]
            out_pos += 1
            count += 1
    keep_pk = keep_pk[:out_pos]

    result = np.empty((out_pos, 2), dtype=np.int32)
    result[:, 0] = (keep_pk // nd).astype(np.int32)
    result[:, 1] = (keep_pk % nd).astype(np.int32)

    elapsed = time.perf_counter() - t0
    return result, len(upos), elapsed


def evaluate_roc1(hits_list, metadata):
    domain_info = metadata["domain_info"]
    family_sizes = metadata["family_sizes_in_db"]
    query_sids = set(metadata["query_sids"])
    hbq = defaultdict(list)
    for qhits in hits_list:
        for h in qhits:
            hbq[h.query_id].append((h.target_id, h.score if h.score != 0 else h.identity))
    vals = []
    for qid in query_sids:
        qi = domain_info.get(qid)
        if qi is None: continue
        fk = str(qi["family"])
        tp = family_sizes.get(fk, 1) - 1
        if tp <= 0: continue
        qh = sorted(hbq.get(qid, []), key=lambda x: -x[1])
        ranked = [RankedHit(target_id=t, score=s, label=classify_hit(qid, t, domain_info))
                  for t, s in qh if classify_hit(qid, t, domain_info) != "IGNORE"]
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
    ft = int(compute_freq_threshold(db_index.kmer_freqs, nd, 99.5))
    k = int(db_index.params.get("kmer_index_k", db_index.params["kmer_size"]))
    al = np.concatenate([query_ds.lengths, db_ds.lengths])
    bw = max(20, int(np.percentile(al, 95) * 0.3))

    print(f"Loaded {nq} queries, {nd} db\n", flush=True)

    # Build reduced indices (one-time cost, would be pre-built)
    print("Building indices...", flush=True)
    rq = _remap_flat(q_flat, REDUCED_ALPHA, len(q_flat))
    rd = _remap_flat(db_ds.flat_sequences, REDUCED_ALPHA, len(db_ds.flat_sequences))
    t0 = time.perf_counter()
    r5o, r5e, r5f = build_kmer_index(rd, db_ds.offsets, db_ds.lengths, 5, "protein", alpha_size=REDUCED_ALPHA_SIZE)
    s2 = build_kmer_index_spaced(rd, db_ds.offsets, db_ds.lengths, "110011", "protein", alpha_size=REDUCED_ALPHA_SIZE)
    bt = time.perf_counter() - t0
    r5ft = int(compute_freq_threshold(r5f, nd, 99.5))
    s2ft = int(compute_freq_threshold(s2[2], nd, 99.5))
    print(f"  Built in {bt:.1f}s (0s if pre-built)\n", flush=True)

    # ── ALL-C PIPELINE ───────────────────────────────────────────────
    print("=" * 80)
    print("ALL-C PIPELINE: 3 indices + optimized union + C SW")
    print("=" * 80, flush=True)

    for N, bwidth, label in [
        (2000, bw, "N=2000, bw=126"),
        (2000, 50, "N=2000, bw=50"),
        (3000, 50, "N=3000, bw=50"),
        (5000, 50, "N=5000, bw=50"),
    ]:
        print(f"\n  Config: {label}", flush=True)
        t_total = time.perf_counter()

        # C scoring: std k=3
        t0 = time.perf_counter()
        ot1, oc1, os1 = c_score(q_flat, q_off, q_lens, nq, k, 20,
            db_index.kmer_offsets, db_index.kmer_entries, db_index.kmer_freqs,
            ft, nd, mc, topk)
        t_std = time.perf_counter() - t0

        # C scoring: reduced k=5
        t0 = time.perf_counter()
        ot2, oc2, os2 = c_score(rq, q_off, q_lens, nq, 5, REDUCED_ALPHA_SIZE,
            r5o, r5e, r5f, r5ft, nd, mc, topk)
        t_r5 = time.perf_counter() - t0

        # C scoring: spaced 110011
        t0 = time.perf_counter()
        ot3, oc3, os3 = c_score_spaced(rq, q_off, q_lens, nq,
            s2[3], int(s2[4]), int(s2[5]), REDUCED_ALPHA_SIZE,
            s2[0], s2[1], s2[2], s2ft, nd, mc, topk)
        t_sp = time.perf_counter() - t0

        t_score = t_std + t_r5 + t_sp
        print(f"    Scoring: std={t_std:.1f}s  red_k5={t_r5:.1f}s  sp_110011={t_sp:.1f}s  total={t_score:.1f}s", flush=True)

        # Union + top-N
        pk1 = flatten_index(ot1, oc1, nq, nd)
        pk2 = flatten_index(ot2, oc2, nq, nd)
        pk3 = flatten_index(ot3, oc3, nq, nd)
        selected, n_union, t_union = fast_topn_select([pk1, pk2, pk3], nd, nq, N)
        print(f"    Union: {n_union} → {len(selected)} ({t_union:.1f}s)", flush=True)

        # C SW alignment
        merged = _merge_sequences_for_alignment(query_ds, db_ds)
        sel_m = _remap_pairs_to_merged(selected, merged["nq"])
        sk = np.minimum(sel_m[:, 0], sel_m[:, 1])
        so = np.argsort(sk, kind="mergesort")
        sel_m = sel_m[so]; selected = selected[so]

        t0 = time.perf_counter()
        sims, scores, mask = c_sw(sel_m, merged["flat_sequences"],
            merged["offsets"].astype(np.int64), merged["lengths"].astype(np.int32), bwidth)
        t_sw = time.perf_counter() - t0

        total = time.perf_counter() - t_total
        p = scores > 0
        hits = _collect_top_k_hits(selected[p], sims[p], nq, 500, query_ds, db_ds,
                                   passing_scores=scores[p].astype(np.float32))
        roc1 = evaluate_roc1(hits, metadata)

        print(f"    SW: {t_sw:.1f}s  Total: {total:.1f}s  ROC1: {roc1:.4f}", flush=True)
        print(f"    Breakdown: score={t_score:.1f}  union={t_union:.1f}  sw={t_sw:.1f}  "
              f"other={total-t_score-t_union-t_sw:.1f}", flush=True)

    print(f"\n  Reference: MMseqs2=0.7942 (14s)  DIAMOND=0.7963 (13s)")
    print(f"  Index build: {bt:.0f}s (not included)")


if __name__ == "__main__":
    main()
