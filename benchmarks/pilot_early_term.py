#!/usr/bin/env python3
"""Two-stage scoring with early termination.

Run fast indices first (std k=3 + red k=5: ~11s). For each query,
check if the top candidate has a high combined count. Skip the slow
indices (red k=4 + sp 11011 + sp 110011: ~64s) for "easy" queries
that already have high-confidence candidates.

This is query-adaptive: fast for easy queries, thorough for hard ones.
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

# ── C extensions ─────────────────────────────────────────────────────
_BASE = Path(__file__).resolve().parent.parent / "clustkit" / "csrc"
_klib = ctypes.cdll.LoadLibrary(str(_BASE / "kmer_score.so"))
_klib.batch_score_queries_c.restype = None
_klib.batch_score_queries_c.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]
_klib.batch_score_queries_spaced_c.restype = None
_klib.batch_score_queries_spaced_c.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]
_swlib = ctypes.cdll.LoadLibrary(str(_BASE / "sw_align.so"))
_swlib.batch_sw_align_c.restype = None
_swlib.batch_sw_align_c.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_float,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]


def c_score(q, qo, ql, nq, k, alpha, ko, ke, kf, ft, nd, mc, topk):
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

def c_score_sp(q, qo, ql, nq, so, w, sp, alpha, ko, ke, kf, ft, nd, mc, topk):
    ot = np.empty((nq, mc), dtype=np.int32)
    oc = np.zeros(nq, dtype=np.int32)
    os_ = np.zeros((nq, mc), dtype=np.int32)
    _klib.batch_score_queries_spaced_c(
        q.ctypes.data, qo.ctypes.data, ql.ctypes.data, nq,
        so.ctypes.data, w, sp, alpha,
        ko.ctypes.data, ke.ctypes.data, kf.ctypes.data, ft, nd,
        2, 2, 10, mc, topk,
        ot.ctypes.data, oc.ctypes.data, os_.ctypes.data)
    return ot, oc, os_

def c_sw(mp, fs, off, lens, bw, dh=None):
    M = len(mp)
    pf = np.ascontiguousarray(mp.flatten())
    si = np.empty(M, dtype=np.float32); sc = np.empty(M, dtype=np.int32)
    mk = np.empty(M, dtype=np.uint8); sm = BLOSUM62.astype(np.int8)
    dh_ptr = dh.ctypes.data if dh is not None else None
    _swlib.batch_sw_align_c(pf.ctypes.data, fs.ctypes.data, off.ctypes.data,
        lens.ctypes.data, M, bw, sm.ctypes.data, 0.1, dh_ptr,
        si.ctypes.data, sc.ctypes.data, mk.ctypes.data)
    return si, sc, mk.astype(np.bool_)

def flatten_scores(ot, oc, os_, nq, nd):
    total = int(oc.sum())
    pk = np.empty(total, dtype=np.int64)
    sc = np.empty(total, dtype=np.int32)
    p = 0
    for qi in range(nq):
        nc = int(oc[qi])
        if nc > 0:
            pk[p:p+nc] = np.int64(qi) * nd + ot[qi, :nc].astype(np.int64)
            sc[p:p+nc] = os_[qi, :nc]
            p += nc
    return pk, sc

def fast_union_features(packed_list, scores_list, nd, nq):
    t0 = time.perf_counter()
    n_idx = len(packed_list)
    all_pk = np.concatenate(packed_list)
    all_sc = np.concatenate(scores_list).astype(np.float32)
    all_id = np.concatenate([np.full(len(p), i, dtype=np.int8) for i, p in enumerate(packed_list)])
    order = np.argsort(all_pk, kind='mergesort')
    all_pk = all_pk[order]; all_sc = all_sc[order]; all_id = all_id[order]
    changes = np.empty(len(all_pk), dtype=np.bool_)
    changes[0] = True; changes[1:] = all_pk[1:] != all_pk[:-1]
    upos = np.nonzero(changes)[0]
    n_unique = len(upos)
    pair_ids = np.cumsum(changes) - 1
    per_idx = np.zeros((n_unique, n_idx), dtype=np.float32)
    n_indices = np.zeros(n_unique, dtype=np.float32)
    max_score = np.zeros(n_unique, dtype=np.float32)
    np.add.at(n_indices, pair_ids, 1.0)
    np.maximum.at(max_score, pair_ids, all_sc)
    for i in range(len(all_pk)):
        per_idx[pair_ids[i], all_id[i]] = all_sc[i]
    sum_score = per_idx.sum(axis=1)
    unique_pk = all_pk[upos]
    pairs = np.empty((n_unique, 2), dtype=np.int32)
    pairs[:, 0] = (unique_pk // nd).astype(np.int32)
    pairs[:, 1] = (unique_pk % nd).astype(np.int32)
    q_lens_f = np.zeros(n_unique, dtype=np.float32)  # filled later
    t_lens_f = np.zeros(n_unique, dtype=np.float32)
    return pairs, per_idx, n_indices, max_score, sum_score, n_unique, time.perf_counter() - t0

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
    mc = 8000; topk = 200000; bw = 20
    k = int(db_index.params.get("kmer_index_k", db_index.params["kmer_size"]))
    ft = int(compute_freq_threshold(db_index.kmer_freqs, nd, 99.5))

    # Build indices
    rq = _remap_flat(q_flat, REDUCED_ALPHA, len(q_flat))
    rd = _remap_flat(db_ds.flat_sequences, REDUCED_ALPHA, len(db_ds.flat_sequences))
    r4o, r4e, r4f = build_kmer_index(rd, db_ds.offsets, db_ds.lengths, 4, "protein", alpha_size=REDUCED_ALPHA_SIZE)
    r5o, r5e, r5f = build_kmer_index(rd, db_ds.offsets, db_ds.lengths, 5, "protein", alpha_size=REDUCED_ALPHA_SIZE)
    s1d = build_kmer_index_spaced(rd, db_ds.offsets, db_ds.lengths, "11011", "protein", alpha_size=REDUCED_ALPHA_SIZE)
    s2d = build_kmer_index_spaced(rd, db_ds.offsets, db_ds.lengths, "110011", "protein", alpha_size=REDUCED_ALPHA_SIZE)
    r4ft = int(compute_freq_threshold(r4f, nd, 99.5))
    r5ft = int(compute_freq_threshold(r5f, nd, 99.5))
    s1ft = int(compute_freq_threshold(s1d[2], nd, 99.5))
    s2ft = int(compute_freq_threshold(s2d[2], nd, 99.5))

    print(f"Loaded {nq} queries, {nd} db\n", flush=True)

    # ── Baseline: all 5 indices ──────────────────────────────────────
    print("=" * 80)
    print("Baseline: All 5 indices (C scoring)")
    print("=" * 80, flush=True)

    t_all = time.perf_counter()
    ot1,oc1,os1,od1 = c_score(q_flat,q_off,q_lens,nq,k,20,
        db_index.kmer_offsets,db_index.kmer_entries,db_index.kmer_freqs,ft,nd,mc,topk)
    t_std = time.perf_counter()-t_all; print(f"  std k=3: {t_std:.1f}s", flush=True)

    t0=time.perf_counter()
    ot2,oc2,os2,_=c_score(rq,q_off,q_lens,nq,4,REDUCED_ALPHA_SIZE,r4o,r4e,r4f,r4ft,nd,mc,topk)
    t_r4=time.perf_counter()-t0; print(f"  red k=4: {t_r4:.1f}s", flush=True)

    t0=time.perf_counter()
    ot3,oc3,os3,_=c_score(rq,q_off,q_lens,nq,5,REDUCED_ALPHA_SIZE,r5o,r5e,r5f,r5ft,nd,mc,topk)
    t_r5=time.perf_counter()-t0; print(f"  red k=5: {t_r5:.1f}s", flush=True)

    t0=time.perf_counter()
    ot4,oc4,os4=c_score_sp(rq,q_off,q_lens,nq,s1d[3],int(s1d[4]),int(s1d[5]),
        REDUCED_ALPHA_SIZE,s1d[0],s1d[1],s1d[2],s1ft,nd,mc,topk)
    t_s1=time.perf_counter()-t0; print(f"  sp 11011: {t_s1:.1f}s", flush=True)

    t0=time.perf_counter()
    ot5,oc5,os5=c_score_sp(rq,q_off,q_lens,nq,s2d[3],int(s2d[4]),int(s2d[5]),
        REDUCED_ALPHA_SIZE,s2d[0],s2d[1],s2d[2],s2ft,nd,mc,topk)
    t_s2=time.perf_counter()-t0; print(f"  sp 110011: {t_s2:.1f}s", flush=True)

    t_all_score = t_std + t_r4 + t_r5 + t_s1 + t_s2
    t_fast = t_std + t_r5  # fast indices only
    t_slow = t_r4 + t_s1 + t_s2  # slow indices
    print(f"  Total: {t_all_score:.1f}s (fast={t_fast:.1f}s, slow={t_slow:.1f}s)\n", flush=True)

    # ── Analyze: which queries need slow indices? ────────────────────
    print("=" * 80)
    print("Analysis: per-query max score from fast indices (std k=3 + red k=5)")
    print("=" * 80, flush=True)

    # For each query, check max Phase B score from fast indices
    max_fast_score = np.zeros(nq, dtype=np.int32)
    for qi in range(nq):
        nc1 = int(oc1[qi])
        nc3 = int(oc3[qi])
        if nc1 > 0:
            max_fast_score[qi] = max(max_fast_score[qi], os1[qi, :nc1].max())
        if nc3 > 0:
            max_fast_score[qi] = max(max_fast_score[qi], os3[qi, :nc3].max())

    print(f"  Max fast score distribution:", flush=True)
    for thresh in [3, 5, 8, 10, 15, 20]:
        n_easy = int((max_fast_score >= thresh).sum())
        print(f"    score >= {thresh}: {n_easy}/{nq} queries ({100*n_easy/nq:.0f}%) would skip slow indices",
              flush=True)

    # ── Early termination test ───────────────────────────────────────
    print(f"\n" + "=" * 80)
    print("Early termination: skip slow indices for easy queries")
    print("=" * 80, flush=True)

    # Helper for full pipeline with given candidates
    merged = _merge_sequences_for_alignment(query_ds, db_ds)
    m_off = merged["offsets"].astype(np.int64)
    m_lens = merged["lengths"].astype(np.int32)

    def run_pipeline(label, pk_list, sc_list, t_score):
        t0 = time.perf_counter()
        pairs, per_idx, n_indices, max_sc, sum_sc, n_union, t_union = \
            fast_union_features(pk_list, sc_list, nd, nq)

        # Simple heuristic top-N (fast, no ML)
        N = 5000
        qi_arr = pairs[:, 0]
        heuristic = n_indices * 10.0 + max_sc
        sort_key = np.lexsort((-heuristic, qi_arr))
        sorted_qi = qi_arr[sort_key]
        keep = np.empty(min(nq * N, len(pairs)), dtype=np.int64)
        out_pos = 0; prev = -1; cnt = 0
        for i in range(len(sorted_qi)):
            qi = int(sorted_qi[i])
            if qi != prev: prev = qi; cnt = 0
            if cnt < N: keep[out_pos] = sort_key[i]; out_pos += 1; cnt += 1
        keep = keep[:out_pos]
        sel = pairs[keep]

        sel_m = _remap_pairs_to_merged(sel, merged["nq"])
        sk = np.minimum(sel_m[:, 0], sel_m[:, 1])
        so = np.argsort(sk, kind="mergesort")
        sel_m = sel_m[so]; sel = sel[so]

        t0_sw = time.perf_counter()
        sims, scores, mask = c_sw(sel_m, merged["flat_sequences"], m_off, m_lens, bw)
        t_sw = time.perf_counter() - t0_sw

        total = t_score + t_union + t_sw
        p = scores > 0
        hits = _collect_top_k_hits(sel[p], sims[p], nq, 500, query_ds, db_ds,
                                   passing_scores=scores[p].astype(np.float32))
        roc1 = evaluate_roc1(hits, metadata)
        print(f"  {label:55s} score={t_score:.0f}s union={t_union:.0f}s sw={t_sw:.0f}s "
              f"total={total:.0f}s ROC1={roc1:.4f} cands={n_union}", flush=True)
        return roc1, total

    # Flatten all index outputs
    all_pk_sc = []
    for ot,oc,os_ in [(ot1,oc1,os1),(ot2,oc2,os2),(ot3,oc3,os3),(ot4,oc4,os4),(ot5,oc5,os5)]:
        pk, sc = flatten_scores(ot, oc, os_, nq, nd)
        all_pk_sc.append((pk, sc))

    fast_pk_sc = [all_pk_sc[0], all_pk_sc[2]]  # std k=3 + red k=5
    slow_pk_sc = [all_pk_sc[1], all_pk_sc[3], all_pk_sc[4]]  # red k=4, sp1, sp2

    print(f"\n  {'Config':55s} {'Score':>6s} {'Union':>6s} {'SW':>5s} "
          f"{'Total':>6s} {'ROC1':>6s} {'Cands':>8s}", flush=True)
    print("  " + "-" * 95, flush=True)

    # Baseline: all 5 indices
    all_pks = [x[0] for x in all_pk_sc]
    all_scs = [x[1] for x in all_pk_sc]
    run_pipeline("All 5 indices (baseline)", all_pks, all_scs, t_all_score)

    # Fast only: std k=3 + red k=5
    fast_pks = [x[0] for x in fast_pk_sc]
    fast_scs = [x[1] for x in fast_pk_sc]
    run_pipeline("Fast only (std k=3 + red k=5)", fast_pks, fast_scs, t_fast)

    # Early termination at various thresholds
    for thresh in [3, 5, 8, 10, 15]:
        hard_queries = set(np.where(max_fast_score < thresh)[0].tolist())
        n_hard = len(hard_queries)
        n_easy = nq - n_hard

        # For hard queries: include slow index candidates
        # For easy queries: only fast index candidates
        combined_pks = list(fast_pks)  # always include fast
        combined_scs = list(fast_scs)

        # Add slow candidates only for hard queries
        for pk, sc in slow_pk_sc:
            # Filter to only hard-query pairs
            # pk = qi * nd + ti, so qi = pk // nd
            qi_vals = (pk // nd).astype(np.int32)
            hard_mask = np.array([int(qi) in hard_queries for qi in qi_vals])
            if hard_mask.sum() > 0:
                combined_pks.append(pk[hard_mask])
                combined_scs.append(sc[hard_mask])

        # Estimate scoring time: fast always + slow only for hard queries
        t_est = t_fast + t_slow * (n_hard / nq)

        label = f"Early term thresh={thresh} ({n_easy} easy, {n_hard} hard)"
        run_pipeline(label, combined_pks, combined_scs, t_est)

    print(f"\n  Reference: MMseqs2=0.7942 (14s)  DIAMOND=0.7963 (13s)")


if __name__ == "__main__":
    main()
