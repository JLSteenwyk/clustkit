#!/usr/bin/env python3
"""All-C pipeline: 5 indices + N=5000 buffer + 1000-query calibration.

Three improvements over v7.3:
1. All 5 indices with C scoring (was 3)
2. N=5000 buffer (was 3000)
3. 1000-query calibration (was 500)
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
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
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
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]

def c_score(q, qo, ql, nq, k, alpha, ko, ke, kf, ft, nd, mc, topk):
    ot = np.empty((nq, mc), dtype=np.int32)
    oc = np.zeros(nq, dtype=np.int32)
    os_ = np.zeros((nq, mc), dtype=np.int32)
    _klib.batch_score_queries_c(
        q.ctypes.data, qo.ctypes.data, ql.ctypes.data, nq, k, alpha,
        ko.ctypes.data, ke.ctypes.data, kf.ctypes.data, ft, nd,
        2, 2, 10, mc, topk,
        ot.ctypes.data, oc.ctypes.data, os_.ctypes.data)
    return ot, oc, os_

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

def c_sw(mp, fs, off, lens, bw):
    M = len(mp)
    pf = np.ascontiguousarray(mp.flatten())
    si = np.empty(M, dtype=np.float32); sc = np.empty(M, dtype=np.int32)
    mk = np.empty(M, dtype=np.uint8); sm = BLOSUM62.astype(np.int8)
    _swlib.batch_sw_align_c(pf.ctypes.data, fs.ctypes.data, off.ctypes.data,
        lens.ctypes.data, M, bw, sm.ctypes.data, 0.1,
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

    q_lens_f = np.empty(n_unique, dtype=np.float32)
    t_lens_f = np.empty(n_unique, dtype=np.float32)
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
    mc = 8000; topk = 200000; bw = 50
    k = int(db_index.params.get("kmer_index_k", db_index.params["kmer_size"]))
    ft = int(compute_freq_threshold(db_index.kmer_freqs, nd, 99.5))

    print(f"Loaded {nq} queries, {nd} db\n", flush=True)

    # Build all 5 indices
    print("Building indices...", flush=True)
    rq = _remap_flat(q_flat, REDUCED_ALPHA, len(q_flat))
    rd = _remap_flat(db_ds.flat_sequences, REDUCED_ALPHA, len(db_ds.flat_sequences))
    t0 = time.perf_counter()
    r4o, r4e, r4f = build_kmer_index(rd, db_ds.offsets, db_ds.lengths, 4, "protein", alpha_size=REDUCED_ALPHA_SIZE)
    r5o, r5e, r5f = build_kmer_index(rd, db_ds.offsets, db_ds.lengths, 5, "protein", alpha_size=REDUCED_ALPHA_SIZE)
    s1d = build_kmer_index_spaced(rd, db_ds.offsets, db_ds.lengths, "11011", "protein", alpha_size=REDUCED_ALPHA_SIZE)
    s2d = build_kmer_index_spaced(rd, db_ds.offsets, db_ds.lengths, "110011", "protein", alpha_size=REDUCED_ALPHA_SIZE)
    bt = time.perf_counter() - t0
    print(f"  Built in {bt:.1f}s (0s if pre-built)\n", flush=True)

    # ── C Scoring: all 5 indices ─────────────────────────────────────
    print("=" * 80)
    print("Step 1: C scoring (ALL 5 indices)")
    print("=" * 80, flush=True)

    t_total_start = time.perf_counter()

    t0 = time.perf_counter()
    ot1, oc1, os1 = c_score(q_flat, q_off, q_lens, nq, k, 20,
        db_index.kmer_offsets, db_index.kmer_entries, db_index.kmer_freqs, ft, nd, mc, topk)
    t_std = time.perf_counter() - t0; print(f"  std k=3: {t_std:.1f}s", flush=True)

    r4ft = int(compute_freq_threshold(r4f, nd, 99.5))
    t0 = time.perf_counter()
    ot2, oc2, os2 = c_score(rq, q_off, q_lens, nq, 4, REDUCED_ALPHA_SIZE,
        r4o, r4e, r4f, r4ft, nd, mc, topk)
    t_r4 = time.perf_counter() - t0; print(f"  red k=4: {t_r4:.1f}s", flush=True)

    r5ft = int(compute_freq_threshold(r5f, nd, 99.5))
    t0 = time.perf_counter()
    ot3, oc3, os3 = c_score(rq, q_off, q_lens, nq, 5, REDUCED_ALPHA_SIZE,
        r5o, r5e, r5f, r5ft, nd, mc, topk)
    t_r5 = time.perf_counter() - t0; print(f"  red k=5: {t_r5:.1f}s", flush=True)

    s1ft = int(compute_freq_threshold(s1d[2], nd, 99.5))
    t0 = time.perf_counter()
    ot4, oc4, os4 = c_score_sp(rq, q_off, q_lens, nq,
        s1d[3], int(s1d[4]), int(s1d[5]), REDUCED_ALPHA_SIZE,
        s1d[0], s1d[1], s1d[2], s1ft, nd, mc, topk)
    t_s1 = time.perf_counter() - t0; print(f"  sp 11011: {t_s1:.1f}s", flush=True)

    s2ft = int(compute_freq_threshold(s2d[2], nd, 99.5))
    t0 = time.perf_counter()
    ot5, oc5, os5 = c_score_sp(rq, q_off, q_lens, nq,
        s2d[3], int(s2d[4]), int(s2d[5]), REDUCED_ALPHA_SIZE,
        s2d[0], s2d[1], s2d[2], s2ft, nd, mc, topk)
    t_s2 = time.perf_counter() - t0; print(f"  sp 110011: {t_s2:.1f}s", flush=True)

    t_score = t_std + t_r4 + t_r5 + t_s1 + t_s2
    print(f"  TOTAL scoring: {t_score:.1f}s\n", flush=True)

    # ── Union + features ─────────────────────────────────────────────
    print("=" * 80)
    print("Step 2: Union + feature building")
    print("=" * 80, flush=True)

    pk_list, sc_list = [], []
    for ot, oc, os_ in [(ot1,oc1,os1),(ot2,oc2,os2),(ot3,oc3,os3),(ot4,oc4,os4),(ot5,oc5,os5)]:
        pk, sc = flatten_scores(ot, oc, os_, nq, nd)
        pk_list.append(pk); sc_list.append(sc)

    pairs, per_idx, n_indices, max_score, sum_score, n_union, t_union = \
        fast_union_features(pk_list, sc_list, nd, nq)

    q_lens_f = q_lens[pairs[:, 0]].astype(np.float32)
    t_lens_f = db_ds.lengths[pairs[:, 1]].astype(np.float32)
    shorter = np.minimum(q_lens_f, t_lens_f)
    longer = np.maximum(q_lens_f, t_lens_f)
    len_ratio = np.where(longer > 0, shorter / longer, 0).astype(np.float32)
    len_diff = np.abs(q_lens_f - t_lens_f)

    features = np.column_stack([
        per_idx, n_indices, max_score, sum_score,
        len_ratio, len_diff, q_lens_f, t_lens_f,
    ])
    print(f"  Union: {n_union} pairs, features: {features.shape}, {t_union:.1f}s\n", flush=True)

    # ── Calibration: SW on 1000 queries ──────────────────────────────
    print("=" * 80)
    print("Step 3: Calibration (1000 queries)")
    print("=" * 80, flush=True)

    merged = _merge_sequences_for_alignment(query_ds, db_ds)
    m_off = merged["offsets"].astype(np.int64)
    m_lens = merged["lengths"].astype(np.int32)

    cal_mask = pairs[:, 0] < 1000
    cal_pairs = pairs[cal_mask]; cal_feat = features[cal_mask]
    cal_merged = _remap_pairs_to_merged(cal_pairs, merged["nq"])
    sk = np.minimum(cal_merged[:, 0], cal_merged[:, 1])
    so = np.argsort(sk, kind="mergesort")
    cal_merged = cal_merged[so]; cal_pairs = cal_pairs[so]; cal_feat = cal_feat[so]

    t0 = time.perf_counter()
    _, cal_scores, _ = c_sw(cal_merged, merged["flat_sequences"], m_off, m_lens, bw)
    t_cal_sw = time.perf_counter() - t0
    print(f"  Calibration SW: {len(cal_pairs)} pairs in {t_cal_sw:.1f}s", flush=True)

    import lightgbm as lgb
    cal_qi = cal_pairs[:, 0]
    cal_uq = np.unique(cal_qi)
    np.random.seed(42); np.random.shuffle(cal_uq)
    cal_tq = set(cal_uq[:int(len(cal_uq)*0.8)].tolist())
    cal_tm = np.array([int(q) in cal_tq for q in cal_qi])

    t0 = time.perf_counter()
    model = lgb.LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.05,
                               n_jobs=-1, random_state=42, verbose=-1)
    model.fit(cal_feat[cal_tm], cal_scores[cal_tm])
    t_train = time.perf_counter() - t0

    pred_val = model.predict(cal_feat[~cal_tm])
    r = np.corrcoef(cal_scores[~cal_tm], pred_val)[0, 1]
    print(f"  LGB-50-d4: train={t_train:.1f}s, r={r:.4f}\n", flush=True)

    # ── Two-tier: LGB select top-N, C SW align ──────────────────────
    print("=" * 80)
    print("Step 4: Two-tier selection + C SW alignment")
    print("=" * 80, flush=True)

    t0 = time.perf_counter()
    predicted = model.predict(features)
    t_lgb = time.perf_counter() - t0
    print(f"  LGB inference: {t_lgb:.1f}s\n", flush=True)

    qi_arr = pairs[:, 0]

    for N in [3000, 5000, 8000]:
        # Vectorized top-N using lexsort
        sort_key = np.lexsort((-predicted, qi_arr))
        sorted_qi = qi_arr[sort_key]

        keep_idx = np.empty(min(nq * N, len(pairs)), dtype=np.int64)
        out_pos = 0; prev_qi = -1; count = 0
        for i in range(len(sorted_qi)):
            qi = int(sorted_qi[i])
            if qi != prev_qi: prev_qi = qi; count = 0
            if count < N:
                keep_idx[out_pos] = sort_key[i]; out_pos += 1; count += 1
        keep_idx = keep_idx[:out_pos]

        sel = pairs[keep_idx]
        sel_m = _remap_pairs_to_merged(sel, merged["nq"])
        sk2 = np.minimum(sel_m[:, 0], sel_m[:, 1])
        so2 = np.argsort(sk2, kind="mergesort")
        sel_m = sel_m[so2]; sel = sel[so2]

        t0 = time.perf_counter()
        sims, scores, mask = c_sw(sel_m, merged["flat_sequences"], m_off, m_lens, bw)
        t_sw = time.perf_counter() - t0

        total = time.perf_counter() - t_total_start
        total_pipeline = t_score + t_union + t_lgb + t_sw

        p = scores > 0
        hits = _collect_top_k_hits(sel[p], sims[p], nq, 500, query_ds, db_ds,
                                   passing_scores=scores[p].astype(np.float32))
        roc1 = evaluate_roc1(hits, metadata)
        vs = roc1 - 0.7942

        print(f"  N={N}: score={t_score:.0f}s union={t_union:.0f}s lgb={t_lgb:.0f}s "
              f"sw={t_sw:.0f}s → total={total_pipeline:.0f}s  "
              f"ROC1={roc1:.4f} (vs MMseqs2: {vs:+.4f})", flush=True)

    print(f"\n  Reference: MMseqs2=0.7942 (14s)  DIAMOND=0.7963 (13s)")
    print(f"  Previous best speed: 3idx+LGB-50 N=3000 → 65s, ROC1=0.796")
    print(f"  Previous best sensitivity: 5idx full align → 115s, ROC1=0.818")


if __name__ == "__main__":
    main()
