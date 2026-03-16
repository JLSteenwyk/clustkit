#!/usr/bin/env python3
"""All-C scoring + LightGBM two-tier + C SW alignment.

Combines the fastest components:
- C/OpenMP scoring for all indices (~32s)
- Fast sort-merge union with inline feature building (~14s)
- LightGBM prediction for top-N selection (~6-11s)
- C/OpenMP SW alignment (~5-12s)

Also tests smaller LightGBM models (fewer trees, shallower depth)
to find the speed/accuracy sweet spot.
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

# ── Load C extensions ────────────────────────────────────────────────
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
    os = np.zeros((nq, mc), dtype=np.int32)
    _klib.batch_score_queries_c(
        q.ctypes.data, qo.ctypes.data, ql.ctypes.data, nq, k, alpha,
        ko.ctypes.data, ke.ctypes.data, kf.ctypes.data, ft, nd,
        2, 2, 10, mc, topk,
        ot.ctypes.data, oc.ctypes.data, os.ctypes.data)
    return ot, oc, os

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
    mk = np.empty(M, dtype=np.uint8)
    sm = BLOSUM62.astype(np.int8)
    _swlib.batch_sw_align_c(pf.ctypes.data, fs.ctypes.data, off.ctypes.data,
        lens.ctypes.data, M, bw, sm.ctypes.data, 0.1,
        si.ctypes.data, sc.ctypes.data, mk.ctypes.data)
    return si, sc, mk.astype(np.bool_)


def flatten(ot, oc, nq, nd):
    total = int(oc.sum())
    pk = np.empty(total, dtype=np.int64)
    scores = np.empty(total, dtype=np.int32)
    p = 0
    for qi in range(nq):
        nc = int(oc[qi])
        if nc > 0:
            pk[p:p+nc] = np.int64(qi) * nd + ot[qi, :nc].astype(np.int64)
            scores[p:p+nc] = ot[qi, :nc]  # placeholder — we use index id below
            p += nc
    return pk


def fast_union_features(packed_list, scores_list, nd, nq):
    """Union + build 12 features inline. Returns (pairs, features, n_union)."""
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

    # Per-index scores
    per_idx = np.zeros((n_unique, n_idx), dtype=np.float32)
    n_indices = np.zeros(n_unique, dtype=np.float32)
    max_score = np.zeros(n_unique, dtype=np.float32)

    np.add.at(n_indices, pair_ids, 1.0)
    np.maximum.at(max_score, pair_ids, all_sc)
    for i in range(len(all_pk)):
        per_idx[pair_ids[i], all_id[i]] = all_sc[i]

    sum_score = per_idx.sum(axis=1)

    unique_pk = all_pk[upos]
    qi_arr = (unique_pk // nd).astype(np.int32)
    ti_arr = (unique_pk % nd).astype(np.int32)

    pairs = np.empty((n_unique, 2), dtype=np.int32)
    pairs[:, 0] = qi_arr; pairs[:, 1] = ti_arr

    elapsed = time.perf_counter() - t0
    return pairs, per_idx, n_indices, max_score, sum_score, n_unique, elapsed


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
    al = np.concatenate([query_ds.lengths, db_ds.lengths])
    bw = 50  # reduced band width

    print(f"Loaded {nq} queries, {nd} db, bw={bw}\n", flush=True)

    # Build indices
    rq = _remap_flat(q_flat, REDUCED_ALPHA, len(q_flat))
    rd = _remap_flat(db_ds.flat_sequences, REDUCED_ALPHA, len(db_ds.flat_sequences))
    r5o, r5e, r5f = build_kmer_index(rd, db_ds.offsets, db_ds.lengths, 5, "protein", alpha_size=REDUCED_ALPHA_SIZE)
    s2d = build_kmer_index_spaced(rd, db_ds.offsets, db_ds.lengths, "110011", "protein", alpha_size=REDUCED_ALPHA_SIZE)
    r5ft = int(compute_freq_threshold(r5f, nd, 99.5))
    s2ft = int(compute_freq_threshold(s2d[2], nd, 99.5))

    # ── C Scoring ────────────────────────────────────────────────────
    print("=" * 80)
    print("Step 1: C scoring (3 indices)")
    print("=" * 80, flush=True)

    t_score_start = time.perf_counter()
    ot1, oc1, os1 = c_score(q_flat, q_off, q_lens, nq, k, 20,
        db_index.kmer_offsets, db_index.kmer_entries, db_index.kmer_freqs, ft, nd, mc, topk)
    ot2, oc2, os2 = c_score(rq, q_off, q_lens, nq, 5, REDUCED_ALPHA_SIZE,
        r5o, r5e, r5f, r5ft, nd, mc, topk)
    ot3, oc3, os3 = c_score_sp(rq, q_off, q_lens, nq,
        s2d[3], int(s2d[4]), int(s2d[5]), REDUCED_ALPHA_SIZE,
        s2d[0], s2d[1], s2d[2], s2ft, nd, mc, topk)
    t_score = time.perf_counter() - t_score_start
    print(f"  {t_score:.1f}s\n", flush=True)

    # ── Fast union + features ────────────────────────────────────────
    print("=" * 80)
    print("Step 2: Fast union + feature building")
    print("=" * 80, flush=True)

    pk1 = flatten(ot1, oc1, nq, nd)
    pk2 = flatten(ot2, oc2, nq, nd)
    pk3 = flatten(ot3, oc3, nq, nd)

    # Get actual scores (Phase B scores from C)
    def get_scores(ot, oc, os_arr, nq):
        total = int(oc.sum())
        sc = np.empty(total, dtype=np.int32)
        p = 0
        for qi in range(nq):
            nc = int(oc[qi])
            if nc > 0:
                sc[p:p+nc] = os_arr[qi, :nc]
                p += nc
        return sc

    sc1 = get_scores(ot1, oc1, os1, nq)
    sc2 = get_scores(ot2, oc2, os2, nq)
    sc3 = get_scores(ot3, oc3, os3, nq)

    pairs, per_idx, n_indices, max_score, sum_score, n_union, t_union = \
        fast_union_features([pk1, pk2, pk3], [sc1, sc2, sc3], nd, nq)

    # Build full feature matrix (12 features, matching twotier3 format minus 2 unused indices)
    q_lens_f = q_lens[pairs[:, 0]].astype(np.float32)
    t_lens_f = db_ds.lengths[pairs[:, 1]].astype(np.float32)
    shorter = np.minimum(q_lens_f, t_lens_f)
    longer = np.maximum(q_lens_f, t_lens_f)
    len_ratio = np.where(longer > 0, shorter / longer, 0).astype(np.float32)
    len_diff = np.abs(q_lens_f - t_lens_f)

    # Pad per_idx to 5 columns (3 indices, 2 zeros for missing r4 and sp1)
    per_idx_5 = np.zeros((len(pairs), 5), dtype=np.float32)
    per_idx_5[:, 0] = per_idx[:, 0]  # std k3
    per_idx_5[:, 2] = per_idx[:, 1]  # red k5 → column 2
    per_idx_5[:, 4] = per_idx[:, 2]  # sp 110011 → column 4

    features = np.column_stack([
        per_idx_5, n_indices, max_score, sum_score,
        len_ratio, len_diff, q_lens_f, t_lens_f,
    ])  # 12 features

    print(f"  Union: {n_union} pairs, features: {features.shape}, {t_union:.1f}s\n", flush=True)

    # ── Train LightGBM models ────────────────────────────────────────
    print("=" * 80)
    print("Step 3: Train LightGBM (need ground truth SW scores first)")
    print("=" * 80, flush=True)

    # Load cached training data if available
    # Train on calibration sample: SW-align 500 queries worth of pairs
    print("  Running SW on calibration sample (500 queries)...", flush=True)
    merged = _merge_sequences_for_alignment(query_ds, db_ds)
    m_off = merged["offsets"].astype(np.int64)
    m_lens = merged["lengths"].astype(np.int32)

    # Select calibration queries (first 500)
    cal_mask = pairs[:, 0] < 500
    cal_pairs = pairs[cal_mask]
    cal_feat = features[cal_mask]

    cal_merged = _remap_pairs_to_merged(cal_pairs, merged["nq"])
    sk = np.minimum(cal_merged[:, 0], cal_merged[:, 1])
    so = np.argsort(sk, kind="mergesort")
    cal_merged = cal_merged[so]; cal_pairs = cal_pairs[so]; cal_feat = cal_feat[so]

    t0 = time.perf_counter()
    cal_sims, cal_scores, cal_mask_sw = c_sw(cal_merged, merged["flat_sequences"], m_off, m_lens, bw)
    cal_sw_time = time.perf_counter() - t0
    print(f"  Calibration SW: {len(cal_pairs)} pairs in {cal_sw_time:.1f}s", flush=True)

    # Train models on calibration data (12 features → SW score)
    import lightgbm as lgb

    # Split calibration into train/val by query
    cal_qi = cal_pairs[:, 0]
    cal_uq = np.unique(cal_qi)
    np.random.seed(42); np.random.shuffle(cal_uq)
    cal_train_q = set(cal_uq[:int(len(cal_uq)*0.8)].tolist())
    cal_tmask = np.array([int(q) in cal_train_q for q in cal_qi])

    models = {}
    for name, n_est, depth in [
        ("LGB-500-d12", 500, 12),
        ("LGB-200-d8", 200, 8),
        ("LGB-100-d6", 100, 6),
        ("LGB-50-d4", 50, 4),
    ]:
        t0 = time.perf_counter()
        m = lgb.LGBMRegressor(n_estimators=n_est, max_depth=depth,
                              learning_rate=0.05, n_jobs=-1, random_state=42, verbose=-1)
        m.fit(cal_feat[cal_tmask], cal_scores[cal_tmask])
        tt = time.perf_counter() - t0

        # Inference time on FULL feature set
        t0 = time.perf_counter()
        _ = m.predict(features)
        it = time.perf_counter() - t0

        pred = m.predict(cal_feat[~cal_tmask])
        r = np.corrcoef(cal_scores[~cal_tmask], pred)[0, 1]
        print(f"  {name}: train={tt:.1f}s  infer={it:.1f}s  r={r:.4f}", flush=True)
        models[name] = (m, it, r)

    # ── Test each model ──────────────────────────────────────────────
    print(f"\n" + "=" * 80)
    print("Step 4: Two-tier with each LightGBM model + C SW")
    print("=" * 80, flush=True)

    print(f"\n  {'Model':15s} {'Score':>6s} {'Union':>6s} {'LGB':>6s} {'SW':>6s} "
          f"{'Total':>7s} {'ROC1':>7s} {'vs MMseqs2':>10s}", flush=True)
    print("  " + "-" * 75, flush=True)

    for N in [2000, 3000]:
        for name, (model, inf_time, corr) in models.items():
            t_total = time.perf_counter()

            # LightGBM predict
            t0 = time.perf_counter()
            predicted = model.predict(features)
            t_lgb = time.perf_counter() - t0

            # Top-N per query by predicted score (vectorized with lexsort)
            qi_arr = pairs[:, 0]
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
            sk = np.minimum(sel_m[:, 0], sel_m[:, 1])
            so = np.argsort(sk, kind="mergesort")
            sel_m = sel_m[so]; sel = sel[so]

            # C SW alignment
            t0 = time.perf_counter()
            sims, scores, mask = c_sw(sel_m, merged["flat_sequences"], m_off, m_lens, bw)
            t_sw = time.perf_counter() - t0

            total = time.perf_counter() - t_total
            total_with_scoring = t_score + t_union + t_lgb + t_sw

            p = scores > 0
            hits = _collect_top_k_hits(sel[p], sims[p], nq, 500, query_ds, db_ds,
                                       passing_scores=scores[p].astype(np.float32))
            roc1 = evaluate_roc1(hits, metadata)
            vs = roc1 - 0.7942

            print(f"  {name+f' N={N}':15s} {t_score:5.1f}s {t_union:5.1f}s {t_lgb:5.1f}s {t_sw:5.1f}s "
                  f"{total_with_scoring:6.1f}s {roc1:7.4f} {vs:+10.4f}", flush=True)

    print(f"\n  Reference: MMseqs2=0.7942 (14s)  DIAMOND=0.7963 (13s)")


if __name__ == "__main__":
    main()
