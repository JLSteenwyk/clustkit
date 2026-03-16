#!/usr/bin/env python3
"""Fused multi-index scoring: all 5 indices in ONE C pass per query.

Eliminates the 37s union step and reduces OpenMP overhead by processing
all indices within a single parallel loop. Per-index scores are output
directly as ML features — no post-hoc feature computation needed.
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

_flib = ctypes.cdll.LoadLibrary(str(_BASE / "kmer_score_fused.so"))
_flib.batch_score_fused_c.restype = None
_flib.batch_score_fused_c.argtypes = [
    # q_flat_std, q_flat_red, q_offsets, q_lengths, nq, num_db, mc
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    # idx0: k, alpha, ko, ke, kf, ft
    ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
    # idx1
    ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
    # idx2
    ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
    # idx3: so, w, sp, alpha, ko, ke, kf, ft
    ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
    # idx4
    ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
    # outputs
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]

_swlib = ctypes.cdll.LoadLibrary(str(_BASE / "sw_align.so"))
_swlib.batch_sw_align_c.restype = None
_swlib.batch_sw_align_c.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_float,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]

def c_sw(mp, fs, off, lens, bw):
    M = len(mp)
    pf = np.ascontiguousarray(mp.flatten())
    si = np.empty(M, dtype=np.float32); sc = np.empty(M, dtype=np.int32)
    mk = np.empty(M, dtype=np.uint8); sm = BLOSUM62.astype(np.int8)
    _swlib.batch_sw_align_c(pf.ctypes.data, fs.ctypes.data, off.ctypes.data,
        lens.ctypes.data, M, bw, sm.ctypes.data, 0.1,
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
    mc = 8000; bw = 50
    k = int(db_index.params.get("kmer_index_k", db_index.params["kmer_size"]))
    ft0 = int(compute_freq_threshold(db_index.kmer_freqs, nd, 99.5))

    # Build indices
    print("Building indices...", flush=True)
    rq = _remap_flat(q_flat, REDUCED_ALPHA, len(q_flat))
    rd = _remap_flat(db_ds.flat_sequences, REDUCED_ALPHA, len(db_ds.flat_sequences))
    r4o, r4e, r4f = build_kmer_index(rd, db_ds.offsets, db_ds.lengths, 4, "protein", alpha_size=REDUCED_ALPHA_SIZE)
    r5o, r5e, r5f = build_kmer_index(rd, db_ds.offsets, db_ds.lengths, 5, "protein", alpha_size=REDUCED_ALPHA_SIZE)
    s1d = build_kmer_index_spaced(rd, db_ds.offsets, db_ds.lengths, "11011", "protein", alpha_size=REDUCED_ALPHA_SIZE)
    s2d = build_kmer_index_spaced(rd, db_ds.offsets, db_ds.lengths, "110011", "protein", alpha_size=REDUCED_ALPHA_SIZE)
    ft1 = int(compute_freq_threshold(r4f, nd, 99.5))
    ft2 = int(compute_freq_threshold(r5f, nd, 99.5))
    ft3 = int(compute_freq_threshold(s1d[2], nd, 99.5))
    ft4 = int(compute_freq_threshold(s2d[2], nd, 99.5))
    print("  Done\n", flush=True)

    # ── Fused scoring ────────────────────────────────────────────────
    print("=" * 80)
    print("Fused 5-index scoring (ONE pass per query)")
    print("=" * 80, flush=True)

    out_targets = np.empty((nq, mc), dtype=np.int32)
    out_counts = np.zeros(nq, dtype=np.int32)
    out_pidx = np.zeros((nq, mc * 5), dtype=np.float32)
    out_nidx = np.zeros((nq, mc), dtype=np.float32)
    out_maxs = np.zeros((nq, mc), dtype=np.float32)
    out_sums = np.zeros((nq, mc), dtype=np.float32)

    t0 = time.perf_counter()
    _flib.batch_score_fused_c(
        q_flat.ctypes.data, rq.ctypes.data, q_off.ctypes.data, q_lens.ctypes.data,
        nq, nd, mc,
        k, 20, db_index.kmer_offsets.ctypes.data, db_index.kmer_entries.ctypes.data,
        db_index.kmer_freqs.ctypes.data, ft0,
        4, REDUCED_ALPHA_SIZE, r4o.ctypes.data, r4e.ctypes.data, r4f.ctypes.data, ft1,
        5, REDUCED_ALPHA_SIZE, r5o.ctypes.data, r5e.ctypes.data, r5f.ctypes.data, ft2,
        s1d[3].ctypes.data, int(s1d[4]), int(s1d[5]), REDUCED_ALPHA_SIZE,
        s1d[0].ctypes.data, s1d[1].ctypes.data, s1d[2].ctypes.data, ft3,
        s2d[3].ctypes.data, int(s2d[4]), int(s2d[5]), REDUCED_ALPHA_SIZE,
        s2d[0].ctypes.data, s2d[1].ctypes.data, s2d[2].ctypes.data, ft4,
        out_targets.ctypes.data, out_counts.ctypes.data, out_pidx.ctypes.data,
        out_nidx.ctypes.data, out_maxs.ctypes.data, out_sums.ctypes.data,
    )
    t_fused = time.perf_counter() - t0
    total_cands = int(out_counts.sum())
    print(f"  Fused scoring: {total_cands} candidates in {t_fused:.1f}s", flush=True)
    print(f"  (Previous separate: 77s scoring + 37s union = 114s)\n", flush=True)

    # ── Build features + pairs (already computed by fused function!) ──
    print("=" * 80)
    print("Build features from fused output")
    print("=" * 80, flush=True)

    t0 = time.perf_counter()
    # Flatten outputs into candidate pairs + feature matrix
    pairs = np.empty((total_cands, 2), dtype=np.int32)
    features = np.empty((total_cands, 12), dtype=np.float32)
    p = 0
    for qi in range(nq):
        nc = int(out_counts[qi])
        if nc > 0:
            pairs[p:p+nc, 0] = qi
            pairs[p:p+nc, 1] = out_targets[qi, :nc]
            # Per-index scores (5 columns)
            for idx in range(5):
                features[p:p+nc, idx] = out_pidx[qi, idx*mc:idx*mc+nc]
            features[p:p+nc, 5] = out_nidx[qi, :nc]
            features[p:p+nc, 6] = out_maxs[qi, :nc]
            features[p:p+nc, 7] = out_sums[qi, :nc]
            # Length features
            ql = float(q_lens[qi])
            tls = db_ds.lengths[out_targets[qi, :nc]].astype(np.float32)
            shorter = np.minimum(ql, tls)
            longer = np.maximum(ql, tls)
            features[p:p+nc, 8] = np.where(longer > 0, shorter / longer, 0)
            features[p:p+nc, 9] = np.abs(ql - tls)
            features[p:p+nc, 10] = ql
            features[p:p+nc, 11] = tls
            p += nc
    t_feat = time.perf_counter() - t0
    print(f"  Features: {features.shape} in {t_feat:.1f}s\n", flush=True)

    # ── Calibration + LightGBM ───────────────────────────────────────
    print("=" * 80)
    print("Calibration (1000 queries) + LightGBM")
    print("=" * 80, flush=True)

    merged = _merge_sequences_for_alignment(query_ds, db_ds)
    m_off = merged["offsets"].astype(np.int64)
    m_lens = merged["lengths"].astype(np.int32)

    # Calibrate on first 1000 queries
    cal_mask = pairs[:, 0] < 1000
    cal_pairs = pairs[cal_mask]; cal_feat = features[cal_mask]
    cal_m = _remap_pairs_to_merged(cal_pairs, merged["nq"])
    sk = np.minimum(cal_m[:, 0], cal_m[:, 1])
    so = np.argsort(sk, kind="mergesort")
    cal_m = cal_m[so]; cal_pairs = cal_pairs[so]; cal_feat = cal_feat[so]

    t0 = time.perf_counter()
    _, cal_scores, _ = c_sw(cal_m, merged["flat_sequences"], m_off, m_lens, bw)
    t_cal = time.perf_counter() - t0
    print(f"  Calibration SW: {len(cal_pairs)} pairs in {t_cal:.1f}s", flush=True)

    import lightgbm as lgb
    cal_qi = cal_pairs[:, 0]
    cal_uq = np.unique(cal_qi)
    np.random.seed(42); np.random.shuffle(cal_uq)
    cal_tq = set(cal_uq[:int(len(cal_uq)*0.8)].tolist())
    cal_tm = np.array([int(q) in cal_tq for q in cal_qi])

    model = lgb.LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.05,
                               n_jobs=-1, random_state=42, verbose=-1)
    model.fit(cal_feat[cal_tm], cal_scores[cal_tm])
    pred_val = model.predict(cal_feat[~cal_tm])
    r = np.corrcoef(cal_scores[~cal_tm], pred_val)[0, 1]
    print(f"  LGB-50-d4: r={r:.4f}\n", flush=True)

    # ── Two-tier selection + C SW ────────────────────────────────────
    print("=" * 80)
    print("Two-tier: LGB select + C SW alignment")
    print("=" * 80, flush=True)

    t0 = time.perf_counter()
    predicted = model.predict(features)
    t_lgb = time.perf_counter() - t0
    print(f"  LGB inference: {t_lgb:.1f}s\n", flush=True)

    qi_arr = pairs[:, 0]

    for N in [3000, 5000, 8000]:
        t_start = time.perf_counter()

        sort_key = np.lexsort((-predicted, qi_arr))
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
        sk2 = np.minimum(sel_m[:, 0], sel_m[:, 1])
        so2 = np.argsort(sk2, kind="mergesort")
        sel_m = sel_m[so2]; sel = sel[so2]

        t0 = time.perf_counter()
        sims, scores, mask = c_sw(sel_m, merged["flat_sequences"], m_off, m_lens, bw)
        t_sw = time.perf_counter() - t0

        total = t_fused + t_feat + t_lgb + t_sw
        p2 = scores > 0
        hits = _collect_top_k_hits(sel[p2], sims[p2], nq, 500, query_ds, db_ds,
                                   passing_scores=scores[p2].astype(np.float32))
        roc1 = evaluate_roc1(hits, metadata)
        vs = roc1 - 0.7942

        print(f"  N={N}: fused={t_fused:.0f}s feat={t_feat:.0f}s lgb={t_lgb:.0f}s "
              f"sw={t_sw:.0f}s → total={total:.0f}s  "
              f"ROC1={roc1:.4f} (vs MMseqs2: {vs:+.4f})", flush=True)

    print(f"\n  Previous (separate 5idx): score=77s + union=37s = 114s")
    print(f"  Reference: MMseqs2=0.7942 (14s)  DIAMOND=0.7963 (13s)")


if __name__ == "__main__":
    main()
