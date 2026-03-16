#!/usr/bin/env python3
"""Speed optimization tiers: test each improvement incrementally.

Tier 0: 5 indices + heuristic + C SW bw=126 (baseline with heuristic)
Tier 1: 5 indices + heuristic + C SW bw=50 (+ band reduction)
Tier 2: 3 indices + heuristic + C SW bw=126 (+ fewer indices)
Tier 3: 3 indices + heuristic + C SW bw=50 (both)
Tier 4-5: Tier 3 with larger N buffers (3000, 5000) for ROC1 comparison

Uses C extension for SW alignment. Heuristic = n_indices * 10 + max_score
replaces LightGBM (saves ~35s ML overhead).
"""

import ctypes
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from numba import int32 as nb_int32, float32 as nb_float32

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.benchmark_scop_search import (
    classify_hit, compute_roc_n, read_fasta, write_fasta, RankedHit,
)

import numba
numba.set_num_threads(8)

from clustkit.database import load_database
from clustkit.io import read_sequences
from clustkit.kmer_index import (
    build_kmer_index, compute_freq_threshold,
    _batch_score_queries_with_scores, _batch_score_queries_spaced_with_scores,
    build_kmer_index_spaced,
    REDUCED_ALPHA, REDUCED_ALPHA_SIZE, _remap_flat,
)
from clustkit.pairwise import BLOSUM62
from clustkit.search import (
    _merge_sequences_for_alignment, _remap_pairs_to_merged, _collect_top_k_hits,
)

# ── C SW extension ───────────────────────────────────────────────────
SW_LIB = ctypes.cdll.LoadLibrary(
    str(Path(__file__).resolve().parent.parent / "clustkit" / "csrc" / "sw_align.so"))
SW_LIB.batch_sw_align_c.restype = None
SW_LIB.batch_sw_align_c.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_float,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]

def c_sw(merged_pairs, flat_seqs, offsets, lengths, bw):
    M = len(merged_pairs)
    pf = np.ascontiguousarray(merged_pairs.flatten())
    sims = np.empty(M, dtype=np.float32)
    scores = np.empty(M, dtype=np.int32)
    mask = np.empty(M, dtype=np.uint8)
    sm = BLOSUM62.astype(np.int8)
    SW_LIB.batch_sw_align_c(pf.ctypes.data, flat_seqs.ctypes.data,
        offsets.ctypes.data, lengths.ctypes.data,
        M, bw, sm.ctypes.data, 0.1,
        sims.ctypes.data, scores.ctypes.data, mask.ctypes.data)
    return sims, scores, mask.astype(np.bool_)


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


def run_idx(label, nq, mc, nd, is_spaced=False, **kw):
    t0 = time.perf_counter()
    ot = np.empty((nq, mc), dtype=np.int32)
    oc = np.zeros(nq, dtype=np.int32)
    os = np.zeros((nq, mc), dtype=np.int32)
    if is_spaced:
        _batch_score_queries_spaced_with_scores(**kw, out_targets=ot, out_counts=oc, out_scores=os)
    else:
        _batch_score_queries_with_scores(**kw, out_targets=ot, out_counts=oc, out_scores=os)
    tot = int(oc.sum())
    pairs = np.empty((tot, 2), dtype=np.int32)
    scores = np.empty(tot, dtype=np.int32)
    p = 0
    for qi in range(nq):
        nc = int(oc[qi])
        if nc > 0:
            pairs[p:p+nc, 0] = qi; pairs[p:p+nc, 1] = ot[qi, :nc]
            scores[p:p+nc] = os[qi, :nc]; p += nc
    pk = pairs[:, 0].astype(np.int64) * nd + pairs[:, 1].astype(np.int64)
    el = time.perf_counter() - t0
    print(f"    {label}: {tot} ({el:.1f}s)", flush=True)
    return pk, scores, el


def heuristic_select(packed_list, scores_list, nd, nq, N):
    """Union + heuristic top-N selection. Returns (selected_pairs, n_union, time)."""
    t0 = time.perf_counter()
    all_pk = np.concatenate(packed_list)
    all_sc = np.concatenate(scores_list).astype(np.float32)

    order = np.argsort(all_pk, kind='mergesort')
    all_pk = all_pk[order]; all_sc = all_sc[order]

    changes = np.empty(len(all_pk), dtype=np.bool_)
    changes[0] = True; changes[1:] = all_pk[1:] != all_pk[:-1]
    upos = np.nonzero(changes)[0]
    n_unique = len(upos)
    pair_ids = np.cumsum(changes) - 1

    n_idx = np.zeros(n_unique, dtype=np.float32)
    max_sc = np.zeros(n_unique, dtype=np.float32)
    np.add.at(n_idx, pair_ids, 1.0)
    np.maximum.at(max_sc, pair_ids, all_sc)
    heuristic = n_idx * 10.0 + max_sc

    qi_arr = (all_pk[upos] // nd).astype(np.int32)
    ti_arr = (all_pk[upos] % nd).astype(np.int32)

    keep = np.zeros(n_unique, dtype=np.bool_)
    for qi in range(nq):
        m = qi_arr == qi
        if m.sum() == 0: continue
        idx = np.where(m)[0]
        if len(idx) <= N: keep[idx] = True
        else:
            top = np.argsort(-heuristic[idx])[:N]
            keep[idx[top]] = True

    result = np.empty((int(keep.sum()), 2), dtype=np.int32)
    result[:, 0] = qi_arr[keep]; result[:, 1] = ti_arr[keep]
    return result, n_unique, time.perf_counter() - t0


def run_tier(name, cfgs, nq, nd, mc, topk, N, q_flat, q_off, q_lens,
             db_ds, query_ds, metadata, bw):
    print(f"\n  {name}", flush=True)
    print(f"  {'='*70}", flush=True)

    pks, scs = [], []
    t_score = 0
    for c in cfgs:
        pk, sc, t = run_idx(nq=nq, mc=mc, nd=nd, **c)
        pks.append(pk); scs.append(sc); t_score += t

    sel, n_union, t_union = heuristic_select(pks, scs, nd, nq, N)
    print(f"    Union: {n_union} → {len(sel)} selected ({t_union:.1f}s)", flush=True)

    merged = _merge_sequences_for_alignment(query_ds, db_ds)
    sm = _remap_pairs_to_merged(sel, merged["nq"])
    sk = np.minimum(sm[:, 0], sm[:, 1])
    so = np.argsort(sk, kind="mergesort")
    sm = sm[so]; sel = sel[so]

    t0 = time.perf_counter()
    sims, scores, mask = c_sw(sm, merged["flat_sequences"],
        merged["offsets"].astype(np.int64), merged["lengths"].astype(np.int32), bw)
    t_sw = time.perf_counter() - t0

    p = scores > 0
    hits = _collect_top_k_hits(sel[p], sims[p], nq, 500, query_ds, db_ds,
                               passing_scores=scores[p].astype(np.float32))
    roc1 = evaluate_roc1(hits, metadata)

    total = t_score + t_union + t_sw
    print(f"    Score: {t_score:.1f}s  Union: {t_union:.1f}s  SW: {t_sw:.1f}s", flush=True)
    print(f"    TOTAL: {total:.1f}s  ROC1: {roc1:.4f}  Aligned: {len(sel)}", flush=True)
    return {"tier": name, "roc1": roc1, "total": total, "score": t_score,
            "union": t_union, "sw": t_sw, "aligned": len(sel), "n_union": n_union}


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
    k = nb_int32(db_index.params.get("kmer_index_k", db_index.params["kmer_size"]))
    ft = compute_freq_threshold(db_index.kmer_freqs, nd, 99.5)
    al = np.concatenate([query_ds.lengths, db_ds.lengths])
    bw = max(20, int(np.percentile(al, 95) * 0.3))
    print(f"Loaded {nq} queries, {nd} db, bw={bw}\n", flush=True)

    # Build indices
    print("Building indices...", flush=True)
    rq = _remap_flat(q_flat, REDUCED_ALPHA, len(q_flat))
    rd = _remap_flat(db_ds.flat_sequences, REDUCED_ALPHA, len(db_ds.flat_sequences))
    t0 = time.perf_counter()
    r4o, r4e, r4f = build_kmer_index(rd, db_ds.offsets, db_ds.lengths, 4, "protein", alpha_size=REDUCED_ALPHA_SIZE)
    r5o, r5e, r5f = build_kmer_index(rd, db_ds.offsets, db_ds.lengths, 5, "protein", alpha_size=REDUCED_ALPHA_SIZE)
    s1 = build_kmer_index_spaced(rd, db_ds.offsets, db_ds.lengths, "11011", "protein", alpha_size=REDUCED_ALPHA_SIZE)
    s2 = build_kmer_index_spaced(rd, db_ds.offsets, db_ds.lengths, "110011", "protein", alpha_size=REDUCED_ALPHA_SIZE)
    bt = time.perf_counter() - t0
    print(f"  Built in {bt:.1f}s (0s if pre-built)\n", flush=True)

    # Config dicts
    base = {"q_offsets": q_off, "q_lengths": q_lens, "num_db": nb_int32(nd),
            "min_total_hits": nb_int32(2), "min_diag_hits": nb_int32(2),
            "diag_bin_width": nb_int32(10), "max_cands": nb_int32(mc), "phase_a_topk": nb_int32(topk)}
    c_std = {**base, "label": "std_k3", "q_flat": q_flat, "k": k, "alpha_size": nb_int32(20),
             "kmer_offsets": db_index.kmer_offsets, "kmer_entries": db_index.kmer_entries,
             "kmer_freqs": db_index.kmer_freqs, "freq_thresh": ft}
    c_r4 = {**base, "label": "red_k4", "q_flat": rq, "k": nb_int32(4), "alpha_size": nb_int32(REDUCED_ALPHA_SIZE),
            "kmer_offsets": r4o, "kmer_entries": r4e, "kmer_freqs": r4f,
            "freq_thresh": compute_freq_threshold(r4f, nd, 99.5)}
    c_r5 = {**base, "label": "red_k5", "q_flat": rq, "k": nb_int32(5), "alpha_size": nb_int32(REDUCED_ALPHA_SIZE),
            "kmer_offsets": r5o, "kmer_entries": r5e, "kmer_freqs": r5f,
            "freq_thresh": compute_freq_threshold(r5f, nd, 99.5)}
    c_s1 = {**base, "label": "sp_11011", "is_spaced": True, "q_flat": rq,
            "seed_offsets": s1[3], "weight": nb_int32(s1[4]), "span": nb_int32(s1[5]),
            "alpha_size": nb_int32(REDUCED_ALPHA_SIZE),
            "kmer_offsets": s1[0], "kmer_entries": s1[1], "kmer_freqs": s1[2],
            "freq_thresh": compute_freq_threshold(s1[2], nd, 99.5)}
    c_s2 = {**base, "label": "sp_110011", "is_spaced": True, "q_flat": rq,
            "seed_offsets": s2[3], "weight": nb_int32(s2[4]), "span": nb_int32(s2[5]),
            "alpha_size": nb_int32(REDUCED_ALPHA_SIZE),
            "kmer_offsets": s2[0], "kmer_entries": s2[1], "kmer_freqs": s2[2],
            "freq_thresh": compute_freq_threshold(s2[2], nd, 99.5)}

    print("=" * 80)
    print("SPEED TIERS (heuristic selection, C SW alignment)")
    print("=" * 80, flush=True)

    results = []
    for name, cfgs, N, bwidth in [
        ("T0: 5idx + heuristic + C_SW bw=126", [c_std, c_r4, c_r5, c_s1, c_s2], 2000, bw),
        ("T1: 5idx + heuristic + C_SW bw=50",  [c_std, c_r4, c_r5, c_s1, c_s2], 2000, 50),
        ("T2: 3idx + heuristic + C_SW bw=126", [c_std, c_r5, c_s2], 2000, bw),
        ("T3: 3idx + heuristic + C_SW bw=50",  [c_std, c_r5, c_s2], 2000, 50),
        ("T4: 3idx + heuristic + C_SW bw=50 N=3000", [c_std, c_r5, c_s2], 3000, 50),
        ("T5: 3idx + heuristic + C_SW bw=50 N=5000", [c_std, c_r5, c_s2], 5000, 50),
    ]:
        r = run_tier(name, cfgs, nq, nd, mc, topk, N,
                     q_flat, q_off, q_lens, db_ds, query_ds, metadata, bwidth)
        results.append(r)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  {'Tier':50s} {'Score':>6s} {'Union':>6s} {'SW':>6s} {'Total':>7s} {'ROC1':>7s} {'vs MMseqs2':>10s}")
    print("  " + "-" * 95)
    for r in results:
        vs = r["roc1"] - 0.7942
        print(f"  {r['tier']:50s} {r['score']:5.1f}s {r['union']:5.1f}s {r['sw']:5.1f}s "
              f"{r['total']:6.1f}s {r['roc1']:7.4f} {vs:+10.4f}")
    print(f"\n  MMseqs2: 14s (ROC1=0.7942)  DIAMOND: 13s (ROC1=0.7963)")
    print(f"  Index build: {bt:.0f}s (not included — 0s if pre-built)")

    with open(out_dir / "speed_final_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
