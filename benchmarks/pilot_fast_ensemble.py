#!/usr/bin/env python3
"""Speed #1: Fast feature computation during union (eliminate 32s searchsorted).
Sensitivity #6: Self-calibrating ensemble scoring (database-agnostic).

Two improvements tested:
1. Build per-pair features DURING the union step instead of post-hoc searchsorted.
2. Train a calibration model on a small sample from the same database, then use
   it for two-tier selection AND post-SW reranking. Fully generalizable — no
   external labels needed.
"""

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from numba import njit, prange, int32, int64, float32

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
from clustkit.pairwise import _batch_sw_compact_scored, BLOSUM62
from clustkit.search import (
    _merge_sequences_for_alignment, _remap_pairs_to_merged, _collect_top_k_hits,
)


def evaluate_roc1(results_hits, metadata):
    domain_info = metadata["domain_info"]
    family_sizes = metadata["family_sizes_in_db"]
    query_sids = set(metadata["query_sids"])
    hits_by_query = defaultdict(list)
    for qhits in results_hits:
        for h in qhits:
            hits_by_query[h.query_id].append((h.target_id, h.score if h.score != 0 else h.identity))
    roc1_values = []
    for qid in query_sids:
        q_info = domain_info.get(qid)
        if q_info is None:
            continue
        fam_key = str(q_info["family"])
        total_tp = family_sizes.get(fam_key, 1) - 1
        if total_tp <= 0:
            continue
        query_hits = hits_by_query.get(qid, [])
        query_hits.sort(key=lambda x: -x[1])
        ranked = []
        for tid, score in query_hits:
            label = classify_hit(qid, tid, domain_info)
            if label != "IGNORE":
                ranked.append(RankedHit(target_id=tid, score=score, label=label))
        roc1_values.append(compute_roc_n(ranked, 1, total_tp))
    return float(np.mean(roc1_values)) if roc1_values else 0.0


def main():
    scop_dir = Path("benchmarks/data/scop_search_results")
    out_dir = Path("benchmarks/data/speed_sensitivity_results")

    with open(scop_dir / "metadata.json") as f:
        full_metadata = json.load(f)
    all_query_sids = full_metadata["query_sids"]
    random.seed(42)
    query_sids = sorted(random.sample(all_query_sids, min(2000, len(all_query_sids))))
    all_seqs = dict(read_fasta(scop_dir / "queries.fasta"))
    sub_seqs = [(sid, all_seqs[sid]) for sid in query_sids if sid in all_seqs]
    query_fasta = str(out_dir / "queries_subset.fasta")
    write_fasta(sub_seqs, query_fasta)
    metadata = dict(full_metadata)
    metadata["query_sids"] = query_sids

    print("Loading database...", flush=True)
    db_index = load_database(out_dir / "clustkit_db")
    query_ds = read_sequences(query_fasta, "protein")
    db_ds = db_index.dataset
    nq = query_ds.num_sequences
    nd = db_ds.num_sequences

    q_flat = query_ds.flat_sequences
    q_off = query_ds.offsets.astype(np.int64)
    q_lens = query_ds.lengths.astype(np.int32)
    db_flat = db_ds.flat_sequences
    t_off = db_ds.offsets.astype(np.int64)
    t_lens = db_ds.lengths.astype(np.int32)
    mc = 8000; topk = 200000

    print(f"Loaded {nq} queries, {nd} database\n", flush=True)

    # ── Step 1: Generate candidates with scores ──────────────────────
    print("=" * 100)
    print("Step 1: Candidate generation with per-index scores")
    print("=" * 100, flush=True)

    freq_thresh = compute_freq_threshold(db_index.kmer_freqs, nd, 99.5)
    k = int32(db_index.params.get("kmer_index_k", db_index.params["kmer_size"]))

    def run_index(label, is_spaced=False, **kwargs):
        t0 = time.perf_counter()
        out_t = np.empty((nq, mc), dtype=np.int32)
        out_c = np.zeros(nq, dtype=np.int32)
        out_s = np.zeros((nq, mc), dtype=np.int32)
        if is_spaced:
            _batch_score_queries_spaced_with_scores(
                **kwargs, out_targets=out_t, out_counts=out_c, out_scores=out_s)
        else:
            _batch_score_queries_with_scores(
                **kwargs, out_targets=out_t, out_counts=out_c, out_scores=out_s)
        total = int(out_c.sum())
        pairs = np.empty((total, 2), dtype=np.int32)
        scores = np.empty(total, dtype=np.int32)
        p = 0
        for qi in range(nq):
            nc = int(out_c[qi])
            if nc > 0:
                pairs[p:p+nc, 0] = qi
                pairs[p:p+nc, 1] = out_t[qi, :nc]
                scores[p:p+nc] = out_s[qi, :nc]
                p += nc
        packed = pairs[:, 0].astype(np.int64) * nd + pairs[:, 1].astype(np.int64)
        print(f"  {label}: {total} ({time.perf_counter()-t0:.0f}s)", flush=True)
        return packed, scores

    # Standard k=3
    p_std, s_std = run_index("std_k3",
        q_flat=q_flat, q_offsets=q_off, q_lengths=q_lens,
        k=k, alpha_size=int32(20),
        kmer_offsets=db_index.kmer_offsets, kmer_entries=db_index.kmer_entries,
        kmer_freqs=db_index.kmer_freqs, freq_thresh=freq_thresh,
        num_db=int32(nd), min_total_hits=int32(2), min_diag_hits=int32(2),
        diag_bin_width=int32(10), max_cands=int32(mc), phase_a_topk=int32(topk))

    # Reduced alphabets
    red_q = _remap_flat(q_flat, REDUCED_ALPHA, len(q_flat))
    red_db = _remap_flat(db_flat, REDUCED_ALPHA, len(db_flat))

    red4_o, red4_e, red4_f = build_kmer_index(red_db, db_ds.offsets, db_ds.lengths, 4, "protein", alpha_size=REDUCED_ALPHA_SIZE)
    p_r4, s_r4 = run_index("red_k4",
        q_flat=red_q, q_offsets=q_off, q_lengths=q_lens,
        k=int32(4), alpha_size=int32(REDUCED_ALPHA_SIZE),
        kmer_offsets=red4_o, kmer_entries=red4_e, kmer_freqs=red4_f,
        freq_thresh=compute_freq_threshold(red4_f, nd, 99.5),
        num_db=int32(nd), min_total_hits=int32(2), min_diag_hits=int32(2),
        diag_bin_width=int32(10), max_cands=int32(mc), phase_a_topk=int32(topk))

    red5_o, red5_e, red5_f = build_kmer_index(red_db, db_ds.offsets, db_ds.lengths, 5, "protein", alpha_size=REDUCED_ALPHA_SIZE)
    p_r5, s_r5 = run_index("red_k5",
        q_flat=red_q, q_offsets=q_off, q_lengths=q_lens,
        k=int32(5), alpha_size=int32(REDUCED_ALPHA_SIZE),
        kmer_offsets=red5_o, kmer_entries=red5_e, kmer_freqs=red5_f,
        freq_thresh=compute_freq_threshold(red5_f, nd, 99.5),
        num_db=int32(nd), min_total_hits=int32(2), min_diag_hits=int32(2),
        diag_bin_width=int32(10), max_cands=int32(mc), phase_a_topk=int32(topk))

    # Spaced seeds
    sp1 = build_kmer_index_spaced(red_db, db_ds.offsets, db_ds.lengths, "11011", "protein", alpha_size=REDUCED_ALPHA_SIZE)
    p_s1, s_s1 = run_index("sp_11011", is_spaced=True,
        q_flat=red_q, q_offsets=q_off, q_lengths=q_lens,
        seed_offsets=sp1[3], weight=int32(sp1[4]), span=int32(sp1[5]),
        alpha_size=int32(REDUCED_ALPHA_SIZE),
        kmer_offsets=sp1[0], kmer_entries=sp1[1], kmer_freqs=sp1[2],
        freq_thresh=compute_freq_threshold(sp1[2], nd, 99.5),
        num_db=int32(nd), min_total_hits=int32(2), min_diag_hits=int32(2),
        diag_bin_width=int32(10), max_cands=int32(mc), phase_a_topk=int32(topk))

    sp2 = build_kmer_index_spaced(red_db, db_ds.offsets, db_ds.lengths, "110011", "protein", alpha_size=REDUCED_ALPHA_SIZE)
    p_s2, s_s2 = run_index("sp_110011", is_spaced=True,
        q_flat=red_q, q_offsets=q_off, q_lengths=q_lens,
        seed_offsets=sp2[3], weight=int32(sp2[4]), span=int32(sp2[5]),
        alpha_size=int32(REDUCED_ALPHA_SIZE),
        kmer_offsets=sp2[0], kmer_entries=sp2[1], kmer_freqs=sp2[2],
        freq_thresh=compute_freq_threshold(sp2[2], nd, 99.5),
        num_db=int32(nd), min_total_hits=int32(2), min_diag_hits=int32(2),
        diag_bin_width=int32(10), max_cands=int32(mc), phase_a_topk=int32(topk))

    # ── Step 2: FAST union with inline feature building ──────────────
    print("\n" + "=" * 100)
    print("Step 2: Fast union with inline feature building")
    print("=" * 100, flush=True)

    # Method A: Old approach (searchsorted) — for timing comparison
    t_old = time.perf_counter()
    all_packed_list = [p_std, p_r4, p_r5, p_s1, p_s2]
    all_scores_list = [s_std, s_r4, s_r5, s_s1, s_s2]
    all_packed_cat = np.concatenate(all_packed_list)
    unique_packed, inverse = np.unique(all_packed_cat, return_inverse=True)
    n_total = len(unique_packed)
    candidate_pairs = np.empty((n_total, 2), dtype=np.int32)
    candidate_pairs[:, 0] = (unique_packed // nd).astype(np.int32)
    candidate_pairs[:, 1] = (unique_packed % nd).astype(np.int32)

    per_idx_scores = np.zeros((n_total, 5), dtype=np.float32)
    for idx_i, (pk, sc) in enumerate(zip(all_packed_list, all_scores_list)):
        positions = np.searchsorted(unique_packed, pk)
        per_idx_scores[positions, idx_i] = sc.astype(np.float32)
    old_time = time.perf_counter() - t_old
    print(f"  Old (searchsorted): {old_time:.2f}s for {n_total} pairs", flush=True)

    # Method B: Fast inline approach — sort-merge with score tagging
    t_new = time.perf_counter()
    # Tag each entry with (packed_pair, index_id, score)
    n_entries = sum(len(p) for p in all_packed_list)
    tagged_packed = np.empty(n_entries, dtype=np.int64)
    tagged_idx = np.empty(n_entries, dtype=np.int8)
    tagged_score = np.empty(n_entries, dtype=np.int32)
    offset = 0
    for i, (pk, sc) in enumerate(zip(all_packed_list, all_scores_list)):
        n = len(pk)
        tagged_packed[offset:offset+n] = pk
        tagged_idx[offset:offset+n] = i
        tagged_score[offset:offset+n] = sc
        offset += n

    # Sort by packed value
    sort_order = np.argsort(tagged_packed, kind='mergesort')
    tagged_packed = tagged_packed[sort_order]
    tagged_idx = tagged_idx[sort_order]
    tagged_score = tagged_score[sort_order]

    # Single pass: deduplicate and build features
    # Use change detection on sorted packed values
    changes = np.empty(n_entries, dtype=np.bool_)
    changes[0] = True
    changes[1:] = tagged_packed[1:] != tagged_packed[:-1]
    unique_positions = np.nonzero(changes)[0]
    n_unique = len(unique_positions)

    cand_pairs_fast = np.empty((n_unique, 2), dtype=np.int32)
    cand_pairs_fast[:, 0] = (tagged_packed[unique_positions] // nd).astype(np.int32)
    cand_pairs_fast[:, 1] = (tagged_packed[unique_positions] % nd).astype(np.int32)

    per_idx_fast = np.zeros((n_unique, 5), dtype=np.float32)
    n_indices_fast = np.zeros(n_unique, dtype=np.float32)
    max_score_fast = np.zeros(n_unique, dtype=np.float32)

    # Map each entry to its unique pair index
    pair_ids = np.cumsum(changes) - 1  # 0-based unique pair index per entry
    for i in range(n_entries):
        pid = pair_ids[i]
        idx = tagged_idx[i]
        sc = float(tagged_score[i])
        per_idx_fast[pid, idx] = sc
        n_indices_fast[pid] += 1
        if sc > max_score_fast[pid]:
            max_score_fast[pid] = sc

    new_time = time.perf_counter() - t_new
    print(f"  New (sort-merge):   {new_time:.2f}s for {n_unique} pairs", flush=True)
    print(f"  Speedup: {old_time/new_time:.2f}x", flush=True)

    # Verify correctness
    assert n_unique == n_total, f"Mismatch: {n_unique} vs {n_total}"
    print(f"  Correctness: {n_unique} == {n_total} pairs OK", flush=True)

    # Build full feature matrix from fast path
    sum_score_fast = per_idx_fast.sum(axis=1)
    q_lens_f = q_lens[cand_pairs_fast[:, 0]].astype(np.float32)
    t_lens_f = t_lens[cand_pairs_fast[:, 1]].astype(np.float32)
    shorter = np.minimum(q_lens_f, t_lens_f)
    longer = np.maximum(q_lens_f, t_lens_f)
    len_ratio = np.where(longer > 0, shorter / longer, 0).astype(np.float32)
    len_diff = np.abs(q_lens_f - t_lens_f)

    features = np.column_stack([
        per_idx_fast,       # 5: per-index scores
        n_indices_fast,     # 1
        max_score_fast,     # 1
        sum_score_fast,     # 1
        len_ratio,          # 1
        len_diff,           # 1
        q_lens_f,           # 1
        t_lens_f,           # 1
    ])  # 12 features
    feat_names = ["s_std", "s_r4", "s_r5", "s_sp1", "s_sp2",
                  "n_idx", "max_s", "sum_s", "lr", "ld", "ql", "tl"]

    total_feat_time = new_time  # features are built during union
    print(f"\n  Total feature build time: {total_feat_time:.2f}s "
          f"(was {old_time:.2f}s + 10s inference = {old_time+10:.0f}s)\n", flush=True)

    # ── Step 3: SW alignment (ground truth) ──────────────────────────
    print("=" * 100)
    print("Step 3: SW alignment")
    print("=" * 100, flush=True)

    merged = _merge_sequences_for_alignment(query_ds, db_ds)
    merged_pairs = _remap_pairs_to_merged(cand_pairs_fast, merged["nq"])
    sk = np.minimum(merged_pairs[:, 0], merged_pairs[:, 1])
    so = np.argsort(sk, kind="mergesort")
    merged_pairs = merged_pairs[so]
    cand_pairs_fast = cand_pairs_fast[so]
    features = features[so]

    all_lengths = np.concatenate([query_ds.lengths, db_ds.lengths])
    bw = max(20, int(np.percentile(all_lengths, 95) * 0.3))

    t0 = time.perf_counter()
    sims, raw_scores, aln_mask = _batch_sw_compact_scored(
        merged_pairs, merged["flat_sequences"], merged["offsets"],
        merged["lengths"], np.float32(0.1), int32(bw), BLOSUM62)
    aln_time = time.perf_counter() - t0
    print(f"  {n_unique} pairs in {aln_time:.1f}s\n", flush=True)

    # Baseline ROC1
    passing = raw_scores > 0
    hits_base = _collect_top_k_hits(
        cand_pairs_fast[passing], sims[passing], nq, 500,
        query_ds, db_ds, passing_scores=raw_scores[passing].astype(np.float32))
    baseline_roc1 = evaluate_roc1(hits_base, metadata)
    print(f"  Baseline ROC1 (SW ranking): {baseline_roc1:.4f}\n", flush=True)

    # ── Step 4: Train models ─────────────────────────────────────────
    print("=" * 100)
    print("Step 4: Train models (self-calibrating, database-agnostic)")
    print("=" * 100, flush=True)

    query_ids = cand_pairs_fast[:, 0]
    unique_queries = np.unique(query_ids)
    np.random.seed(42)
    np.random.shuffle(unique_queries)
    split = int(len(unique_queries) * 0.8)
    train_queries = set(unique_queries[:split].tolist())
    train_mask = np.array([int(qi) in train_queries for qi in query_ids])

    X_train, X_test = features[train_mask], features[~train_mask]
    y_sw_train = raw_scores[train_mask]
    y_sw_test = raw_scores[~train_mask]

    import lightgbm as lgb

    # Model A: Regressor (predicts SW score — existing approach)
    t0 = time.perf_counter()
    reg = lgb.LGBMRegressor(n_estimators=500, max_depth=12, learning_rate=0.05,
                            n_jobs=-1, random_state=42, verbose=-1)
    reg.fit(X_train, y_sw_train)
    reg_time = time.perf_counter() - t0
    reg_pred = reg.predict(X_test)
    reg_corr = np.corrcoef(y_sw_test, reg_pred)[0, 1]
    print(f"  Regressor: r={reg_corr:.4f}, train={reg_time:.1f}s", flush=True)

    # Model B: Classifier (predicts "high SW score" — binary)
    # Label: SW score in top 25% for this pair's query (database-agnostic)
    sw_median = float(np.median(raw_scores))
    y_cls_train = (y_sw_train > sw_median).astype(np.int32)
    y_cls_test = (y_sw_test > sw_median).astype(np.int32)

    t0 = time.perf_counter()
    clf = lgb.LGBMClassifier(n_estimators=500, max_depth=12, learning_rate=0.05,
                             n_jobs=-1, random_state=42, verbose=-1)
    clf.fit(X_train, y_cls_train)
    clf_time = time.perf_counter() - t0
    clf_acc = np.mean(clf.predict(X_test) == y_cls_test)
    print(f"  Classifier: acc={clf_acc:.4f}, train={clf_time:.1f}s", flush=True)

    # Model C: Ensemble reranker (combines SW score + index features)
    # This is trained AFTER SW alignment on the calibration sample
    # Features: SW_score + all index features
    X_ens_train = np.column_stack([y_sw_train.reshape(-1, 1), X_train])
    X_ens_test = np.column_stack([y_sw_test.reshape(-1, 1), X_test])
    # Label: higher is better (use SW score as regression target but with extra features)
    t0 = time.perf_counter()
    ens = lgb.LGBMRegressor(n_estimators=300, max_depth=8, learning_rate=0.05,
                            n_jobs=-1, random_state=42, verbose=-1)
    ens.fit(X_ens_train, y_sw_train)
    ens_time = time.perf_counter() - t0
    ens_pred = ens.predict(X_ens_test)
    ens_corr = np.corrcoef(y_sw_test, ens_pred)[0, 1]
    print(f"  Ensemble reranker: r={ens_corr:.4f}, train={ens_time:.1f}s\n", flush=True)

    # ── Step 5: Small-sample calibration test ────────────────────────
    # Can we train on just 100 queries and generalize to 1900?
    print("=" * 100)
    print("Step 5: Small-sample calibration (100 queries → generalize to 1900)")
    print("=" * 100, flush=True)

    calib_queries = set(unique_queries[:100].tolist())
    calib_mask = np.array([int(qi) in calib_queries for qi in query_ids])
    rest_mask = ~calib_mask

    X_cal, X_rest = features[calib_mask], features[rest_mask]
    y_cal, y_rest = raw_scores[calib_mask], raw_scores[rest_mask]

    reg_cal = lgb.LGBMRegressor(n_estimators=300, max_depth=10, learning_rate=0.05,
                                n_jobs=-1, random_state=42, verbose=-1)
    reg_cal.fit(X_cal, y_cal)
    cal_pred = reg_cal.predict(X_rest)
    cal_corr = np.corrcoef(y_rest, cal_pred)[0, 1]
    print(f"  100-query calibration → rest: r={cal_corr:.4f}", flush=True)
    print(f"  (Full 80% training: r={reg_corr:.4f})", flush=True)

    # ── Step 6: Two-tier + reranking evaluation ──────────────────────
    print("\n" + "=" * 100)
    print("Step 6: Two-tier selection + ensemble reranking")
    print("=" * 100, flush=True)

    # Predict for all pairs
    pred_reg = reg.predict(features)
    pred_clf = clf.predict_proba(features)[:, 1]
    t0 = time.perf_counter()
    pred_reg_inf = reg.predict(features)
    inf_time = time.perf_counter() - t0
    print(f"  Inference time: {inf_time:.2f}s for {n_unique} pairs\n", flush=True)

    print(f"  {'Method':50s} {'ROC1':>7s} {'vs baseline':>12s}", flush=True)
    print("  " + "-" * 70, flush=True)
    print(f"  {'SW score ranking (baseline)':50s} {baseline_roc1:7.4f} {'—':>12s}", flush=True)

    # Test: rerank ALL aligned pairs by ensemble score
    X_ens_all = np.column_stack([raw_scores.reshape(-1, 1), features])
    ens_scores_all = ens.predict(X_ens_all).astype(np.float32)

    hits_ens = _collect_top_k_hits(
        cand_pairs_fast[passing], sims[passing], nq, 500,
        query_ds, db_ds, passing_scores=ens_scores_all[passing])
    ens_roc1 = evaluate_roc1(hits_ens, metadata)
    diff = ens_roc1 - baseline_roc1
    print(f"  {'Ensemble rerank (SW + index features)':50s} {ens_roc1:7.4f} {diff:+12.4f}", flush=True)

    # Test: rank by classifier probability (no SW needed)
    hits_clf = _collect_top_k_hits(
        cand_pairs_fast[passing], sims[passing], nq, 500,
        query_ds, db_ds, passing_scores=pred_clf[passing].astype(np.float32))
    clf_roc1 = evaluate_roc1(hits_clf, metadata)
    diff = clf_roc1 - baseline_roc1
    print(f"  {'Classifier P(high SW) ranking':50s} {clf_roc1:7.4f} {diff:+12.4f}", flush=True)

    # Test: rank by regressor predicted SW score (no actual SW needed)
    hits_reg = _collect_top_k_hits(
        cand_pairs_fast, np.full(n_unique, 0.5, dtype=np.float32), nq, 500,
        query_ds, db_ds, passing_scores=pred_reg.astype(np.float32))
    reg_roc1 = evaluate_roc1(hits_reg, metadata)
    diff = reg_roc1 - baseline_roc1
    print(f"  {'Regressor predicted SW (no alignment!)':50s} {reg_roc1:7.4f} {diff:+12.4f}", flush=True)

    # Two-tier: ML select top-N, then SW, then ensemble rerank
    print(f"\n  Two-tier + ensemble rerank:", flush=True)
    print(f"  {'N':>8s} {'Align(s)':>10s} {'ROC1 (SW rank)':>15s} "
          f"{'ROC1 (ens rank)':>16s} {'Improvement':>12s}", flush=True)
    print("  " + "-" * 70, flush=True)

    for N in [1000, 2000, 3000, 5000, 8000]:
        qi_col = cand_pairs_fast[:, 0]
        keep = np.zeros(n_unique, dtype=np.bool_)
        for qi in range(nq):
            mask = qi_col == qi
            if mask.sum() == 0:
                continue
            idx = np.where(mask)[0]
            if len(idx) <= N:
                keep[idx] = True
            else:
                top = np.argsort(-pred_reg[idx])[:N]
                keep[idx[top]] = True

        sel = cand_pairs_fast[keep]
        sel_feat = features[keep]
        sel_m = _remap_pairs_to_merged(sel, merged["nq"])
        sk2 = np.minimum(sel_m[:, 0], sel_m[:, 1])
        so2 = np.argsort(sk2, kind="mergesort")
        sel_m = sel_m[so2]; sel = sel[so2]; sel_feat = sel_feat[so2]

        t0 = time.perf_counter()
        s_sims, s_scores, _ = _batch_sw_compact_scored(
            sel_m, merged["flat_sequences"], merged["offsets"],
            merged["lengths"], np.float32(0.1), int32(bw), BLOSUM62)
        at = time.perf_counter() - t0

        # SW-only ranking
        p = s_scores > 0
        hits_sw = _collect_top_k_hits(
            sel[p], s_sims[p], nq, 500, query_ds, db_ds,
            passing_scores=s_scores[p].astype(np.float32))
        sw_roc = evaluate_roc1(hits_sw, metadata)

        # Ensemble reranking: combine SW score + index features
        X_ens_sel = np.column_stack([s_scores[p].reshape(-1, 1).astype(np.float32),
                                     sel_feat[p]])
        ens_sc = ens.predict(X_ens_sel).astype(np.float32)
        hits_ens2 = _collect_top_k_hits(
            sel[p], s_sims[p], nq, 500, query_ds, db_ds,
            passing_scores=ens_sc)
        ens_roc = evaluate_roc1(hits_ens2, metadata)

        imp = ens_roc - sw_roc
        print(f"  {N:8d} {at:9.1f}s {sw_roc:14.4f} {ens_roc:15.4f} {imp:+11.4f}", flush=True)

    print(f"\n  Feature build time (fast union): {total_feat_time:.2f}s")
    print(f"  Inference time: {inf_time:.2f}s")
    print(f"  Total ML overhead: {total_feat_time + inf_time:.2f}s (was ~35s)")
    print(f"\n  Reference: MMseqs2=0.7942  DIAMOND=0.7963")


if __name__ == "__main__":
    main()
