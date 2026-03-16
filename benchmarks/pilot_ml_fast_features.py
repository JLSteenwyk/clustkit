#!/usr/bin/env python3
"""Zero-cost ML features: use Phase A/B scores directly, no recomputation.

Previous approach: compute shared k-mer counts per pair (25s for 65M pairs).
New approach: use the scores already extracted during candidate generation (<1s).

The Phase A/B scores ARE the most important features (importance 0.85 + 0.39).
Length features are trivial numpy lookups. This eliminates the 35s ML overhead.
"""

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from numba import int32, int64, float32

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

    print(f"Loaded {nq} queries, {nd} database sequences\n", flush=True)

    # ── Step 1: Candidate generation WITH scores ─────────────────────
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
        # Flatten
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
        elapsed = time.perf_counter() - t0
        print(f"  {label}: {total} candidates ({elapsed:.1f}s)", flush=True)
        return packed, pairs, scores

    t_total_cand = time.perf_counter()

    # Standard k=3
    p_std, pairs_std, scores_std = run_index("Standard k=3",
        q_flat=q_flat, q_offsets=q_off, q_lengths=q_lens,
        k=k, alpha_size=int32(20),
        kmer_offsets=db_index.kmer_offsets, kmer_entries=db_index.kmer_entries,
        kmer_freqs=db_index.kmer_freqs, freq_thresh=freq_thresh,
        num_db=int32(nd), min_total_hits=int32(2), min_diag_hits=int32(2),
        diag_bin_width=int32(10), max_cands=int32(mc), phase_a_topk=int32(topk))

    # Reduced k=4
    red_q_flat = _remap_flat(q_flat, REDUCED_ALPHA, len(q_flat))
    red_db_flat = _remap_flat(db_flat, REDUCED_ALPHA, len(db_flat))
    red4_off, red4_ent, red4_freq = build_kmer_index(
        red_db_flat, db_ds.offsets, db_ds.lengths, 4, "protein", alpha_size=REDUCED_ALPHA_SIZE)
    red4_ft = compute_freq_threshold(red4_freq, nd, 99.5)
    p_red4, pairs_red4, scores_red4 = run_index("Reduced k=4",
        q_flat=red_q_flat, q_offsets=q_off, q_lengths=q_lens,
        k=int32(4), alpha_size=int32(REDUCED_ALPHA_SIZE),
        kmer_offsets=red4_off, kmer_entries=red4_ent, kmer_freqs=red4_freq,
        freq_thresh=red4_ft, num_db=int32(nd), min_total_hits=int32(2),
        min_diag_hits=int32(2), diag_bin_width=int32(10),
        max_cands=int32(mc), phase_a_topk=int32(topk))

    # Reduced k=5
    red5_off, red5_ent, red5_freq = build_kmer_index(
        red_db_flat, db_ds.offsets, db_ds.lengths, 5, "protein", alpha_size=REDUCED_ALPHA_SIZE)
    red5_ft = compute_freq_threshold(red5_freq, nd, 99.5)
    p_red5, pairs_red5, scores_red5 = run_index("Reduced k=5",
        q_flat=red_q_flat, q_offsets=q_off, q_lengths=q_lens,
        k=int32(5), alpha_size=int32(REDUCED_ALPHA_SIZE),
        kmer_offsets=red5_off, kmer_entries=red5_ent, kmer_freqs=red5_freq,
        freq_thresh=red5_ft, num_db=int32(nd), min_total_hits=int32(2),
        min_diag_hits=int32(2), diag_bin_width=int32(10),
        max_cands=int32(mc), phase_a_topk=int32(topk))

    # Spaced seeds
    sp1_data = build_kmer_index_spaced(
        red_db_flat, db_ds.offsets, db_ds.lengths, "11011", "protein", alpha_size=REDUCED_ALPHA_SIZE)
    sp1_off, sp1_ent, sp1_freq, sp1_so, sp1_w, sp1_span = sp1_data
    sp1_ft = compute_freq_threshold(sp1_freq, nd, 99.5)
    p_sp1, pairs_sp1, scores_sp1 = run_index("Spaced 11011", is_spaced=True,
        q_flat=red_q_flat, q_offsets=q_off, q_lengths=q_lens,
        seed_offsets=sp1_so, weight=int32(sp1_w), span=int32(sp1_span),
        alpha_size=int32(REDUCED_ALPHA_SIZE),
        kmer_offsets=sp1_off, kmer_entries=sp1_ent, kmer_freqs=sp1_freq,
        freq_thresh=sp1_ft, num_db=int32(nd), min_total_hits=int32(2),
        min_diag_hits=int32(2), diag_bin_width=int32(10),
        max_cands=int32(mc), phase_a_topk=int32(topk))

    sp2_data = build_kmer_index_spaced(
        red_db_flat, db_ds.offsets, db_ds.lengths, "110011", "protein", alpha_size=REDUCED_ALPHA_SIZE)
    sp2_off, sp2_ent, sp2_freq, sp2_so, sp2_w, sp2_span = sp2_data
    sp2_ft = compute_freq_threshold(sp2_freq, nd, 99.5)
    p_sp2, pairs_sp2, scores_sp2 = run_index("Spaced 110011", is_spaced=True,
        q_flat=red_q_flat, q_offsets=q_off, q_lengths=q_lens,
        seed_offsets=sp2_so, weight=int32(sp2_w), span=int32(sp2_span),
        alpha_size=int32(REDUCED_ALPHA_SIZE),
        kmer_offsets=sp2_off, kmer_entries=sp2_ent, kmer_freqs=sp2_freq,
        freq_thresh=sp2_ft, num_db=int32(nd), min_total_hits=int32(2),
        min_diag_hits=int32(2), diag_bin_width=int32(10),
        max_cands=int32(mc), phase_a_topk=int32(topk))

    cand_gen_time = time.perf_counter() - t_total_cand

    # ── Step 2: Build features from pre-computed scores (ZERO COST) ──
    print("\n" + "=" * 100)
    print("Step 2: Build features from pre-computed scores")
    print("=" * 100, flush=True)

    t_feat_start = time.perf_counter()

    # Union with score tracking
    index_data = [
        ("std_k3", p_std, pairs_std, scores_std),
        ("red_k4", p_red4, pairs_red4, scores_red4),
        ("red_k5", p_red5, pairs_red5, scores_red5),
        ("sp_11011", p_sp1, pairs_sp1, scores_sp1),
        ("sp_110011", p_sp2, pairs_sp2, scores_sp2),
    ]

    all_packed = np.concatenate([d[1] for d in index_data])
    unique_packed, inverse = np.unique(all_packed, return_inverse=True)

    candidate_pairs = np.empty((len(unique_packed), 2), dtype=np.int32)
    candidate_pairs[:, 0] = (unique_packed // nd).astype(np.int32)
    candidate_pairs[:, 1] = (unique_packed % nd).astype(np.int32)
    n_total = len(candidate_pairs)

    # Map per-index scores to unified pair positions
    per_index_scores = np.zeros((n_total, 5), dtype=np.float32)
    for idx_i, (name, packed, pairs, scores) in enumerate(index_data):
        packed_vals = pairs[:, 0].astype(np.int64) * nd + pairs[:, 1].astype(np.int64)
        positions = np.searchsorted(unique_packed, packed_vals)
        per_index_scores[positions, idx_i] = scores.astype(np.float32)

    # Build feature matrix — ALL from pre-computed data, no per-pair scanning
    n_indices = (per_index_scores > 0).sum(axis=1).astype(np.float32)
    max_score = per_index_scores.max(axis=1)
    sum_score = per_index_scores.sum(axis=1)

    q_lens_f = q_lens[candidate_pairs[:, 0]].astype(np.float32)
    t_lens_f = t_lens[candidate_pairs[:, 1]].astype(np.float32)
    shorter = np.minimum(q_lens_f, t_lens_f)
    longer = np.maximum(q_lens_f, t_lens_f)
    len_ratio = np.where(longer > 0, shorter / longer, 0).astype(np.float32)
    len_diff = np.abs(q_lens_f - t_lens_f)

    features = np.column_stack([
        per_index_scores,  # 5: Phase A/B score per index
        n_indices,         # 1: how many indices found this pair
        max_score,         # 1
        sum_score,         # 1
        len_ratio,         # 1
        len_diff,          # 1
        q_lens_f,          # 1
        t_lens_f,          # 1
    ])  # 12 features total

    feat_time = time.perf_counter() - t_feat_start
    print(f"  {n_total} pairs, {features.shape[1]} features in {feat_time:.2f}s", flush=True)
    print(f"  (Previous per-pair k-mer scanning: ~25s)", flush=True)

    # ── Step 3: SW alignment (labels) ────────────────────────────────
    print("\n" + "=" * 100)
    print("Step 3: SW alignment (ground truth)")
    print("=" * 100, flush=True)

    merged = _merge_sequences_for_alignment(query_ds, db_ds)
    merged_pairs = _remap_pairs_to_merged(candidate_pairs, merged["nq"])
    sort_key = np.minimum(merged_pairs[:, 0], merged_pairs[:, 1])
    sort_order = np.argsort(sort_key, kind="mergesort")
    merged_pairs = merged_pairs[sort_order]
    candidate_pairs = candidate_pairs[sort_order]
    features = features[sort_order]

    all_lengths = np.concatenate([query_ds.lengths, db_ds.lengths])
    band_width = max(20, int(np.percentile(all_lengths, 95) * 0.3))

    t0 = time.perf_counter()
    sims, raw_scores, aln_mask = _batch_sw_compact_scored(
        merged_pairs, merged["flat_sequences"], merged["offsets"],
        merged["lengths"], np.float32(0.1), int32(band_width), BLOSUM62)
    aln_time = time.perf_counter() - t0
    print(f"  {n_total} pairs in {aln_time:.1f}s", flush=True)

    # ── Step 4: Train LightGBM ───────────────────────────────────────
    print("\n" + "=" * 100)
    print("Step 4: Train LightGBM on zero-cost features")
    print("=" * 100, flush=True)

    query_ids = candidate_pairs[:, 0]
    unique_queries = np.unique(query_ids)
    np.random.seed(42)
    np.random.shuffle(unique_queries)
    split = int(len(unique_queries) * 0.8)
    train_queries = set(unique_queries[:split].tolist())
    train_mask = np.array([int(qi) in train_queries for qi in query_ids])

    X_train, X_test = features[train_mask], features[~train_mask]
    y_train, y_test = raw_scores[train_mask], raw_scores[~train_mask]

    import lightgbm as lgb
    t0 = time.perf_counter()
    model = lgb.LGBMRegressor(
        n_estimators=500, max_depth=12, learning_rate=0.05,
        n_jobs=-1, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    y_pred = model.predict(X_test)
    corr = np.corrcoef(y_test, y_pred)[0, 1]
    mae = float(np.mean(np.abs(y_test - y_pred)))
    print(f"  Trained in {train_time:.1f}s", flush=True)
    print(f"  Pearson r: {corr:.4f}, MAE: {mae:.2f}", flush=True)
    print(f"  (Previous with per-pair k-mer features: r=0.9837, MAE=4.22)", flush=True)

    # Inference time
    t0 = time.perf_counter()
    predicted = model.predict(features)
    inf_time = time.perf_counter() - t0
    print(f"  Inference: {inf_time:.2f}s for {n_total} pairs", flush=True)

    total_ml_overhead = feat_time + inf_time
    print(f"\n  TOTAL ML overhead: {total_ml_overhead:.2f}s "
          f"(was ~35s with k-mer scanning)", flush=True)

    # ── Step 5: Two-tier evaluation ──────────────────────────────────
    print("\n" + "=" * 100)
    print("Step 5: Two-tier alignment with zero-cost features")
    print("=" * 100, flush=True)

    # Baseline ROC1
    passing = raw_scores > 0
    hits_base = _collect_top_k_hits(
        candidate_pairs[passing], sims[passing], nq, 500,
        query_ds, db_ds, passing_scores=raw_scores[passing].astype(np.float32))
    baseline_roc1 = evaluate_roc1(hits_base, metadata)
    print(f"  Baseline: ROC1={baseline_roc1:.4f}, align={aln_time:.1f}s\n", flush=True)

    print(f"  {'N':>8s} {'Pairs':>12s} {'Align(s)':>10s} {'Total(s)':>10s} "
          f"{'Speedup':>8s} {'ROC1':>7s}", flush=True)
    print("  " + "-" * 65, flush=True)

    for N in [500, 1000, 2000, 3000, 5000, 8000]:
        qi_col = candidate_pairs[:, 0]
        keep = np.zeros(n_total, dtype=np.bool_)
        for qi in range(nq):
            mask = qi_col == qi
            if mask.sum() == 0:
                continue
            idx = np.where(mask)[0]
            if len(idx) <= N:
                keep[idx] = True
            else:
                top = np.argsort(-predicted[idx])[:N]
                keep[idx[top]] = True

        sel = candidate_pairs[keep]
        sel_m = _remap_pairs_to_merged(sel, merged["nq"])
        sk = np.minimum(sel_m[:, 0], sel_m[:, 1])
        so = np.argsort(sk, kind="mergesort")
        sel_m = sel_m[so]; sel = sel[so]

        t0 = time.perf_counter()
        s_sims, s_scores, _ = _batch_sw_compact_scored(
            sel_m, merged["flat_sequences"], merged["offsets"],
            merged["lengths"], np.float32(0.1), int32(band_width), BLOSUM62)
        at = time.perf_counter() - t0

        p = s_scores > 0
        hits = _collect_top_k_hits(
            sel[p], s_sims[p], nq, 500, query_ds, db_ds,
            passing_scores=s_scores[p].astype(np.float32))
        roc1 = evaluate_roc1(hits, metadata)

        total = cand_gen_time + total_ml_overhead + at
        baseline_total = cand_gen_time + aln_time
        speedup = baseline_total / total

        print(f"  {N:8d} {len(sel):12d} {at:9.1f}s {total:9.1f}s "
              f"{speedup:7.2f}x {roc1:7.4f}", flush=True)

    print(f"\n  Timing breakdown:")
    print(f"    Candidate gen:  {cand_gen_time:.1f}s")
    print(f"    ML features:    {feat_time:.2f}s (was ~25s)")
    print(f"    ML inference:   {inf_time:.2f}s")
    print(f"    Alignment (N=2000): ~126s")
    print(f"  Reference: MMseqs2=0.7942 (14s)  DIAMOND=0.7963 (13s)")


if __name__ == "__main__":
    main()
