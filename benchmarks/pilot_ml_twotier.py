#!/usr/bin/env python3
"""Two-tier ML-accelerated alignment.

Instead of SW-aligning all 31M candidates, use the RF model to predict
scores and select the top-N per query (with safety buffer above the
final top-K). Run actual SW only on this reduced set, then select
final top-K from real scores.

Tests various buffer sizes (N) to find the sweet spot between
speed and ROC1 preservation.
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
    _batch_score_queries, REDUCED_ALPHA, REDUCED_ALPHA_SIZE,
    _remap_flat, _build_query_kmer_sets, _compute_prefilter_features,
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
    cache_dir = out_dir / "ml_cache"

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
    print(f"Loaded {nq} queries, {nd} database sequences\n", flush=True)

    # ── Train RF model ───────────────────────────────────────────────
    print("=" * 100)
    print("Step 1: Train RandomForest model")
    print("=" * 100, flush=True)

    features = np.load(cache_dir / "ml_features.npy")
    sw_scores = np.load(cache_dir / "ml_sw_scores.npy")
    pairs_cached = np.load(cache_dir / "ml_pairs.npy")

    query_ids = pairs_cached[:, 0]
    unique_queries = np.unique(query_ids)
    np.random.seed(42)
    np.random.shuffle(unique_queries)
    split = int(len(unique_queries) * 0.8)
    train_queries = set(unique_queries[:split].tolist())
    train_mask = np.array([int(qi) in train_queries for qi in query_ids])

    from sklearn.ensemble import RandomForestRegressor
    t0 = time.perf_counter()
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=12, n_jobs=-1, random_state=42,
    )
    rf.fit(features[train_mask], sw_scores[train_mask])
    print(f"  Trained RF in {time.perf_counter()-t0:.1f}s\n", flush=True)

    # ── Generate candidates ──────────────────────────────────────────
    print("=" * 100)
    print("Step 2: Generate candidates (dual k=5, mc=8K)")
    print("=" * 100, flush=True)

    q_flat = query_ds.flat_sequences
    q_off = query_ds.offsets.astype(np.int64)
    q_lens = query_ds.lengths.astype(np.int32)
    db_flat = db_ds.flat_sequences
    mc = 8000

    k = int32(db_index.params.get("kmer_index_k", db_index.params["kmer_size"]))
    freq_thresh = compute_freq_threshold(db_index.kmer_freqs, nd, 99.5)
    out_targets = np.empty((nq, mc), dtype=np.int32)
    out_counts = np.zeros(nq, dtype=np.int32)

    t_cand_start = time.perf_counter()
    _batch_score_queries(
        q_flat, q_off, q_lens, k, int32(20),
        db_index.kmer_offsets, db_index.kmer_entries, db_index.kmer_freqs,
        freq_thresh, int32(nd), int32(2), int32(2), int32(10),
        int32(mc), int32(200000), out_targets, out_counts,
    )

    total_std = int(out_counts.sum())
    candidate_pairs = np.empty((total_std, 2), dtype=np.int32)
    pos = 0
    for qi in range(nq):
        nc = int(out_counts[qi])
        if nc > 0:
            candidate_pairs[pos:pos+nc, 0] = qi
            candidate_pairs[pos:pos+nc, 1] = out_targets[qi, :nc]
            pos += nc

    # Reduced k=5
    red_q_flat = _remap_flat(q_flat, REDUCED_ALPHA, len(q_flat))
    red_db_flat = _remap_flat(db_flat, REDUCED_ALPHA, len(db_flat))
    red_off, red_ent, red_freq = build_kmer_index(
        red_db_flat, db_ds.offsets, db_ds.lengths, 5, "protein",
        alpha_size=REDUCED_ALPHA_SIZE,
    )
    red_freq_thresh = compute_freq_threshold(red_freq, nd, 99.5)
    red_out_t = np.empty((nq, mc), dtype=np.int32)
    red_out_c = np.zeros(nq, dtype=np.int32)
    _batch_score_queries(
        red_q_flat, q_off, q_lens, int32(5), int32(REDUCED_ALPHA_SIZE),
        red_off, red_ent, red_freq, red_freq_thresh,
        int32(nd), int32(2), int32(2), int32(10),
        int32(mc), int32(200000), red_out_t, red_out_c,
    )

    # Union
    all_packed = [candidate_pairs[:, 0].astype(np.int64) * nd + candidate_pairs[:, 1].astype(np.int64)]
    red_total = int(red_out_c.sum())
    if red_total > 0:
        rp = np.empty((red_total, 2), dtype=np.int32)
        p = 0
        for qi in range(nq):
            nc = int(red_out_c[qi])
            if nc > 0:
                rp[p:p+nc, 0] = qi
                rp[p:p+nc, 1] = red_out_t[qi, :nc]
                p += nc
        all_packed.append(rp[:, 0].astype(np.int64) * nd + rp[:, 1].astype(np.int64))

    union = np.unique(np.concatenate(all_packed))
    candidate_pairs = np.empty((len(union), 2), dtype=np.int32)
    candidate_pairs[:, 0] = (union // nd).astype(np.int32)
    candidate_pairs[:, 1] = (union % nd).astype(np.int32)
    t_cand = time.perf_counter() - t_cand_start
    print(f"  {len(candidate_pairs)} candidates in {t_cand:.1f}s\n", flush=True)

    # ── Compute features + ML predictions ────────────────────────────
    print("=" * 100)
    print("Step 3: ML score prediction")
    print("=" * 100, flush=True)

    t0 = time.perf_counter()
    q_kmer_sets = _build_query_kmer_sets(q_flat, q_off, q_lens, int32(3), int32(20))
    feat = _compute_prefilter_features(
        candidate_pairs, q_kmer_sets, q_lens,
        db_flat, db_ds.offsets.astype(np.int64), db_ds.lengths.astype(np.int32),
        int32(3), int32(20),
    )
    feat_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    predicted_scores = rf.predict(feat)
    predict_time = time.perf_counter() - t0

    ml_overhead = feat_time + predict_time
    print(f"  Features: {feat_time:.1f}s, Prediction: {predict_time:.1f}s, "
          f"Total ML: {ml_overhead:.1f}s\n", flush=True)

    # ── Prepare alignment infrastructure ─────────────────────────────
    merged = _merge_sequences_for_alignment(query_ds, db_ds)
    all_lengths = np.concatenate([query_ds.lengths, db_ds.lengths])
    band_width = max(20, int(np.percentile(all_lengths, 95) * 0.3))

    # ── Baseline: align ALL candidates ───────────────────────────────
    print("=" * 100)
    print("Step 4: Baseline (align all candidates)")
    print("=" * 100, flush=True)

    merged_pairs_all = _remap_pairs_to_merged(candidate_pairs, merged["nq"])
    sort_key = np.minimum(merged_pairs_all[:, 0], merged_pairs_all[:, 1])
    sort_order = np.argsort(sort_key, kind="mergesort")
    merged_pairs_sorted = merged_pairs_all[sort_order]
    cand_pairs_sorted = candidate_pairs[sort_order]

    t0 = time.perf_counter()
    sims_all, scores_all, mask_all = _batch_sw_compact_scored(
        merged_pairs_sorted, merged["flat_sequences"], merged["offsets"],
        merged["lengths"], np.float32(0.1), int32(band_width), BLOSUM62,
    )
    baseline_aln_time = time.perf_counter() - t0

    passing = scores_all > 0
    hits_baseline = _collect_top_k_hits(
        cand_pairs_sorted[passing], sims_all[passing], nq, 500,
        query_ds, db_ds,
        passing_scores=scores_all[passing].astype(np.float32),
    )
    baseline_roc1 = evaluate_roc1(hits_baseline, metadata)
    baseline_total = t_cand + baseline_aln_time
    print(f"  Aligned {len(candidate_pairs)} pairs in {baseline_aln_time:.1f}s", flush=True)
    print(f"  Baseline ROC1: {baseline_roc1:.4f}, Total time: {baseline_total:.1f}s\n", flush=True)

    # ── Two-tier: ML select top-N, then SW on reduced set ────────────
    print("=" * 100)
    print("Step 5: Two-tier ML alignment (varying buffer size N)")
    print("=" * 100, flush=True)

    # For each query, select top-N by predicted score
    buffer_sizes = [500, 1000, 2000, 3000, 5000, 8000]
    results = []

    print(f"\n  {'Buffer N':>10s} {'Pairs aligned':>15s} {'Align time':>12s} "
          f"{'Total time':>12s} {'Speedup':>8s} {'ROC1':>7s} {'ROC1 loss':>10s}", flush=True)
    print("  " + "-" * 85, flush=True)

    for N in buffer_sizes:
        # Select top-N per query by predicted score
        # Build per-query index
        query_indices = candidate_pairs[:, 0]
        keep_mask = np.zeros(len(candidate_pairs), dtype=np.bool_)

        for qi in range(nq):
            qi_mask = query_indices == qi
            if qi_mask.sum() == 0:
                continue
            qi_preds = predicted_scores[qi_mask]
            qi_indices = np.where(qi_mask)[0]

            if len(qi_indices) <= N:
                keep_mask[qi_indices] = True
            else:
                top_n_idx = np.argsort(-qi_preds)[:N]
                keep_mask[qi_indices[top_n_idx]] = True

        selected_pairs = candidate_pairs[keep_mask]
        n_selected = len(selected_pairs)

        # Remap and sort
        merged_sel = _remap_pairs_to_merged(selected_pairs, merged["nq"])
        sort_key = np.minimum(merged_sel[:, 0], merged_sel[:, 1])
        sort_order = np.argsort(sort_key, kind="mergesort")
        merged_sel = merged_sel[sort_order]
        selected_pairs = selected_pairs[sort_order]

        # SW alignment on selected pairs only
        t0 = time.perf_counter()
        sims_sel, scores_sel, mask_sel = _batch_sw_compact_scored(
            merged_sel, merged["flat_sequences"], merged["offsets"],
            merged["lengths"], np.float32(0.1), int32(band_width), BLOSUM62,
        )
        sel_aln_time = time.perf_counter() - t0

        passing_sel = scores_sel > 0
        hits_sel = _collect_top_k_hits(
            selected_pairs[passing_sel], sims_sel[passing_sel], nq, 500,
            query_ds, db_ds,
            passing_scores=scores_sel[passing_sel].astype(np.float32),
        )
        sel_roc1 = evaluate_roc1(hits_sel, metadata)

        total_time = t_cand + ml_overhead + sel_aln_time
        speedup = baseline_total / total_time
        roc1_loss = baseline_roc1 - sel_roc1

        print(f"  {N:10d} {n_selected:15d} {sel_aln_time:11.1f}s "
              f"{total_time:11.1f}s {speedup:7.2f}x {sel_roc1:7.4f} {roc1_loss:+10.4f}",
              flush=True)

        results.append({
            "buffer_N": N, "pairs_aligned": n_selected,
            "align_time": sel_aln_time, "total_time": total_time,
            "speedup": speedup, "roc1": sel_roc1, "roc1_loss": roc1_loss,
        })

    print(f"\n  Baseline: ROC1={baseline_roc1:.4f}, total={baseline_total:.1f}s "
          f"(cand={t_cand:.1f}s + align={baseline_aln_time:.1f}s)", flush=True)
    print(f"  ML overhead: {ml_overhead:.1f}s (features={feat_time:.1f}s + predict={predict_time:.1f}s)", flush=True)
    print(f"\n  Reference: MMseqs2=0.7942 (14s)  DIAMOND=0.7963 (13s)")

    with open(out_dir / "ml_twotier_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
