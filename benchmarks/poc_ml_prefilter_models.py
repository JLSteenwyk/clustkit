#!/usr/bin/env python3
"""Compare multiple ML models for the SW alignment prefilter.

Trains and evaluates several model types on the same feature set,
measuring prediction accuracy, inference speed, and simulated
prefilter performance (ROC1 at various filtering rates with MAE margin).

Models tested:
  - HistGradientBoostingRegressor (baseline from POC)
  - RandomForestRegressor
  - Ridge regression (linear baseline)
  - MLPRegressor (neural network)
  - Simple shared_k3 threshold (non-ML baseline)

Features are saved to disk after first generation so subsequent
model experiments don't require re-running the expensive pipeline.
"""

import json
import os
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
    search_kmer_index, build_kmer_index, compute_freq_threshold,
    _batch_score_queries, REDUCED_ALPHA, REDUCED_ALPHA_SIZE,
    _remap_flat, _batch_score_queries_spaced, build_kmer_index_spaced,
    _build_query_kmer_sets, _compute_prefilter_features,
)
from clustkit.pairwise import _batch_sw_compact_scored, BLOSUM62
from clustkit.search import _merge_sequences_for_alignment, _remap_pairs_to_merged


# ──────────────────────────────────────────────────────────────────────
# ROC1 evaluation helpers
# ──────────────────────────────────────────────────────────────────────

def evaluate_roc1(hits_by_query, metadata):
    domain_info = metadata["domain_info"]
    family_sizes = metadata["family_sizes_in_db"]
    query_sids = set(metadata["query_sids"])
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


def collect_hits(candidate_pairs, sw_scores, query_ds, db_ds, nq, top_k=500):
    hits_by_query = defaultdict(list)
    for idx in range(len(candidate_pairs)):
        qi = int(candidate_pairs[idx, 0])
        ti = int(candidate_pairs[idx, 1])
        score = float(sw_scores[idx])
        qid = query_ds.ids[qi]
        tid = db_ds.ids[ti]
        hits_by_query[qid].append((tid, score))
    for qid in hits_by_query:
        hits_by_query[qid] = sorted(hits_by_query[qid], key=lambda x: -x[1])[:top_k]
    return hits_by_query


# ──────────────────────────────────────────────────────────────────────
# Data generation (cached to disk)
# ──────────────────────────────────────────────────────────────────────

def generate_or_load_data(cache_dir):
    """Generate training data or load from cache."""
    cache_dir = Path(cache_dir)
    feat_path = cache_dir / "ml_features.npy"
    scores_path = cache_dir / "ml_sw_scores.npy"
    pairs_path = cache_dir / "ml_pairs.npy"

    if feat_path.exists() and scores_path.exists() and pairs_path.exists():
        print("  Loading cached training data...", flush=True)
        features = np.load(feat_path)
        sw_scores = np.load(scores_path)
        pairs = np.load(pairs_path)
        print(f"  Loaded {len(features)} pairs from cache", flush=True)
        return features, sw_scores, pairs

    print("  Generating training data (this takes ~25 min)...", flush=True)

    scop_dir = Path("benchmarks/data/scop_search_results")
    out_dir = Path("benchmarks/data/speed_sensitivity_results")

    db_index = load_database(out_dir / "clustkit_db")
    with open(scop_dir / "metadata.json") as f:
        full_metadata = json.load(f)
    all_query_sids = full_metadata["query_sids"]
    random.seed(42)
    query_sids = sorted(random.sample(all_query_sids, min(2000, len(all_query_sids))))
    all_seqs = dict(read_fasta(scop_dir / "queries.fasta"))
    sub_seqs = [(sid, all_seqs[sid]) for sid in query_sids if sid in all_seqs]
    query_fasta = str(out_dir / "queries_subset.fasta")
    write_fasta(sub_seqs, query_fasta)
    query_ds = read_sequences(query_fasta, "protein")
    db_ds = db_index.dataset

    nq = query_ds.num_sequences
    nd = db_ds.num_sequences
    q_flat = query_ds.flat_sequences
    q_off = query_ds.offsets.astype(np.int64)
    q_lens = query_ds.lengths.astype(np.int32)
    db_flat = db_ds.flat_sequences

    # Phase A standard k=3
    k = int32(db_index.params.get("kmer_index_k", db_index.params["kmer_size"]))
    mc = 8000
    freq_thresh = compute_freq_threshold(db_index.kmer_freqs, nd, 99.5)
    out_targets = np.empty((nq, mc), dtype=np.int32)
    out_counts = np.zeros(nq, dtype=np.int32)

    t0 = time.perf_counter()
    _batch_score_queries(
        q_flat, q_off, q_lens, k, int32(20),
        db_index.kmer_offsets, db_index.kmer_entries, db_index.kmer_freqs,
        freq_thresh, int32(nd), int32(2), int32(2), int32(10),
        int32(mc), int32(200000), out_targets, out_counts,
    )
    print(f"  Standard k=3: {int(out_counts.sum())} cands ({time.perf_counter()-t0:.0f}s)", flush=True)

    # Flatten
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
    print(f"  Reduced k=5: {int(red_out_c.sum())} cands", flush=True)

    # Union
    all_packed = [candidate_pairs[:, 0].astype(np.int64) * nd + candidate_pairs[:, 1].astype(np.int64)]
    for out_t, out_c in [(red_out_t, red_out_c)]:
        tot = int(out_c.sum())
        if tot > 0:
            pairs = np.empty((tot, 2), dtype=np.int32)
            p = 0
            for qi in range(nq):
                nc = int(out_c[qi])
                if nc > 0:
                    pairs[p:p+nc, 0] = qi
                    pairs[p:p+nc, 1] = out_t[qi, :nc]
                    p += nc
            all_packed.append(pairs[:, 0].astype(np.int64) * nd + pairs[:, 1].astype(np.int64))
    union = np.unique(np.concatenate(all_packed))
    candidate_pairs = np.empty((len(union), 2), dtype=np.int32)
    candidate_pairs[:, 0] = (union // nd).astype(np.int32)
    candidate_pairs[:, 1] = (union % nd).astype(np.int32)
    print(f"  Union: {len(candidate_pairs)} total pairs", flush=True)

    # Features
    t0 = time.perf_counter()
    q_kmer_sets = _build_query_kmer_sets(q_flat, q_off, q_lens, int32(3), int32(20))
    features = _compute_prefilter_features(
        candidate_pairs, q_kmer_sets, q_lens,
        db_flat, db_ds.offsets.astype(np.int64), db_ds.lengths.astype(np.int32),
        int32(3), int32(20),
    )
    print(f"  Features: {features.shape} ({time.perf_counter()-t0:.0f}s)", flush=True)

    # SW alignment
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
        merged["lengths"], np.float32(0.1), int32(band_width), BLOSUM62,
    )
    print(f"  SW alignment: {len(candidate_pairs)} pairs ({time.perf_counter()-t0:.0f}s)", flush=True)

    # Save
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(feat_path, features)
    np.save(scores_path, raw_scores)
    np.save(pairs_path, candidate_pairs)
    print(f"  Saved to {cache_dir}", flush=True)

    return features, raw_scores, candidate_pairs


# ──────────────────────────────────────────────────────────────────────
# Main: model comparison
# ──────────────────────────────────────────────────────────────────────

def main():
    scop_dir = Path("benchmarks/data/scop_search_results")
    out_dir = Path("benchmarks/data/speed_sensitivity_results")
    cache_dir = out_dir / "ml_cache"

    # Load metadata for ROC1
    with open(scop_dir / "metadata.json") as f:
        full_metadata = json.load(f)
    all_query_sids = full_metadata["query_sids"]
    random.seed(42)
    query_sids = sorted(random.sample(all_query_sids, min(2000, len(all_query_sids))))
    metadata = dict(full_metadata)
    metadata["query_sids"] = query_sids

    # Load datasets for ROC1 eval
    query_fasta = str(out_dir / "queries_subset.fasta")
    query_ds = read_sequences(query_fasta, "protein")
    db_index = load_database(out_dir / "clustkit_db")
    db_ds = db_index.dataset
    nq = query_ds.num_sequences

    print("=" * 110)
    print("ML Prefilter Model Comparison")
    print("=" * 110, flush=True)

    # Generate or load data
    print("\nStep 1: Training data", flush=True)
    features, sw_scores, candidate_pairs = generate_or_load_data(cache_dir)
    n_total = len(features)

    # Train/test split by query
    query_ids = candidate_pairs[:, 0]
    unique_queries = np.unique(query_ids)
    np.random.seed(42)
    np.random.shuffle(unique_queries)
    split = int(len(unique_queries) * 0.8)
    train_queries = set(unique_queries[:split].tolist())
    train_mask = np.array([int(qi) in train_queries for qi in query_ids])
    test_mask = ~train_mask

    X_train, X_test = features[train_mask], features[test_mask]
    y_train, y_test = sw_scores[train_mask], sw_scores[test_mask]

    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}", flush=True)

    # Baseline ROC1 (no filtering)
    passing_mask = sw_scores > 0
    hits_all = collect_hits(candidate_pairs[passing_mask], sw_scores[passing_mask],
                            query_ds, db_ds, nq)
    baseline_roc1 = evaluate_roc1(hits_all, metadata)
    print(f"  Baseline ROC1 (no prefilter): {baseline_roc1:.4f}\n", flush=True)

    # ── Define models ────────────────────────────────────────────────
    from sklearn.ensemble import (
        HistGradientBoostingRegressor,
        RandomForestRegressor,
        GradientBoostingRegressor,
    )
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor

    models = [
        ("HistGBR (d=6, n=200)", HistGradientBoostingRegressor(
            max_iter=200, max_depth=6, learning_rate=0.1, random_state=42)),
        ("HistGBR (d=4, n=100)", HistGradientBoostingRegressor(
            max_iter=100, max_depth=4, learning_rate=0.1, random_state=42)),
        ("RandomForest (n=50)", RandomForestRegressor(
            n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)),
        ("RandomForest (n=100)", RandomForestRegressor(
            n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)),
        ("Ridge (linear)", Ridge(alpha=1.0)),
        ("MLP (64-32)", MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=50, random_state=42,
            early_stopping=True, validation_fraction=0.1)),
        ("MLP (128-64-32)", MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), max_iter=100, random_state=42,
            early_stopping=True, validation_fraction=0.1)),
    ]

    # ── Train and evaluate each model ────────────────────────────────
    print("Step 2: Model training and evaluation")
    print("=" * 110, flush=True)
    print(f"  {'Model':35s} {'Train(s)':>8s} {'Infer(s)':>8s} {'Pearson r':>10s} "
          f"{'MAE':>8s} {'RMSE':>8s}", flush=True)
    print("  " + "-" * 85, flush=True)

    results = []
    for name, model in models:
        try:
            # Train
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0

            # Inference (3 runs, take median)
            inf_times = []
            for _ in range(3):
                t0 = time.perf_counter()
                y_pred_test = model.predict(X_test)
                inf_times.append(time.perf_counter() - t0)
            inf_time = np.median(inf_times)

            # Scale inference to full dataset
            inf_time_full = inf_time * (n_total / len(X_test))

            # Metrics
            corr = np.corrcoef(y_test, y_pred_test)[0, 1]
            mae = float(np.mean(np.abs(y_test - y_pred_test)))
            rmse = float(np.sqrt(np.mean((y_test - y_pred_test)**2)))

            print(f"  {name:35s} {train_time:8.1f} {inf_time_full:8.1f} "
                  f"{corr:10.4f} {mae:8.2f} {rmse:8.2f}", flush=True)

            results.append({
                "name": name, "model": model,
                "train_time": train_time, "inf_time_full": inf_time_full,
                "corr": corr, "mae": mae, "rmse": rmse,
            })
        except Exception as e:
            print(f"  {name:35s} FAILED: {e}", flush=True)

    # ── Non-ML baseline: threshold on shared_k3 alone ────────────────
    # shared_k3 is feature index 0
    shared_k3_test = X_test[:, 0]
    shared_k3_corr = np.corrcoef(y_test, shared_k3_test)[0, 1]
    print(f"\n  {'shared_k3 alone (no ML)':35s} {'—':>8s} {'<0.1':>8s} "
          f"{shared_k3_corr:10.4f} {'—':>8s} {'—':>8s}", flush=True)

    # ── Prefilter simulation with MAE margin ─────────────────────────
    print(f"\n\nStep 3: Prefilter simulation (MAE-margin approach)")
    print("=" * 110, flush=True)

    median_score = float(np.median(sw_scores))
    print(f"  Score threshold (median): {median_score:.1f}")
    print(f"  Total pairs: {n_total:,}\n", flush=True)

    for r in results:
        name = r["name"]
        model = r["model"]
        mae = r["mae"]
        margin = 1.5 * mae

        y_pred_all = model.predict(features)
        keep_mask = y_pred_all + margin >= median_score

        n_kept = int(keep_mask.sum())
        pct_kept = 100 * n_kept / n_total

        # ROC1 on kept pairs
        kept_pairs = candidate_pairs[keep_mask]
        kept_scores = sw_scores[keep_mask]
        kept_passing = kept_scores > 0
        if kept_passing.sum() > 0:
            hits_filt = collect_hits(kept_pairs[kept_passing], kept_scores[kept_passing],
                                    query_ds, db_ds, nq)
            filt_roc1 = evaluate_roc1(hits_filt, metadata)
        else:
            filt_roc1 = 0.0

        roc1_loss = baseline_roc1 - filt_roc1
        overhead = 13.0 + r["inf_time_full"]  # feature computation + inference
        est_saved = 1250 * (1 - n_kept / n_total)
        net_saving = est_saved - overhead

        print(f"  {name:35s}  kept={pct_kept:5.1f}%  ROC1={filt_roc1:.4f} "
              f"(loss={roc1_loss:+.4f})  MAE={mae:.1f}  margin={margin:.1f}  "
              f"net_save={net_saving:.0f}s", flush=True)

    # ── Also test with different margin factors ──────────────────────
    print(f"\n\nStep 4: Margin factor sensitivity (best model)")
    print("=" * 110, flush=True)

    # Pick model with best correlation
    best = max(results, key=lambda r: r["corr"])
    print(f"  Best model: {best['name']} (r={best['corr']:.4f}, MAE={best['mae']:.2f})\n", flush=True)

    y_pred_all = best["model"].predict(features)
    mae = best["mae"]

    print(f"  {'Margin':>10s} {'Factor':>8s} {'% Kept':>8s} {'ROC1':>7s} "
          f"{'ROC1 loss':>10s} {'Net save':>10s}", flush=True)
    print("  " + "-" * 60, flush=True)

    for factor in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        margin = factor * mae
        keep_mask = y_pred_all + margin >= median_score

        n_kept = int(keep_mask.sum())
        pct_kept = 100 * n_kept / n_total

        kept_pairs = candidate_pairs[keep_mask]
        kept_scores = sw_scores[keep_mask]
        kept_passing = kept_scores > 0
        if kept_passing.sum() > 0:
            hits_filt = collect_hits(kept_pairs[kept_passing], kept_scores[kept_passing],
                                    query_ds, db_ds, nq)
            filt_roc1 = evaluate_roc1(hits_filt, metadata)
        else:
            filt_roc1 = 0.0

        roc1_loss = baseline_roc1 - filt_roc1
        overhead = 13.0 + best["inf_time_full"]
        est_saved = 1250 * (1 - n_kept / n_total)
        net_saving = est_saved - overhead

        print(f"  {margin:10.1f} {factor:8.1f} {pct_kept:7.1f}% {filt_roc1:7.4f} "
              f"{roc1_loss:+10.4f} {net_saving:9.0f}s", flush=True)

    print(f"\n  Reference: baseline ROC1={baseline_roc1:.4f}, MMseqs2=0.7942")

    # Save results
    summary = [{k: v for k, v in r.items() if k != "model"} for r in results]
    with open(out_dir / "ml_model_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {out_dir / 'ml_model_comparison.json'}")


if __name__ == "__main__":
    main()
