#!/usr/bin/env python3
"""End-to-end test: RandomForest prefilter integrated into search pipeline.

Trains an RF model on cached data, then runs full search with the
prefilter at various margin factors to measure speed vs ROC1 tradeoff.
"""

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

import numba
numba.set_num_threads(8)

from clustkit.database import load_database
from clustkit.io import read_sequences
from clustkit.kmer_index import search_kmer_index


def evaluate_roc1(results, metadata, query_ds):
    domain_info = metadata["domain_info"]
    family_sizes = metadata["family_sizes_in_db"]
    query_sids = set(metadata["query_sids"])
    hits_by_query = defaultdict(list)
    for qhits in results.hits:
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
    print(f"Loaded {query_ds.num_sequences} queries\n", flush=True)

    # ── Train RF model on cached data ────────────────────────────────
    print("=" * 100)
    print("Step 1: Train RandomForest on cached training data")
    print("=" * 100, flush=True)

    features = np.load(cache_dir / "ml_features.npy")
    sw_scores = np.load(cache_dir / "ml_sw_scores.npy")
    pairs = np.load(cache_dir / "ml_pairs.npy")

    # Train/test split by query
    query_ids = pairs[:, 0]
    unique_queries = np.unique(query_ids)
    np.random.seed(42)
    np.random.shuffle(unique_queries)
    split = int(len(unique_queries) * 0.8)
    train_queries = set(unique_queries[:split].tolist())
    train_mask = np.array([int(qi) in train_queries for qi in query_ids])

    X_train = features[train_mask]
    y_train = sw_scores[train_mask]
    X_test = features[~train_mask]
    y_test = sw_scores[~train_mask]

    from sklearn.ensemble import RandomForestRegressor

    t0 = time.perf_counter()
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=12, n_jobs=-1, random_state=42,
    )
    rf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    y_pred = rf.predict(X_test)
    mae = float(np.mean(np.abs(y_test - y_pred)))
    corr = np.corrcoef(y_test, y_pred)[0, 1]
    median_score = float(np.median(sw_scores))

    print(f"  Trained in {train_time:.1f}s", flush=True)
    print(f"  Pearson r: {corr:.4f}", flush=True)
    print(f"  MAE: {mae:.2f}", flush=True)
    print(f"  Median SW score (threshold): {median_score:.1f}\n", flush=True)

    # ── Run search with and without prefilter ────────────────────────
    print("=" * 100)
    print("Step 2: End-to-end search comparison")
    print("=" * 100, flush=True)

    base = dict(
        freq_percentile=99.5, phase_a_topk=200000, kmer_score_thresh=0,
        local_alignment=True, evalue_normalize=False, max_cands_per_query=8000,
        reduced_alphabet=True, reduced_k=5, use_idf=False,
    )

    configs = [
        ("No prefilter (baseline)", {**base, "ml_prefilter_model": None}),
        ("RF prefilter margin=2.0×MAE", {**base, "ml_prefilter_model": (rf, mae * 2.0, median_score)}),
        ("RF prefilter margin=1.5×MAE", {**base, "ml_prefilter_model": (rf, mae * 1.5, median_score)}),
        ("RF prefilter margin=1.0×MAE", {**base, "ml_prefilter_model": (rf, mae * 1.0, median_score)}),
        ("RF prefilter margin=0.5×MAE", {**base, "ml_prefilter_model": (rf, mae * 0.5, median_score)}),
    ]

    results_list = []
    elapsed_times = []

    for i, (label, kwargs) in enumerate(configs, 1):
        print(f"\n>>> Config {i}/{len(configs)}: {label}", flush=True)
        if elapsed_times:
            avg = sum(elapsed_times) / len(elapsed_times)
            remaining = (len(configs) - i + 1) * avg
            print(f"    ETA: ~{remaining/60:.1f} min remaining", flush=True)

        t0 = time.perf_counter()
        results = search_kmer_index(
            db_index, query_ds, threshold=0.1, top_k=500, **kwargs,
        )
        elapsed = time.perf_counter() - t0

        roc1 = evaluate_roc1(results, metadata, query_ds)

        print(f"  RESULT: {label}", flush=True)
        print(f"          ROC1={roc1:.4f}  aligned={results.num_aligned:>10d}  "
              f"time={elapsed:.1f}s", flush=True)

        results_list.append({
            "label": label, "roc1": roc1,
            "aligned": results.num_aligned, "time": elapsed,
        })
        elapsed_times.append(elapsed)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100, flush=True)

    baseline_time = results_list[0]["time"]
    print(f"  {'Config':45s} {'ROC1':>7s} {'Aligned':>10s} {'Time':>8s} {'Speedup':>8s} {'ROC1 loss':>10s}")
    print("  " + "-" * 95, flush=True)

    for r in results_list:
        speedup = baseline_time / max(r["time"], 0.1)
        loss = results_list[0]["roc1"] - r["roc1"]
        print(f"  {r['label']:45s} {r['roc1']:7.4f} {r['aligned']:10d} "
              f"{r['time']:7.1f}s {speedup:7.2f}x {loss:+10.4f}", flush=True)

    print(f"\n  RF model: MAE={mae:.2f}, inference overhead ~15s")
    print(f"  Reference: MMseqs2=0.7942 (14s)  DIAMOND=0.7963 (13s)")

    with open(out_dir / "rf_prefilter_results.json", "w") as f:
        json.dump(results_list, f, indent=2)


if __name__ == "__main__":
    main()
