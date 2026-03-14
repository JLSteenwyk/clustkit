#!/usr/bin/env python3
"""Pilot: find optimal ungapped pre-filter threshold.

Tests different min_ungapped_score values to find the best trade-off
between filtering rate and sensitivity loss.
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
    classify_hit,
    compute_roc_n,
    compute_average_precision,
    read_fasta,
    write_fasta,
    RankedHit,
)

import numba
numba.set_num_threads(8)

from clustkit.database import load_database
from clustkit.io import read_sequences
from clustkit.kmer_index import search_kmer_index


def evaluate_hits(hits, metadata):
    domain_info = metadata["domain_info"]
    family_sizes = metadata["family_sizes_in_db"]
    query_sids = set(metadata["query_sids"])

    hits_by_query = defaultdict(list)
    for qid, tid, score in hits:
        hits_by_query[qid].append((tid, score))

    roc1_values = []
    map_values = []

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
        map_values.append(compute_average_precision(ranked, total_tp))

    n = len(roc1_values)
    return {
        "n_queries": n,
        "mean_roc1": float(np.mean(roc1_values)) if n else 0,
        "mean_map": float(np.mean(map_values)) if n else 0,
    }


def run_config(db_index, query_ds, metadata, label, **kwargs):
    t0 = time.perf_counter()
    results = search_kmer_index(db_index, query_ds, threshold=0.1, top_k=500, **kwargs)
    elapsed = time.perf_counter() - t0

    hits = []
    for qhits in results.hits:
        for h in qhits:
            hits.append((h.query_id, h.target_id, h.score if h.score != 0 else h.identity))

    metrics = evaluate_hits(hits, metadata)
    avg_cands = results.num_candidates / max(results.num_queries, 1)
    print(
        f"  {label:55s} ROC1={metrics['mean_roc1']:.4f} "
        f"MAP={metrics['mean_map']:.4f} "
        f"cands={avg_cands:.0f}/q "
        f"aligned={results.num_aligned} "
        f"time={elapsed:.1f}s"
    )
    return {"label": label, "time": elapsed, "avg_cands": avg_cands,
            "num_aligned": results.num_aligned, **metrics}


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

    print("Loading database index...")
    db_index = load_database(out_dir / "clustkit_db")

    query_ds = read_sequences(query_fasta, "protein")
    print(f"Loaded {query_ds.num_sequences} queries\n")

    results = []

    # Use very-sensitive mode (phase_a_topk=50000) to maximize the number of
    # candidates and see the full effect of different thresholds
    print("=" * 120)
    print("Ungapped pre-filter threshold sweep (very-sensitive mode, phase_a_topk=50000)")
    print("=" * 120)

    configs = [
        # Baseline: no ungapped pre-filter
        {"label": "No ungapped filter (baseline)",
         "phase_a_topk": 50000, "min_ungapped_score": 0},

        # Threshold sweep
        {"label": "min_ungapped_score=5",
         "phase_a_topk": 50000, "min_ungapped_score": 5},
        {"label": "min_ungapped_score=8",
         "phase_a_topk": 50000, "min_ungapped_score": 8},
        {"label": "min_ungapped_score=10",
         "phase_a_topk": 50000, "min_ungapped_score": 10},
        {"label": "min_ungapped_score=12",
         "phase_a_topk": 50000, "min_ungapped_score": 12},
        {"label": "min_ungapped_score=15",
         "phase_a_topk": 50000, "min_ungapped_score": 15},
        {"label": "min_ungapped_score=20",
         "phase_a_topk": 50000, "min_ungapped_score": 20},
        {"label": "min_ungapped_score=25",
         "phase_a_topk": 50000, "min_ungapped_score": 25},
        {"label": "min_ungapped_score=30",
         "phase_a_topk": 50000, "min_ungapped_score": 30},
    ]

    for cfg in configs:
        label = cfg.pop("label")
        try:
            r = run_config(db_index, query_ds, metadata, label, **cfg)
            results.append(r)
        except Exception as e:
            import traceback
            print(f"  {label:55s} FAILED: {e}")
            traceback.print_exc()

    with open(out_dir / "ungapped_threshold_pilot.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(f"{'Config':55s} {'ROC1':>7s} {'MAP':>7s} {'Aligned':>10s} {'Time':>8s}")
    print("-" * 95)
    for r in results:
        print(
            f"{r['label']:55s} {r['mean_roc1']:7.4f} {r['mean_map']:7.4f} "
            f"{r['num_aligned']:10d} {r['time']:8.1f}"
        )

    # Show ROC1 loss relative to baseline
    if results:
        baseline_roc1 = results[0]["mean_roc1"]
        print(f"\nROC1 loss relative to baseline ({baseline_roc1:.4f}):")
        for r in results[1:]:
            delta = r["mean_roc1"] - baseline_roc1
            pct_filtered = 100 * (1 - r["num_aligned"] / max(results[0]["num_aligned"], 1))
            speedup = results[0]["time"] / max(r["time"], 0.1)
            print(
                f"  {r['label']:55s} delta={delta:+.4f} "
                f"filtered={pct_filtered:.1f}% speedup={speedup:.2f}x"
            )


if __name__ == "__main__":
    main()
