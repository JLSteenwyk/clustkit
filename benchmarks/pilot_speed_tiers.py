#!/usr/bin/env python3
"""Pilot: find optimal speed-sensitivity tiers by varying max_cands + phase_a_topk.

The key insight: all queries hit the 2000 max_cands cap, so reducing max_cands
directly reduces alignment work. The question is whether the Phase A/B ranking
is good enough that top-N candidates contain most true homologs.
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
        f"  {label:60s} ROC1={metrics['mean_roc1']:.4f} "
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

    print("=" * 130)
    print("Speed tier optimization: vary max_cands_per_query + phase_a_topk")
    print("=" * 130)

    configs = [
        # Ultra-fast: few candidates
        {"label": "mc=200, topk=2000",
         "max_cands_per_query": 200, "phase_a_topk": 2000},
        {"label": "mc=300, topk=3000",
         "max_cands_per_query": 300, "phase_a_topk": 3000},
        {"label": "mc=500, topk=3000",
         "max_cands_per_query": 500, "phase_a_topk": 3000},

        # Fast
        {"label": "mc=500, topk=5000",
         "max_cands_per_query": 500, "phase_a_topk": 5000},
        {"label": "mc=800, topk=5000",
         "max_cands_per_query": 800, "phase_a_topk": 5000},

        # Default
        {"label": "mc=800, topk=10000",
         "max_cands_per_query": 800, "phase_a_topk": 10000},
        {"label": "mc=1000, topk=10000",
         "max_cands_per_query": 1000, "phase_a_topk": 10000},

        # Sensitive
        {"label": "mc=1000, topk=20000",
         "max_cands_per_query": 1000, "phase_a_topk": 20000},
        {"label": "mc=1500, topk=20000",
         "max_cands_per_query": 1500, "phase_a_topk": 20000},

        # Very sensitive
        {"label": "mc=1500, topk=50000",
         "max_cands_per_query": 1500, "phase_a_topk": 50000},
        {"label": "mc=2000, topk=50000",
         "max_cands_per_query": 2000, "phase_a_topk": 50000},

        # Reference: current settings
        {"label": "mc=2000, topk=3000 (current fast)",
         "max_cands_per_query": 2000, "phase_a_topk": 3000},
        {"label": "mc=2000, topk=10000 (current default)",
         "max_cands_per_query": 2000, "phase_a_topk": 10000},
    ]

    for cfg in configs:
        label = cfg.pop("label")
        try:
            r = run_config(db_index, query_ds, metadata, label, **cfg)
            results.append(r)
        except Exception as e:
            import traceback
            print(f"  {label:60s} FAILED: {e}")
            traceback.print_exc()

    with open(out_dir / "speed_tier_pilot.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 130)
    print("SUMMARY (sorted by time)")
    print("=" * 130)
    print(f"{'Config':60s} {'ROC1':>7s} {'MAP':>7s} {'Aligned':>10s} {'Time':>8s}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: x["time"]):
        print(
            f"{r['label']:60s} {r['mean_roc1']:7.4f} {r['mean_map']:7.4f} "
            f"{r['num_aligned']:10d} {r['time']:8.1f}"
        )

    print(f"\nPareto frontier (sorted by ROC1, showing only Pareto-optimal):")
    print("-" * 100)
    # Find Pareto frontier: configs where no other config is both faster AND more sensitive
    pareto = []
    for r in sorted(results, key=lambda x: -x["mean_roc1"]):
        if not pareto or r["time"] < pareto[-1]["time"]:
            pareto.append(r)
    for r in pareto:
        print(
            f"{r['label']:60s} {r['mean_roc1']:7.4f} {r['mean_map']:7.4f} "
            f"{r['num_aligned']:10d} {r['time']:8.1f}"
        )

    print("\nReference (from previous benchmarks):")
    print("  MMseqs2 (s=1):   ROC1=0.6452  Time=~5s")
    print("  MMseqs2 (s=4):   ROC1=0.7311  Time=~10s")
    print("  MMseqs2 (s=5.7): ROC1=0.7692  Time=~18s")
    print("  MMseqs2 (s=7.5): ROC1=0.7942  Time=~44s")


if __name__ == "__main__":
    main()
