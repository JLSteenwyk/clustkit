#!/usr/bin/env python3
"""Pilot: test two-stage Phase A → selective Phase B approach.

Compares the optimized two-stage diagonal scoring against baseline (Phase A only)
to verify sensitivity improvement and measure speed.
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
    return {"label": label, "time": elapsed, "avg_cands": avg_cands, **metrics}


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

    print("=" * 120)
    print("Two-stage Phase A → selective Phase B optimization")
    print("=" * 120)

    configs = [
        # Phase A only baselines
        {"label": "Phase A only (mc=2000, mdh=1)",
         "max_cands_per_query": 2000, "min_diag_hits": 1},

        # Two-stage with different phase_a_topk values
        {"label": "Two-stage (topk=2000, mc=2000, mdh=2)",
         "max_cands_per_query": 2000, "min_diag_hits": 2, "phase_a_topk": 2000},
        {"label": "Two-stage (topk=3000, mc=2000, mdh=2)",
         "max_cands_per_query": 2000, "min_diag_hits": 2, "phase_a_topk": 3000},
        {"label": "Two-stage (topk=5000, mc=2000, mdh=2)",
         "max_cands_per_query": 2000, "min_diag_hits": 2, "phase_a_topk": 5000},
        {"label": "Two-stage (topk=10000, mc=2000, mdh=2)",
         "max_cands_per_query": 2000, "min_diag_hits": 2, "phase_a_topk": 10000},
        {"label": "Two-stage (topk=20000, mc=2000, mdh=2)",
         "max_cands_per_query": 2000, "min_diag_hits": 2, "phase_a_topk": 20000},

        # Two-stage with higher min_diag_hits
        {"label": "Two-stage (topk=5000, mc=2000, mdh=3)",
         "max_cands_per_query": 2000, "min_diag_hits": 3, "phase_a_topk": 5000},

        # Two-stage with lower freq threshold
        {"label": "Two-stage (topk=5000, mc=2000, mdh=2, fp=95)",
         "max_cands_per_query": 2000, "min_diag_hits": 2, "phase_a_topk": 5000,
         "freq_percentile": 95.0},
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

    with open(out_dir / "two_stage_pilot_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 120)
    print("SUMMARY (sorted by ROC1)")
    print("=" * 120)
    print(f"{'Config':55s} {'ROC1':>7s} {'MAP':>7s} {'Avg cands':>10s} {'Time':>8s}")
    print("-" * 95)
    for r in sorted(results, key=lambda x: -x["mean_roc1"]):
        print(
            f"{r['label']:55s} {r['mean_roc1']:7.4f} {r['mean_map']:7.4f} "
            f"{r['avg_cands']:10.0f} {r['time']:8.1f}"
        )

    # Compare with MMseqs2 reference
    print("\nReference (from full benchmark):")
    print("  MMseqs2 (s=1):   ROC1=0.6452")
    print("  MMseqs2 (s=4):   ROC1=0.7311")
    print("  MMseqs2 (s=5.7): ROC1=0.7692")
    print("  MMseqs2 (s=7.5): ROC1=0.7942")


if __name__ == "__main__":
    main()
