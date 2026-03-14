#!/usr/bin/env python3
"""Quick pilot: test different k-mer index parameters on the SCOPe benchmark.

Varies min_total_hits, max_cands_per_query, and min_diag_hits to find the
best speed-sensitivity trade-off.
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
    compute_sensitivity_at_fdr,
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
    """Evaluate search hits."""
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
    """Run one configuration and return results."""
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
        f"  {label:50s} ROC1={metrics['mean_roc1']:.4f} "
        f"MAP={metrics['mean_map']:.4f} "
        f"cands={avg_cands:.0f}/q "
        f"aligned={results.num_aligned} "
        f"time={elapsed:.1f}s"
    )
    return {"label": label, "time": elapsed, "avg_cands": avg_cands, **metrics}


def main():
    scop_dir = Path("benchmarks/data/scop_search_results")
    out_dir = Path("benchmarks/data/speed_sensitivity_results")

    # Load metadata
    with open(scop_dir / "metadata.json") as f:
        full_metadata = json.load(f)

    # Subsample queries (same as benchmark)
    all_query_sids = full_metadata["query_sids"]
    random.seed(42)
    query_sids = sorted(random.sample(all_query_sids, min(2000, len(all_query_sids))))

    # Write subsampled query FASTA
    all_seqs = dict(read_fasta(scop_dir / "queries.fasta"))
    sub_seqs = [(sid, all_seqs[sid]) for sid in query_sids if sid in all_seqs]
    query_fasta = str(out_dir / "queries_subset.fasta")
    write_fasta(sub_seqs, query_fasta)

    metadata = dict(full_metadata)
    metadata["query_sids"] = query_sids

    # Load database
    print("Loading database index...")
    db_index = load_database(out_dir / "clustkit_db")

    # Load queries
    query_ds = read_sequences(query_fasta, "protein")
    print(f"Loaded {query_ds.num_sequences} queries\n")

    results = []

    # ── Test different configurations ──
    print("=" * 110)
    print("Testing parameter configurations")
    print("=" * 110)

    configs = [
        # Baseline
        {"label": "baseline (mc=2000, mth=2, mdh=1)",
         "max_cands_per_query": 2000, "min_total_hits": 2, "min_diag_hits": 1},

        # Vary min_total_hits
        {"label": "mth=3 (mc=2000)",
         "max_cands_per_query": 2000, "min_total_hits": 3, "min_diag_hits": 1},
        {"label": "mth=4 (mc=2000)",
         "max_cands_per_query": 2000, "min_total_hits": 4, "min_diag_hits": 1},
        {"label": "mth=5 (mc=2000)",
         "max_cands_per_query": 2000, "min_total_hits": 5, "min_diag_hits": 1},
        {"label": "mth=8 (mc=2000)",
         "max_cands_per_query": 2000, "min_total_hits": 8, "min_diag_hits": 1},
        {"label": "mth=12 (mc=2000)",
         "max_cands_per_query": 2000, "min_total_hits": 12, "min_diag_hits": 1},

        # Vary max_cands
        {"label": "mc=500 (mth=2)",
         "max_cands_per_query": 500, "min_total_hits": 2, "min_diag_hits": 1},
        {"label": "mc=1000 (mth=2)",
         "max_cands_per_query": 1000, "min_total_hits": 2, "min_diag_hits": 1},
        {"label": "mc=5000 (mth=2)",
         "max_cands_per_query": 5000, "min_total_hits": 2, "min_diag_hits": 1},

        # Phase B diagonal filtering
        {"label": "mdh=2 (mc=2000, mth=2)",
         "max_cands_per_query": 2000, "min_total_hits": 2, "min_diag_hits": 2},

        # Combined: higher min_total_hits + fewer max_cands (fast)
        {"label": "mth=5, mc=1000 (fast combo)",
         "max_cands_per_query": 1000, "min_total_hits": 5, "min_diag_hits": 1},
        {"label": "mth=8, mc=1000 (faster combo)",
         "max_cands_per_query": 1000, "min_total_hits": 8, "min_diag_hits": 1},

        # Lower freq threshold (filter more common k-mers)
        {"label": "freq_pctl=97 (mc=2000, mth=2)",
         "max_cands_per_query": 2000, "min_total_hits": 2, "min_diag_hits": 1,
         "freq_percentile": 97.0},
        {"label": "freq_pctl=95 (mc=2000, mth=2)",
         "max_cands_per_query": 2000, "min_total_hits": 2, "min_diag_hits": 1,
         "freq_percentile": 95.0},
    ]

    for cfg in configs:
        label = cfg.pop("label")
        try:
            r = run_config(db_index, query_ds, metadata, label, **cfg)
            results.append(r)
        except Exception as e:
            print(f"  {label:50s} FAILED: {e}")

    # Save results
    with open(out_dir / "param_pilot_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)
    print(f"{'Config':50s} {'ROC1':>7s} {'MAP':>7s} {'Avg cands':>10s} {'Time':>8s}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: -x["mean_roc1"]):
        print(
            f"{r['label']:50s} {r['mean_roc1']:7.4f} {r['mean_map']:7.4f} "
            f"{r['avg_cands']:10.0f} {r['time']:8.1f}"
        )


if __name__ == "__main__":
    main()
