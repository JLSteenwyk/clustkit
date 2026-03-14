#!/usr/bin/env python3
"""Pilot: test different hit ranking methods + candidate generation parameters.

The key insight: phase_a_topk dramatically affects ROC1. With topk=10000, ROC1=0.696.
The reference ROC1=0.736 used topk=50000.

This pilot sweeps:
  1. phase_a_topk to find optimal candidate generation
  2. min_diag_hits to test Phase B vs Phase A-only
  3. Ranking methods on the best configuration
"""

import json
import math
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


LAMBDA_BL62 = 0.267
K_BL62 = 0.041
LN2 = math.log(2)


def collect_hits(results):
    """Convert SearchResults into a dict of query_id -> list of hit dicts."""
    hits_by_query = defaultdict(list)
    for qhits in results.hits:
        for h in qhits:
            hits_by_query[h.query_id].append({
                "target_id": h.target_id,
                "identity": h.identity,
                "score": h.score,
                "query_length": h.query_length,
                "target_length": h.target_length,
            })
    return hits_by_query


def evaluate_ranking(hits_by_query, metadata, rank_fn):
    """Evaluate ROC1/MAP using a custom ranking function."""
    domain_info = metadata["domain_info"]
    family_sizes = metadata["family_sizes_in_db"]
    query_sids = set(metadata["query_sids"])

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
        if not query_hits:
            roc1_values.append(0.0)
            map_values.append(0.0)
            continue

        scored = [(h, rank_fn(h)) for h in query_hits]
        scored.sort(key=lambda x: -x[1])

        ranked = []
        for h, _ in scored:
            label_str = classify_hit(qid, h["target_id"], domain_info)
            if label_str != "IGNORE":
                ranked.append(RankedHit(
                    target_id=h["target_id"],
                    score=rank_fn(h),
                    label=label_str,
                ))

        roc1_values.append(compute_roc_n(ranked, 1, total_tp))
        map_values.append(compute_average_precision(ranked, total_tp))

    n = len(roc1_values)
    return {
        "n_queries": n,
        "mean_roc1": float(np.mean(roc1_values)) if n else 0,
        "mean_map": float(np.mean(map_values)) if n else 0,
    }


RANK_FNS = {
    "raw_score": lambda h: h["score"],
    "identity": lambda h: h["identity"],
    "evalue": lambda h: (
        LAMBDA_BL62 * h["score"]
        - math.log(max(K_BL62 * h["query_length"] * h["target_length"], 1e-300))
    ),
    "per_residue": lambda h: (
        h["score"] / max(min(h["query_length"], h["target_length"]), 1)
    ),
    "norm_sqrt": lambda h: (
        h["score"] / max(math.sqrt(h["query_length"] * h["target_length"]), 1)
    ),
}


def run_and_evaluate(db_index, query_ds, metadata, label, **kwargs):
    """Run search and evaluate with all ranking methods."""
    t0 = time.perf_counter()
    results = search_kmer_index(
        db_index, query_ds, threshold=0.1, top_k=5000, **kwargs
    )
    elapsed = time.perf_counter() - t0

    hits_by_query = collect_hits(results)
    total_hits = sum(len(h) for h in results.hits)
    avg_cands = results.num_candidates / max(results.num_queries, 1)

    print(f"\n  {label}: {total_hits} hits, {avg_cands:.0f} cands/q, "
          f"{results.num_aligned} aligned, {elapsed:.1f}s")

    for name, rank_fn in RANK_FNS.items():
        metrics = evaluate_ranking(hits_by_query, metadata, rank_fn)
        print(f"    {name:20s} ROC1={metrics['mean_roc1']:.4f} MAP={metrics['mean_map']:.4f}")

    return elapsed


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
    print(f"Loaded {query_ds.num_sequences} queries")

    print("\n" + "=" * 100)
    print("PART 1: phase_a_topk sweep (mc=2000, min_diag_hits=2)")
    print("=" * 100)

    for topk in [5000, 10000, 20000, 50000, 100000]:
        run_and_evaluate(
            db_index, query_ds, metadata,
            f"topk={topk}",
            max_cands_per_query=2000,
            phase_a_topk=topk,
            kmer_score_thresh=0,
        )

    print("\n" + "=" * 100)
    print("PART 2: Phase A only (min_diag_hits=1, no diagonal scoring)")
    print("=" * 100)

    for topk in [5000, 10000, 50000]:
        run_and_evaluate(
            db_index, query_ds, metadata,
            f"topk={topk}, no_phaseB",
            max_cands_per_query=2000,
            phase_a_topk=topk,
            min_diag_hits=1,  # effectively disables Phase B
            kmer_score_thresh=0,
        )

    print("\n" + "=" * 100)
    print("PART 3: Higher max_cands with best phase_a_topk")
    print("=" * 100)

    for mc in [3000, 5000, 8000]:
        run_and_evaluate(
            db_index, query_ds, metadata,
            f"mc={mc}, topk=50000",
            max_cands_per_query=mc,
            phase_a_topk=50000,
            kmer_score_thresh=0,
        )

    print("\n\nReference:")
    print("  ClustKIT k=3 (very-sensitive): ROC1=0.7360")
    print("  MMseqs2 (s=4):                 ROC1=0.7311")
    print("  MMseqs2 (s=7.5):               ROC1=0.7942")
    print("  DIAMOND (very-sensitive):       ROC1=0.7963")


if __name__ == "__main__":
    main()
