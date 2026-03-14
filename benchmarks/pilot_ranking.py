#!/usr/bin/env python3
"""Pilot: test different hit ranking methods to improve ROC1.

The k=3 baseline gets ROC1=0.736 using raw BLOSUM62 NW scores for ranking.
MMseqs2 gets ROC1=0.794. The gap is likely due to ranking quality.

This pilot tests:
  1. Raw BLOSUM62 score (current baseline)
  2. E-value adjusted score: lambda*S - log(q_len * t_len)
  3. Identity ranking (already length-normalized)
  4. Per-residue score: S / min(q_len, t_len)
  5. Bit score: (lambda*S - ln(K)) / ln(2)
  6. Length-normalized score: S / sqrt(q_len * t_len)
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


# Karlin-Altschul parameters for BLOSUM62 with gaps (-11/-1)
LAMBDA_BL62 = 0.267
K_BL62 = 0.041
LN2 = math.log(2)


def evaluate_with_ranking(hits_by_query, metadata, rank_fn, label):
    """Evaluate ROC1/MAP using a custom ranking function.

    rank_fn: callable(hit) -> float (higher = better)
    """
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

        # Re-rank by the provided ranking function
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

    # Run search with large top_k to get all hits
    print("Running search (k=3, mc=2000, topk=10000, top_k=5000)...")
    t0 = time.perf_counter()
    results = search_kmer_index(
        db_index, query_ds,
        threshold=0.1, top_k=5000,
        max_cands_per_query=2000,
        phase_a_topk=10000,
        kmer_score_thresh=0,  # disable similar matching (k=3 database)
    )
    elapsed = time.perf_counter() - t0
    print(f"Search completed in {elapsed:.1f}s")

    total_hits = sum(len(h) for h in results.hits)
    queries_with = sum(1 for h in results.hits if h)
    print(f"Total hits: {total_hits}, queries with hits: {queries_with}/{results.num_queries}\n")

    # Collect all hits with full metadata
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

    # Print score distribution
    all_scores = [h["score"] for hits in hits_by_query.values() for h in hits]
    all_identities = [h["identity"] for hits in hits_by_query.values() for h in hits]
    print(f"Score range: [{min(all_scores):.0f}, {max(all_scores):.0f}]")
    print(f"Identity range: [{min(all_identities):.3f}, {max(all_identities):.3f}]")
    print()

    # Define ranking functions
    ranking_methods = {
        "raw_score": lambda h: h["score"],
        "identity": lambda h: h["identity"],
        "evalue_score": lambda h: (
            LAMBDA_BL62 * h["score"]
            - math.log(max(h["query_length"] * h["target_length"], 1))
        ),
        "bit_score": lambda h: (
            (LAMBDA_BL62 * h["score"] - math.log(K_BL62)) / LN2
        ),
        "per_residue_score": lambda h: (
            h["score"] / max(min(h["query_length"], h["target_length"]), 1)
        ),
        "norm_score_sqrt": lambda h: (
            h["score"] / max(math.sqrt(h["query_length"] * h["target_length"]), 1)
        ),
        "score_div_shorter": lambda h: (
            h["score"] / max(min(h["query_length"], h["target_length"]), 1)
        ),
        "score_div_longer": lambda h: (
            h["score"] / max(max(h["query_length"], h["target_length"]), 1)
        ),
        "neg_log_evalue": lambda h: (
            LAMBDA_BL62 * h["score"]
            - math.log(max(K_BL62 * h["query_length"] * h["target_length"], 1e-300))
        ),
    }

    print("=" * 100)
    print("Ranking method comparison (same candidates, different ranking)")
    print("=" * 100)
    print(f"{'Method':35s} {'ROC1':>8s} {'MAP':>8s}")
    print("-" * 55)

    results_list = []
    for name, rank_fn in ranking_methods.items():
        metrics = evaluate_with_ranking(hits_by_query, metadata, rank_fn, name)
        print(f"{name:35s} {metrics['mean_roc1']:8.4f} {metrics['mean_map']:8.4f}")
        results_list.append({"method": name, **metrics})

    # Save results
    with open(out_dir / "ranking_pilot.json", "w") as f:
        json.dump(results_list, f, indent=2)

    print("\nReference:")
    print("  ClustKIT k=3 baseline (raw_score): ROC1=0.7360")
    print("  MMseqs2 (s=4):                     ROC1=0.7311")
    print("  MMseqs2 (s=7.5):                   ROC1=0.7942")
    print("  DIAMOND (very-sensitive):           ROC1=0.7963")

    # Also test with higher max_cands
    print("\n\n" + "=" * 100)
    print("Higher max_cands tests")
    print("=" * 100)

    for mc in [3000, 5000]:
        print(f"\nRunning search with mc={mc}...")
        t0 = time.perf_counter()
        results2 = search_kmer_index(
            db_index, query_ds,
            threshold=0.1, top_k=5000,
            max_cands_per_query=mc,
            phase_a_topk=10000,
            kmer_score_thresh=0,
        )
        elapsed2 = time.perf_counter() - t0

        hits2 = defaultdict(list)
        for qhits in results2.hits:
            for h in qhits:
                hits2[h.query_id].append({
                    "target_id": h.target_id,
                    "identity": h.identity,
                    "score": h.score,
                    "query_length": h.query_length,
                    "target_length": h.target_length,
                })

        total_hits2 = sum(len(h) for h in results2.hits)
        print(f"  mc={mc}: {total_hits2} hits, {elapsed2:.1f}s")

        # Test best ranking methods
        for name in ["raw_score", "evalue_score", "identity", "per_residue_score"]:
            rank_fn = ranking_methods[name]
            metrics = evaluate_with_ranking(hits2, metadata, rank_fn, name)
            print(f"    {name:30s} ROC1={metrics['mean_roc1']:.4f} MAP={metrics['mean_map']:.4f}")


if __name__ == "__main__":
    main()
