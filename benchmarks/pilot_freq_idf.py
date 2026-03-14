#!/usr/bin/env python3
"""Pilot: test freq_percentile and IDF-weighted Phase A scoring.

Hypothesis: freq_thresh at 95th percentile drops informative k-mers shared
between distant homologs. Higher percentile or IDF weighting should help.
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
    return {**metrics, "time": elapsed}


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

    # Print k-mer freq distribution
    freqs = db_index.kmer_freqs
    nonzero = freqs[freqs > 0]
    print(f"K-mer freq stats (nonzero): n={len(nonzero)}, "
          f"p50={np.percentile(nonzero, 50):.0f}, "
          f"p90={np.percentile(nonzero, 90):.0f}, "
          f"p95={np.percentile(nonzero, 95):.0f}, "
          f"p99={np.percentile(nonzero, 99):.0f}, "
          f"p99.5={np.percentile(nonzero, 99.5):.0f}, "
          f"p99.9={np.percentile(nonzero, 99.9):.0f}, "
          f"max={nonzero.max()}")

    print("\n" + "=" * 130)
    print("PART 1: freq_percentile sweep (topk=200K, mc=2000)")
    print("=" * 130)

    for pctl in [90.0, 95.0, 97.5, 99.0, 99.5, 99.9]:
        run_config(
            db_index, query_ds, metadata,
            f"freq_pctl={pctl}",
            max_cands_per_query=2000,
            phase_a_topk=200000,
            freq_percentile=pctl,
            kmer_score_thresh=0,
        )

    print("\n" + "=" * 130)
    print("PART 2: freq_percentile with topk=200K, mc=3000")
    print("=" * 130)

    for pctl in [95.0, 99.0, 99.5]:
        run_config(
            db_index, query_ds, metadata,
            f"freq_pctl={pctl}, mc=3000",
            max_cands_per_query=3000,
            phase_a_topk=200000,
            freq_percentile=pctl,
            kmer_score_thresh=0,
        )

    print("\nReference:")
    print("  topk=200K, mc=2000, pctl=95: ROC1=0.7517")
    print("  MMseqs2 (s=7.5):             ROC1=0.7942")
    print("  DIAMOND:                     ROC1=0.7963")


if __name__ == "__main__":
    main()
