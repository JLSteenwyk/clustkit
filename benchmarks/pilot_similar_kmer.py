#!/usr/bin/env python3
"""Pilot: test BLOSUM62 similar k-mer matching (k=5) for sensitivity improvement.

Rebuilds the k-mer index with k=5 (in-memory, reusing existing sketches/LSH),
then sweeps kmer_score_thresh to find optimal speed-sensitivity trade-off.
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
from clustkit.kmer_index import build_kmer_index, search_kmer_index


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

    # Rebuild k-mer index with k=5 for similar matching
    print("Rebuilding k-mer index with k=5...")
    ds = db_index.dataset
    t0 = time.perf_counter()
    kmer_offsets, kmer_entries, kmer_freqs = build_kmer_index(
        ds.flat_sequences, ds.offsets, ds.lengths, 5, "protein"
    )
    print(f"  k=5 index built in {time.perf_counter() - t0:.1f}s")
    print(f"  {len(kmer_freqs)} possible 5-mers, "
          f"{len(kmer_entries):,} total entries")

    # Replace index in the database object
    db_index.kmer_offsets = kmer_offsets
    db_index.kmer_entries = kmer_entries
    db_index.kmer_freqs = kmer_freqs
    db_index.params["kmer_index_k"] = 5

    query_ds = read_sequences(query_fasta, "protein")
    print(f"Loaded {query_ds.num_sequences} queries\n")

    results = []

    # ── First: warmup JIT compilation with a small run ──────────────
    print("Warming up JIT compilation...")
    t0 = time.perf_counter()
    _ = search_kmer_index(
        db_index, query_ds, threshold=0.1, top_k=10,
        max_cands_per_query=100, kmer_score_thresh=17,
    )
    print(f"  JIT warmup: {time.perf_counter() - t0:.1f}s\n")

    print("=" * 130)
    print("Similar k-mer matching: sweep kmer_score_thresh")
    print("=" * 130)

    configs = [
        # Exact matching baseline (k=5, no similar matching)
        {"label": "k=5 exact only (score_thresh=0)",
         "kmer_score_thresh": 0, "max_cands_per_query": 1500},

        # Similar matching at various thresholds
        {"label": "similar k=5, T=17 (fast)",
         "kmer_score_thresh": 17, "max_cands_per_query": 1500},
        {"label": "similar k=5, T=15",
         "kmer_score_thresh": 15, "max_cands_per_query": 1500},
        {"label": "similar k=5, T=13 (default)",
         "kmer_score_thresh": 13, "max_cands_per_query": 1500},
        {"label": "similar k=5, T=11 (sensitive)",
         "kmer_score_thresh": 11, "max_cands_per_query": 1500},
        {"label": "similar k=5, T=9 (very sensitive)",
         "kmer_score_thresh": 9, "max_cands_per_query": 1500},

        # With more candidates at good thresholds
        {"label": "similar k=5, T=13, mc=2000",
         "kmer_score_thresh": 13, "max_cands_per_query": 2000},
        {"label": "similar k=5, T=11, mc=2000",
         "kmer_score_thresh": 11, "max_cands_per_query": 2000},
        {"label": "similar k=5, T=11, mc=3000",
         "kmer_score_thresh": 11, "max_cands_per_query": 3000},
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

    with open(out_dir / "similar_kmer_pilot.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 130)
    print("SUMMARY (sorted by ROC1)")
    print("=" * 130)
    print(f"{'Config':60s} {'ROC1':>7s} {'MAP':>7s} {'Aligned':>10s} {'Time':>8s}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: -x["mean_roc1"]):
        print(
            f"{r['label']:60s} {r['mean_roc1']:7.4f} {r['mean_map']:7.4f} "
            f"{r['num_aligned']:10d} {r['time']:8.1f}"
        )

    print("\nReference (from previous benchmarks):")
    print("  ClustKIT k=3 (very-sensitive):  ROC1=0.7360  Time=~70s")
    print("  MMseqs2 (s=4):                  ROC1=0.7311  Time=~8s")
    print("  MMseqs2 (s=7.5):                ROC1=0.7942  Time=~14s")
    print("  DIAMOND (very-sensitive):        ROC1=0.7963  Time=~13s")


if __name__ == "__main__":
    main()
