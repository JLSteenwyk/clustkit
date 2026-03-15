#!/usr/bin/env python3
"""Pilot: optimized spaced seed patterns from literature.

Tests weight-4 seeds with longer spans (7-8) that complement
the existing 11011 (span=5) and 110011 (span=6).
"""
import json, random, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.benchmark_scop_search import (
    classify_hit, compute_roc_n, read_fasta, write_fasta, RankedHit,
)
import numba; numba.set_num_threads(4)
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
    for qid in query_sids:
        q_info = domain_info.get(qid)
        if q_info is None: continue
        fam_key = str(q_info["family"])
        total_tp = family_sizes.get(fam_key, 1) - 1
        if total_tp <= 0: continue
        query_hits = hits_by_query.get(qid, [])
        query_hits.sort(key=lambda x: -x[1])
        ranked = []
        for tid, score in query_hits:
            label = classify_hit(qid, tid, domain_info)
            if label != "IGNORE":
                ranked.append(RankedHit(target_id=tid, score=score, label=label))
        roc1_values.append(compute_roc_n(ranked, 1, total_tp))
    n = len(roc1_values)
    return {"n_queries": n, "mean_roc1": float(np.mean(roc1_values)) if n else 0}

def run_config(db_index, query_ds, metadata, label, config_num, total_configs,
               elapsed_times, **kwargs):
    print(f"\n>>> Config {config_num}/{total_configs}: {label}", flush=True)
    if elapsed_times:
        avg_time = sum(elapsed_times) / len(elapsed_times)
        remaining = (total_configs - config_num + 1) * avg_time
        print(f"    ETA: ~{remaining/60:.1f} min remaining", flush=True)
    t0 = time.perf_counter()
    results = search_kmer_index(db_index, query_ds, threshold=0.1, top_k=500, **kwargs)
    elapsed = time.perf_counter() - t0
    hits = []
    for qhits in results.hits:
        for h in qhits:
            hits.append((h.query_id, h.target_id, h.score if h.score != 0 else h.identity))
    metrics = evaluate_hits(hits, metadata)
    print(f"  RESULT: {label}\n"
          f"          ROC1={metrics['mean_roc1']:.4f}  aligned={results.num_aligned:>10d}  "
          f"time={elapsed:.1f}s", flush=True)
    return {**metrics, "time": elapsed, "aligned": results.num_aligned, "label": label}

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

    print("Loading database index...", flush=True)
    db_index = load_database(out_dir / "clustkit_db")
    query_ds = read_sequences(query_fasta, "protein")
    print(f"Loaded {query_ds.num_sequences} queries\n", flush=True)

    results = []
    base = dict(freq_percentile=99.5, phase_a_topk=200000, kmer_score_thresh=0,
                local_alignment=True, evalue_normalize=False, max_cands_per_query=8000,
                reduced_alphabet=True, reduced_k=[4, 5], use_idf=True)

    print("=" * 120)
    print("New spaced seed patterns (added to triple index baseline)")
    print("=" * 120)

    configs = [
        # New seed patterns (each tested individually on top of triple index)
        {"label": "+ spaced 101011 (w=4, span=6)",
         **base, "spaced_seeds": ["101011"]},
        {"label": "+ spaced 1010101 (w=4, span=7)",
         **base, "spaced_seeds": ["1010101"]},
        {"label": "+ spaced 1001011 (w=4, span=7)",
         **base, "spaced_seeds": ["1001011"]},
        {"label": "+ spaced 11000011 (w=4, span=8)",
         **base, "spaced_seeds": ["11000011"]},

        # Best new seeds combined with our existing best seeds
        {"label": "+ all seeds [11011,110011,1010101,11000011]",
         **base, "spaced_seeds": ["11011", "110011", "1010101", "11000011"]},
    ]

    total = len(configs)
    elapsed_times = []
    print(f"\nTotal configs: {total}\n", flush=True)

    for i, cfg in enumerate(configs, 1):
        label = cfg.pop("label")
        try:
            r = run_config(db_index, query_ds, metadata, label,
                           config_num=i, total_configs=total,
                           elapsed_times=elapsed_times, **cfg)
            results.append(r)
            elapsed_times.append(r["time"])
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}", flush=True)
            traceback.print_exc()
            elapsed_times.append(600)

    with open(out_dir / "seeds2_pilot.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 120)
    print("SUMMARY (sorted by ROC1)")
    print("=" * 120)
    for r in sorted(results, key=lambda x: -x["mean_roc1"]):
        print(f"  {r['label']:55s} ROC1={r['mean_roc1']:.4f} aligned={r['aligned']:>10d} time={r['time']:.1f}s")
    print(f"\n  Reference: Triple+IDF+spaced[11011,110011] mc=8K = 0.8123")
    print(f"  Previous best: Triple+IDF+spaced[11011,110011] mc=15K = 0.8178")

if __name__ == "__main__":
    main()
