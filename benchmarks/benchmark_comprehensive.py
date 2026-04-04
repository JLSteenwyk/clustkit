#!/usr/bin/env python3
"""Comprehensive benchmark: ClustKIT v8.1 vs MMseqs2 vs DIAMOND vs BLAST.

Runs ClustKIT at multiple speed/sensitivity configurations alongside
existing results from other tools on the same 2K SCOPe query subset.
"""

import json, random, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.benchmark_scop_search import (
    classify_hit, compute_roc_n, compute_average_precision,
    read_fasta, write_fasta, RankedHit,
)
import numba; numba.set_num_threads(8)
from clustkit.database import load_database
from clustkit.io import read_sequences
from clustkit.kmer_index import search_kmer_index


def evaluate(results, metadata):
    di = metadata["domain_info"]; fs = metadata["family_sizes_in_db"]
    qs = set(metadata["query_sids"])
    hbq = defaultdict(list)
    for qh in results.hits:
        for h in qh:
            hbq[h.query_id].append((h.target_id, h.score if h.score != 0 else h.identity))
    roc1_vals, roc5_vals, map_vals = [], [], []
    for qid in qs:
        qi = di.get(qid)
        if qi is None: continue
        fam_key = str(qi["family"])
        total_tp = fs.get(fam_key, 1) - 1
        if total_tp <= 0: continue
        query_hits = sorted(hbq.get(qid, []), key=lambda x: -x[1])
        ranked = [RankedHit(target_id=t, score=s, label=classify_hit(qid, t, di))
                  for t, s in query_hits if classify_hit(qid, t, di) != "IGNORE"]
        roc1_vals.append(compute_roc_n(ranked, 1, total_tp))
        roc5_vals.append(compute_roc_n(ranked, 5, total_tp))
        map_vals.append(compute_average_precision(ranked, total_tp))
    n = len(roc1_vals)
    return {
        "roc1": float(np.mean(roc1_vals)) if n else 0,
        "roc5": float(np.mean(roc5_vals)) if n else 0,
        "map": float(np.mean(map_vals)) if n else 0,
        "n_queries": n,
    }


def main():
    scop_dir = Path("benchmarks/data/scop_search_results")
    out_dir = Path("benchmarks/data/speed_sensitivity_results")
    with open(scop_dir / "metadata.json") as f: fm = json.load(f)
    random.seed(42)
    qsids = sorted(random.sample(fm["query_sids"], min(2000, len(fm["query_sids"]))))
    all_seqs = dict(read_fasta(scop_dir / "queries.fasta"))
    sub = [(s, all_seqs[s]) for s in qsids if s in all_seqs]
    qf = str(out_dir / "queries_subset.fasta")
    write_fasta(sub, qf)
    metadata = dict(fm); metadata["query_sids"] = qsids

    print("Loading database...", flush=True)
    db = load_database(out_dir / "clustkit_db_v3")
    qds = read_sequences(qf, "protein")
    print(f"Loaded {qds.num_sequences} queries, {db.dataset.num_sequences} db\n", flush=True)

    # ── ClustKIT configurations ──────────────────────────────────────
    base = dict(freq_percentile=99.5, phase_a_topk=200000, kmer_score_thresh=0,
                local_alignment=True, evalue_normalize=False)

    clustkit_configs = [
        # Speed tiers
        ("ClustKIT v8.1 (fast, 1idx C bw=20)",
         dict(**base, max_cands_per_query=8000, reduced_alphabet=False,
              use_c_scoring=True, use_c_sw=True, c_sw_band_width=20)),
        ("ClustKIT v8.1 (default, 3idx C bw=50)",
         dict(**base, max_cands_per_query=8000, reduced_alphabet=True, reduced_k=5,
              spaced_seeds=["110011"], use_c_scoring=True, use_c_sw=True, c_sw_band_width=50)),
        ("ClustKIT v8.1 (sensitive, 3idx Numba bw=126)",
         dict(**base, max_cands_per_query=8000, reduced_alphabet=True, reduced_k=5,
              spaced_seeds=["110011"], use_c_scoring=False, use_c_sw=False)),
    ]

    results_all = []

    print("=" * 100)
    print("ClustKIT v8.1 Configurations")
    print("=" * 100, flush=True)

    for label, kwargs in clustkit_configs:
        print(f"\n  >>> {label}", flush=True)
        t0 = time.perf_counter()
        results = search_kmer_index(db, qds, threshold=0.1, top_k=500, **kwargs)
        elapsed = time.perf_counter() - t0
        metrics = evaluate(results, metadata)
        print(f"      ROC1={metrics['roc1']:.4f}  ROC5={metrics['roc5']:.4f}  "
              f"MAP={metrics['map']:.4f}  Time={elapsed:.1f}s", flush=True)
        results_all.append({
            "tool": label, "roc1": metrics["roc1"], "roc5": metrics["roc5"],
            "map": metrics["map"], "time": elapsed, "aligned": results.num_aligned,
        })

    # ── Load existing competitor results ─────────────────────────────
    print("\n" + "=" * 100)
    print("Competitor Results (from previous benchmarks on same 2K subset)")
    print("=" * 100, flush=True)

    with open(out_dir / "speed_sensitivity_results.json") as f:
        existing = json.load(f)

    for r in existing:
        label = r.get("label", "?")
        roc1 = r.get("roc1", r.get("mean_roc1", 0))
        t = r.get("time", r.get("search_time", 0))
        results_all.append({
            "tool": label, "roc1": float(roc1), "roc5": 0, "map": 0,
            "time": float(t), "aligned": 0,
        })

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("COMPREHENSIVE BENCHMARK RESULTS (2K queries, 591K db, 8 threads)")
    print("=" * 100, flush=True)
    print(f"  {'Tool':50s} {'ROC1':>7s} {'Time':>8s} {'vs MMseqs2':>10s}", flush=True)
    print("  " + "-" * 80, flush=True)

    mmseqs_roc1 = 0.7942
    for r in sorted(results_all, key=lambda x: -x["roc1"]):
        vs = r["roc1"] - mmseqs_roc1
        marker = " ***" if "v8.1" in r["tool"] else ""
        print(f"  {r['tool']:50s} {r['roc1']:7.4f} {r['time']:7.1f}s {vs:+10.4f}{marker}",
              flush=True)

    # ── Speed-sensitivity Pareto ─────────────────────────────────────
    print("\n" + "=" * 100)
    print("SPEED-SENSITIVITY PARETO FRONTIER")
    print("=" * 100, flush=True)

    # Find Pareto-optimal points (no other point is both faster AND more sensitive)
    sorted_by_time = sorted(results_all, key=lambda x: x["time"])
    pareto = []
    best_roc1 = -1
    for r in sorted_by_time:
        if r["roc1"] > best_roc1:
            pareto.append(r)
            best_roc1 = r["roc1"]

    print(f"  {'Tool':50s} {'ROC1':>7s} {'Time':>8s}", flush=True)
    print("  " + "-" * 70, flush=True)
    for r in pareto:
        marker = " ***" if "v8.1" in r["tool"] else ""
        print(f"  {r['tool']:50s} {r['roc1']:7.4f} {r['time']:7.1f}s{marker}", flush=True)

    # Save
    with open(out_dir / "comprehensive_benchmark.json", "w") as f:
        json.dump(results_all, f, indent=2)

    print(f"\n  Results saved to {out_dir / 'comprehensive_benchmark.json'}")
    print(f"\n  *** = ClustKIT v8.1 configurations")


if __name__ == "__main__":
    main()
