#!/usr/bin/env python3
"""Integrated C SW: search_kmer_index with use_c_sw=True.

Uses the proven search_kmer_index pipeline (ROC1=0.808) with C SW
acceleration. Compares Numba SW (bw=126) vs C SW (bw=20/50).
"""

import json, random, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.benchmark_scop_search import (
    classify_hit, compute_roc_n, read_fasta, write_fasta, RankedHit,
)
import numba; numba.set_num_threads(8)
from clustkit.database import load_database
from clustkit.io import read_sequences
from clustkit.kmer_index import search_kmer_index


def evaluate_roc1(results, metadata):
    di = metadata["domain_info"]; fs = metadata["family_sizes_in_db"]
    qs = set(metadata["query_sids"])
    hbq = defaultdict(list)
    for qh in results.hits:
        for h in qh:
            hbq[h.query_id].append((h.target_id, h.score if h.score != 0 else h.identity))
    vals = []
    for qid in qs:
        qi = di.get(qid)
        if qi is None: continue
        tp = fs.get(str(qi["family"]), 1) - 1
        if tp <= 0: continue
        qh = sorted(hbq.get(qid, []), key=lambda x: -x[1])
        ranked = [RankedHit(target_id=t, score=s, label=classify_hit(qid, t, di))
                  for t, s in qh if classify_hit(qid, t, di) != "IGNORE"]
        vals.append(compute_roc_n(ranked, 1, tp))
    return float(np.mean(vals)) if vals else 0.0


def main():
    scop_dir = Path("benchmarks/data/scop_search_results")
    out_dir = Path("benchmarks/data/speed_sensitivity_results")
    with open(scop_dir / "metadata.json") as f:
        fm = json.load(f)
    random.seed(42)
    qsids = sorted(random.sample(fm["query_sids"], min(2000, len(fm["query_sids"]))))
    all_seqs = dict(read_fasta(scop_dir / "queries.fasta"))
    sub = [(s, all_seqs[s]) for s in qsids if s in all_seqs]
    qf = str(out_dir / "queries_subset.fasta")
    write_fasta(sub, qf)
    metadata = dict(fm); metadata["query_sids"] = qsids

    print("Loading database...", flush=True)
    db_index = load_database(out_dir / "clustkit_db")
    query_ds = read_sequences(qf, "protein")
    print(f"Loaded {query_ds.num_sequences} queries, {db_index.dataset.num_sequences} db\n", flush=True)

    base = dict(
        freq_percentile=99.5, phase_a_topk=200000, kmer_score_thresh=0,
        local_alignment=True, evalue_normalize=False, max_cands_per_query=8000,
    )

    configs = [
        # 3-index: Numba SW bw=126 (baseline)
        ("3idx Numba SW bw=126",
         {**base, "reduced_alphabet": True, "reduced_k": 5,
          "spaced_seeds": ["110011"], "use_c_sw": False}),

        # 3-index: C SW bw=50
        ("3idx C SW bw=50",
         {**base, "reduced_alphabet": True, "reduced_k": 5,
          "spaced_seeds": ["110011"], "use_c_sw": True, "c_sw_band_width": 50}),

        # 3-index: C SW bw=20
        ("3idx C SW bw=20",
         {**base, "reduced_alphabet": True, "reduced_k": 5,
          "spaced_seeds": ["110011"], "use_c_sw": True, "c_sw_band_width": 20}),

        # 5-index: Numba SW bw=126 (max sensitivity baseline)
        ("5idx Numba SW bw=126",
         {**base, "reduced_alphabet": True, "reduced_k": [4, 5],
          "spaced_seeds": ["11011", "110011"], "use_c_sw": False}),

        # 5-index: C SW bw=50
        ("5idx C SW bw=50",
         {**base, "reduced_alphabet": True, "reduced_k": [4, 5],
          "spaced_seeds": ["11011", "110011"], "use_c_sw": True, "c_sw_band_width": 50}),

        # 5-index: C SW bw=20
        ("5idx C SW bw=20",
         {**base, "reduced_alphabet": True, "reduced_k": [4, 5],
          "spaced_seeds": ["11011", "110011"], "use_c_sw": True, "c_sw_band_width": 20}),
    ]

    print("=" * 90)
    print("Integrated C SW: search_kmer_index with C acceleration")
    print("=" * 90, flush=True)

    elapsed_times = []
    for i, (label, kwargs) in enumerate(configs, 1):
        if elapsed_times:
            avg = sum(elapsed_times) / len(elapsed_times)
            remaining = (len(configs) - i + 1) * avg
            print(f"\n  ETA: ~{remaining/60:.0f} min remaining", flush=True)

        print(f"\n  >>> {label}", flush=True)
        t0 = time.perf_counter()
        results = search_kmer_index(
            db_index, query_ds, threshold=0.1, top_k=500, **kwargs)
        elapsed = time.perf_counter() - t0
        roc1 = evaluate_roc1(results, metadata)
        vs = roc1 - 0.7942
        print(f"      Time: {elapsed:.1f}s  Aligned: {results.num_aligned}  "
              f"ROC1: {roc1:.4f} (vs MMseqs2: {vs:+.4f})", flush=True)
        elapsed_times.append(elapsed)

    print(f"\n  Reference: MMseqs2=0.7942 (14s)  DIAMOND=0.7963 (13s)")


if __name__ == "__main__":
    main()
