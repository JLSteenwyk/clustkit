#!/usr/bin/env python3
"""Time search_kmer_index directly to establish the true v7.4 baseline.

The hybrid pilot got ROC1=0.791 using standalone scoring functions.
This pilot calls search_kmer_index (which gave ROC1=0.802) and
precisely measures each stage to identify where C extensions help most.
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

    configs = [
        ("5idx IDF spaced[11011,110011] full align",
         dict(freq_percentile=99.5, phase_a_topk=200000, kmer_score_thresh=0,
              local_alignment=True, evalue_normalize=False, max_cands_per_query=8000,
              reduced_alphabet=True, reduced_k=[4, 5], use_idf=True,
              spaced_seeds=["11011", "110011"])),
        ("5idx no-IDF spaced[11011,110011] full align",
         dict(freq_percentile=99.5, phase_a_topk=200000, kmer_score_thresh=0,
              local_alignment=True, evalue_normalize=False, max_cands_per_query=8000,
              reduced_alphabet=True, reduced_k=[4, 5], use_idf=False,
              spaced_seeds=["11011", "110011"])),
        ("3idx no-IDF spaced[110011] full align",
         dict(freq_percentile=99.5, phase_a_topk=200000, kmer_score_thresh=0,
              local_alignment=True, evalue_normalize=False, max_cands_per_query=8000,
              reduced_alphabet=True, reduced_k=5, use_idf=False,
              spaced_seeds=["110011"])),
    ]

    print("=" * 90)
    print("search_kmer_index: true ROC1 baseline with full alignment")
    print("=" * 90, flush=True)

    for label, kwargs in configs:
        print(f"\n  {label}", flush=True)
        t0 = time.perf_counter()
        results = search_kmer_index(
            db_index, query_ds, threshold=0.1, top_k=500, **kwargs)
        elapsed = time.perf_counter() - t0
        roc1 = evaluate_roc1(results, metadata)
        vs = roc1 - 0.7942
        print(f"    Time: {elapsed:.1f}s  Aligned: {results.num_aligned}  "
              f"ROC1: {roc1:.4f} (vs MMseqs2: {vs:+.4f})", flush=True)

    print(f"\n  Reference: MMseqs2=0.7942 (14s)  DIAMOND=0.7963 (13s)")


if __name__ == "__main__":
    main()
