"""Pilot: test reducing max_cands_per_query to speed up search.

Tests mc=8000 (current), 6000, 4000, 3000, 2000, 1000 on the 2K query subset.
Uses the same evaluation framework as benchmark_full_scope.py.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.benchmark_scop_search import (
    classify_hit, compute_roc_n, RankedHit,
)

SCOP_DIR = Path("benchmarks/data/scop_search_results")
THREADS = 8


def evaluate(hits_by_query, metadata):
    """Compute ROC1 from hits (same as benchmark_full_scope)."""
    di = metadata["domain_info"]
    fs = metadata["family_sizes_in_db"]
    roc1_vals = []
    for qid in metadata["query_sids"]:
        qi = di.get(qid)
        if qi is None:
            continue
        total_tp = fs.get(str(qi["family"]), 1) - 1
        if total_tp <= 0:
            continue
        query_hits = sorted(hits_by_query.get(qid, []), key=lambda x: -x[1])
        ranked = [RankedHit(target_id=t, score=s, label=classify_hit(qid, t, di))
                  for t, s in query_hits if classify_hit(qid, t, di) != "IGNORE"]
        roc1_vals.append(compute_roc_n(ranked, 1, total_tp))
    return float(np.mean(roc1_vals)) if roc1_vals else 0.0


def run_test(mc, metadata, db, query_ds):
    """Run search with given mc value."""
    from clustkit.kmer_index import search_kmer_index

    start = time.perf_counter()
    results = search_kmer_index(
        db, query_ds, threshold=0.1, top_k=500,
        freq_percentile=99.5, phase_a_topk=200000, kmer_score_thresh=0,
        local_alignment=True, evalue_normalize=False,
        reduced_alphabet=True, reduced_k=5,
        spaced_seeds=["110011"],
        use_c_scoring=True, use_c_sw=True, c_sw_band_width=50,
        max_cands_per_query=mc,
    )
    elapsed = time.perf_counter() - start

    # Collect hits — query_id and target_id are already string domain IDs
    hits = defaultdict(list)
    for qh in results.hits:
        for h in qh:
            score = h.score if h.score != 0 else h.identity
            hits[h.query_id].append((h.target_id, score))

    roc1 = evaluate(hits, metadata)
    return round(roc1, 4), elapsed


if __name__ == "__main__":
    numba.set_num_threads(THREADS)

    with open(SCOP_DIR / "metadata.json") as f:
        metadata = json.load(f)

    print("Loading database...", flush=True)
    from clustkit.database import load_database
    db = load_database(str(Path("benchmarks/data/speed_sensitivity_results/clustkit_db_v3")))
    print("  Loaded database index", flush=True)

    from clustkit.io import read_sequences

    # Write 2K subset query FASTA
    n_queries = 2000
    subset_fasta = Path("/tmp/queries_2k.fasta")
    count = 0
    with open(SCOP_DIR / "queries.fasta") as fin, open(subset_fasta, "w") as fout:
        writing = True
        for line in fin:
            if line.startswith(">"):
                count += 1
                writing = count <= n_queries
            if writing:
                fout.write(line)

    query_ds = read_sequences(str(subset_fasta), mode="protein")
    metadata["query_sids"] = metadata["query_sids"][:n_queries]

    print(f"  Using {n_queries} queries\n", flush=True)

    print(f"{'mc':>6} {'ROC1':>8} {'Time':>8} {'Speedup':>8}")
    print("-" * 36)

    baseline_time = None
    for mc in [8000, 6000, 4000, 3000, 2000, 1000]:
        roc1, elapsed = run_test(mc, metadata, db, query_ds)
        if baseline_time is None:
            baseline_time = elapsed
        speedup = baseline_time / elapsed
        print(f"{mc:>6} {roc1:>8.4f} {elapsed:>7.1f}s {speedup:>7.2f}x", flush=True)
        import gc; gc.collect()
