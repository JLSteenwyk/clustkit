#!/usr/bin/env python3
"""Pilot: reduced alphabet k=4 with mc=15000."""
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

    print(">>> Dual index: std k=3 + reduced k=4 (mc=15000)", flush=True)
    t0 = time.perf_counter()
    results = search_kmer_index(
        db_index, query_ds, threshold=0.1, top_k=500,
        freq_percentile=99.5, phase_a_topk=200000, kmer_score_thresh=0,
        local_alignment=True, evalue_normalize=False,
        max_cands_per_query=15000, reduced_alphabet=True, reduced_k=4,
    )
    elapsed = time.perf_counter() - t0
    hits = []
    for qhits in results.hits:
        for h in qhits:
            hits.append((h.query_id, h.target_id, h.score if h.score != 0 else h.identity))
    metrics = evaluate_hits(hits, metadata)
    print(f"\nRESULT: reduced k=4 mc=15K  ROC1={metrics['mean_roc1']:.4f}  "
          f"aligned={results.num_aligned}  time={elapsed:.1f}s", flush=True)
    print(f"\nReference: MMseqs2=0.7942  DIAMOND=0.7963  reduced k=4 mc=8K=0.7949", flush=True)

if __name__ == "__main__":
    main()
