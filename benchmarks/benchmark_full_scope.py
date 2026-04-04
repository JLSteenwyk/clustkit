#!/usr/bin/env python3
"""Full SCOPe benchmark: ClustKIT v8.1 vs BLAST vs MMseqs2 vs DIAMOND vs LAST vs SWIPE.

Runs all tools on the full 10K SCOPe query set (not 2K subset).
This is the standard benchmark used in the MMseqs2 and DIAMOND papers.
"""

import json
import subprocess
import sys
import time
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.benchmark_scop_search import (
    classify_hit, compute_roc_n, compute_average_precision,
    read_fasta, write_fasta, RankedHit,
)

# Tool paths
BLAST = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/ncbi-blast-2.17.0+/bin"
MMSEQS = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/mmseqs/bin/mmseqs"
DIAMOND = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/diamond-linux64/diamond"
LAST = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/last-1609/bin"
SWIPE = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/swipe"

THREADS = 8


def evaluate(hits_by_query, metadata):
    """Compute ROC1, ROC5, MAP from hits."""
    di = metadata["domain_info"]; fs = metadata["family_sizes_in_db"]
    roc1_vals, roc5_vals, map_vals = [], [], []
    for qid in metadata["query_sids"]:
        qi = di.get(qid)
        if qi is None: continue
        total_tp = fs.get(str(qi["family"]), 1) - 1
        if total_tp <= 0: continue
        query_hits = sorted(hits_by_query.get(qid, []), key=lambda x: -x[1])
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


def run_mmseqs(db_fasta, query_fasta, sensitivity, tmp_dir):
    """Run MMseqs2 search at given sensitivity."""
    label = f"MMseqs2 (s={sensitivity})"
    print(f"  Running {label}...", flush=True)

    db_dir = Path(tmp_dir) / f"mmseqs_db_s{sensitivity}"
    db_dir.mkdir(exist_ok=True)

    # Create database
    subprocess.run([MMSEQS, "createdb", str(db_fasta), str(db_dir / "targetDB")],
                   capture_output=True)
    subprocess.run([MMSEQS, "createdb", str(query_fasta), str(db_dir / "queryDB")],
                   capture_output=True)

    t0 = time.perf_counter()
    subprocess.run([
        MMSEQS, "search",
        str(db_dir / "queryDB"), str(db_dir / "targetDB"),
        str(db_dir / "resultDB"), str(db_dir / "tmp"),
        "-s", str(sensitivity),
        "--threads", str(THREADS),
        "--max-seqs", "500",
    ], capture_output=True)
    elapsed = time.perf_counter() - t0

    # Convert to tab format
    subprocess.run([
        MMSEQS, "convertalis",
        str(db_dir / "queryDB"), str(db_dir / "targetDB"),
        str(db_dir / "resultDB"), str(db_dir / "results.m8"),
        "--format-output", "query,target,evalue,bits",
    ], capture_output=True)

    # Parse results
    hits = defaultdict(list)
    results_file = db_dir / "results.m8"
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    qid, tid = parts[0], parts[1]
                    if qid != tid:
                        score = float(parts[3])  # bit score
                        hits[qid].append((tid, score))

    return label, hits, elapsed


def run_diamond(db_fasta, query_fasta, mode, tmp_dir):
    """Run DIAMOND at given sensitivity mode."""
    label = f"DIAMOND ({mode})"
    print(f"  Running {label}...", flush=True)

    db_dir = Path(tmp_dir) / f"diamond_{mode}"
    db_dir.mkdir(exist_ok=True)
    db_path = str(db_dir / "targetDB")

    subprocess.run([DIAMOND, "makedb", "--in", str(db_fasta), "-d", db_path],
                   capture_output=True)

    out_file = str(db_dir / "results.m8")
    cmd = [
        DIAMOND, "blastp",
        "-d", db_path,
        "-q", str(query_fasta),
        "-o", out_file,
        "--threads", str(THREADS),
        "--max-target-seqs", "500",
        "--outfmt", "6", "qseqid", "sseqid", "evalue", "bitscore",
    ]
    if mode != "default":
        cmd.extend(["--" + mode])

    t0 = time.perf_counter()
    subprocess.run(cmd, capture_output=True)
    elapsed = time.perf_counter() - t0

    hits = defaultdict(list)
    if Path(out_file).exists():
        with open(out_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    qid, tid = parts[0], parts[1]
                    if qid != tid:
                        hits[qid].append((tid, float(parts[3])))

    return label, hits, elapsed


def run_blast(db_fasta, query_fasta, tmp_dir):
    """Run BLAST search."""
    label = "BLAST"
    print(f"  Running {label}...", flush=True)

    db_dir = Path(tmp_dir) / "blast"
    db_dir.mkdir(exist_ok=True)
    db_path = str(db_dir / "targetDB")

    subprocess.run([f"{BLAST}/makeblastdb", "-in", str(db_fasta),
                    "-dbtype", "prot", "-out", db_path], capture_output=True)

    out_file = str(db_dir / "results.m8")
    t0 = time.perf_counter()
    subprocess.run([
        f"{BLAST}/blastp",
        "-db", db_path,
        "-query", str(query_fasta),
        "-out", out_file,
        "-outfmt", "6 qseqid sseqid evalue bitscore",
        "-num_threads", str(THREADS),
        "-max_target_seqs", "500",
        "-evalue", "10",
    ], capture_output=True)
    elapsed = time.perf_counter() - t0

    hits = defaultdict(list)
    if Path(out_file).exists():
        with open(out_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    qid, tid = parts[0], parts[1]
                    if qid != tid:
                        hits[qid].append((tid, float(parts[3])))

    return label, hits, elapsed


def run_last(db_fasta, query_fasta, tmp_dir):
    """Run LAST search."""
    label = "LAST"
    print(f"  Running {label}...", flush=True)

    db_dir = Path(tmp_dir) / "last"
    db_dir.mkdir(exist_ok=True)
    db_path = str(db_dir / "targetDB")

    subprocess.run([f"{LAST}/lastdb", "-p", db_path, str(db_fasta)],
                   capture_output=True)

    out_file = str(db_dir / "results.maf")
    t0 = time.perf_counter()
    result = subprocess.run(
        [f"{LAST}/lastal", "-P", str(THREADS), "-f", "BlastTab", db_path, str(query_fasta)],
        capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    hits = defaultdict(list)
    for line in result.stdout.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) >= 12:
            qid, tid = parts[0], parts[1]
            if qid != tid:
                hits[qid].append((tid, float(parts[11])))  # bit score

    return label, hits, elapsed


def run_swipe(db_fasta, query_fasta, tmp_dir):
    """Run SWIPE search."""
    label = "SWIPE"
    print(f"  Running {label}...", flush=True)

    db_dir = Path(tmp_dir) / "swipe"
    db_dir.mkdir(exist_ok=True)
    db_path = str(db_dir / "targetDB")

    # SWIPE uses BLAST database format
    subprocess.run([f"{BLAST}/makeblastdb", "-in", str(db_fasta),
                    "-dbtype", "prot", "-out", db_path], capture_output=True)

    out_file = str(db_dir / "results.m8")
    t0 = time.perf_counter()
    result = subprocess.run([
        SWIPE,
        "-d", db_path,
        "-i", str(query_fasta),
        "-a", str(THREADS),
        "-v", "500",
        "-b", "500",
        "-m", "8",
    ], capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    hits = defaultdict(list)
    for line in result.stdout.split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) >= 12:
            qid, tid = parts[0], parts[1]
            if qid != tid:
                hits[qid].append((tid, float(parts[11])))

    return label, hits, elapsed


def run_clustkit(query_fasta, metadata, config_name, **kwargs):
    """Run ClustKIT search."""
    label = f"ClustKIT ({config_name})"
    print(f"  Running {label}...", flush=True)

    import numba; numba.set_num_threads(THREADS)
    from clustkit.database import load_database
    from clustkit.io import read_sequences
    from clustkit.kmer_index import search_kmer_index

    out_dir = Path("benchmarks/data/speed_sensitivity_results")
    db = load_database(out_dir / "clustkit_db_v3")
    qds = read_sequences(query_fasta, "protein")

    t0 = time.perf_counter()
    results = search_kmer_index(db, qds, threshold=0.1, top_k=500, **kwargs)
    elapsed = time.perf_counter() - t0

    hits = defaultdict(list)
    for qh in results.hits:
        for h in qh:
            hits[h.query_id].append((h.target_id, h.score if h.score != 0 else h.identity))

    return label, hits, elapsed


def main():
    scop_dir = Path("benchmarks/data/scop_search_results")
    out_dir = Path("benchmarks/data/speed_sensitivity_results")

    with open(scop_dir / "metadata.json") as f:
        metadata = json.load(f)

    db_fasta = scop_dir / "database.fasta"
    query_fasta = scop_dir / "queries.fasta"

    # Use full query set
    n_queries = len(metadata["query_sids"])
    print(f"Full SCOPe benchmark: {n_queries} queries, 591K database, {THREADS} threads\n", flush=True)

    results_all = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        print("=" * 90)
        print("Running all tools")
        print("=" * 90, flush=True)

        # ClustKIT configs
        base = dict(freq_percentile=99.5, phase_a_topk=200000, kmer_score_thresh=0,
                    local_alignment=True, evalue_normalize=False, max_cands_per_query=8000)

        for config_name, kwargs in [
            ("v8.1 fast", dict(**base, reduced_alphabet=False,
                              use_c_scoring=True, use_c_sw=True, c_sw_band_width=20)),
            ("v8.1 default", dict(**base, reduced_alphabet=True, reduced_k=5,
                                  spaced_seeds=["110011"], use_c_scoring=True,
                                  use_c_sw=True, c_sw_band_width=50)),
            ("v8.1 sensitive", dict(**base, reduced_alphabet=True, reduced_k=5,
                                    spaced_seeds=["110011"], use_c_scoring=False,
                                    use_c_sw=False)),
        ]:
            label, hits, elapsed = run_clustkit(str(query_fasta), metadata, config_name, **kwargs)
            metrics = evaluate(hits, metadata)
            print(f"    {label}: ROC1={metrics['roc1']:.4f} ROC5={metrics['roc5']:.4f} "
                  f"MAP={metrics['map']:.4f} Time={elapsed:.1f}s", flush=True)
            results_all.append({"tool": label, **metrics, "time": elapsed})

        # MMseqs2 at multiple sensitivities
        for s in [1, 4, 5.7, 7.5]:
            label, hits, elapsed = run_mmseqs(db_fasta, query_fasta, s, tmp_dir)
            metrics = evaluate(hits, metadata)
            print(f"    {label}: ROC1={metrics['roc1']:.4f} ROC5={metrics['roc5']:.4f} "
                  f"MAP={metrics['map']:.4f} Time={elapsed:.1f}s", flush=True)
            results_all.append({"tool": label, **metrics, "time": elapsed})

        # DIAMOND at multiple sensitivities
        for mode in ["fast", "default", "sensitive", "very-sensitive", "ultra-sensitive"]:
            label, hits, elapsed = run_diamond(db_fasta, query_fasta, mode, tmp_dir)
            metrics = evaluate(hits, metadata)
            print(f"    {label}: ROC1={metrics['roc1']:.4f} ROC5={metrics['roc5']:.4f} "
                  f"MAP={metrics['map']:.4f} Time={elapsed:.1f}s", flush=True)
            results_all.append({"tool": label, **metrics, "time": elapsed})

        # BLAST
        label, hits, elapsed = run_blast(db_fasta, query_fasta, tmp_dir)
        metrics = evaluate(hits, metadata)
        print(f"    {label}: ROC1={metrics['roc1']:.4f} ROC5={metrics['roc5']:.4f} "
              f"MAP={metrics['map']:.4f} Time={elapsed:.1f}s", flush=True)
        results_all.append({"tool": label, **metrics, "time": elapsed})

        # LAST
        label, hits, elapsed = run_last(db_fasta, query_fasta, tmp_dir)
        metrics = evaluate(hits, metadata)
        print(f"    {label}: ROC1={metrics['roc1']:.4f} ROC5={metrics['roc5']:.4f} "
              f"MAP={metrics['map']:.4f} Time={elapsed:.1f}s", flush=True)
        results_all.append({"tool": label, **metrics, "time": elapsed})

        # SWIPE
        label, hits, elapsed = run_swipe(db_fasta, query_fasta, tmp_dir)
        metrics = evaluate(hits, metadata)
        print(f"    {label}: ROC1={metrics['roc1']:.4f} ROC5={metrics['roc5']:.4f} "
              f"MAP={metrics['map']:.4f} Time={elapsed:.1f}s", flush=True)
        results_all.append({"tool": label, **metrics, "time": elapsed})

    # Summary
    print("\n" + "=" * 90)
    print("RESULTS (sorted by ROC1)")
    print("=" * 90, flush=True)
    print(f"  {'Tool':45s} {'ROC1':>7s} {'ROC5':>7s} {'MAP':>7s} {'Time':>8s}", flush=True)
    print("  " + "-" * 80, flush=True)
    for r in sorted(results_all, key=lambda x: -x["roc1"]):
        print(f"  {r['tool']:45s} {r['roc1']:7.4f} {r['roc5']:7.4f} "
              f"{r['map']:7.4f} {r['time']:7.1f}s", flush=True)

    # Save
    with open(out_dir / "full_scope_benchmark.json", "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\n  Saved to {out_dir / 'full_scope_benchmark.json'}")


if __name__ == "__main__":
    main()
