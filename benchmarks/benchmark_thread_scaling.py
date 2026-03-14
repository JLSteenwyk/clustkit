"""Thread scaling benchmark for ClustKIT.

Fixed Pfam dataset, varying thread count from 1 to max.
Also measures MMseqs2 scaling for comparison.

C5 from the publication plan.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.pipeline import run_pipeline

# Import data loading from the Pfam benchmark
from benchmark_pfam_concordance import load_and_mix_families


DATA_DIR = Path(__file__).resolve().parent / "data" / "pfam_families"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "thread_scaling_results"

MMSEQS_BIN = (
    "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/mmseqs/bin/mmseqs"
)


# ──────────────────────────────────────────────────────────────────────
# Tool runners
# ──────────────────────────────────────────────────────────────────────

def run_clustkit(mixed_fasta, out_dir, threshold, threads):
    """Run ClustKIT and return elapsed time."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "input": mixed_fasta,
        "output": out_dir,
        "threshold": threshold,
        "mode": "protein",
        "alignment": "align",
        "sketch_size": 128,
        "kmer_size": 5,
        "sensitivity": "high",
        "cluster_method": "connected",
        "representative": "longest",
        "device": "cpu",
        "threads": threads,
        "format": "tsv",
    }

    start = time.perf_counter()
    run_pipeline(config)
    elapsed = time.perf_counter() - start

    # Count clusters from output
    n_clusters = 0
    tsv_path = out_dir / "clusters.tsv"
    if tsv_path.exists():
        cluster_ids = set()
        with open(tsv_path) as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    cluster_ids.add(int(parts[1]))
        n_clusters = len(cluster_ids)

    return elapsed, n_clusters


def parse_mmseqs_clusters(tsv_path):
    """Parse MMseqs2 cluster TSV and return number of clusters."""
    clusters = set()
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                clusters.add(parts[0])
    return len(clusters)


def run_mmseqs(fasta_path, output_prefix, threshold, threads):
    """Run MMseqs2 easy-cluster and return elapsed time."""
    tmp_dir = tempfile.mkdtemp(prefix="mmseqs_scaling_tmp_")
    cmd = [
        MMSEQS_BIN, "easy-cluster",
        str(fasta_path),
        str(output_prefix),
        tmp_dir,
        "--min-seq-id", str(threshold),
        "--threads", str(threads),
        "-c", "0.8",
        "--cov-mode", "0",
    ]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.perf_counter() - start

    shutil.rmtree(tmp_dir, ignore_errors=True)

    n_clusters = 0
    tsv_file = str(output_prefix) + "_cluster.tsv"
    if result.returncode == 0 and os.path.exists(tsv_file):
        n_clusters = parse_mmseqs_clusters(tsv_file)

    return elapsed, n_clusters, result.returncode == 0


# ──────────────────────────────────────────────────────────────────────
# Main benchmark
# ──────────────────────────────────────────────────────────────────────

def run_thread_scaling(
    max_threads=192,
    threshold=0.5,
    max_per_family=500,
    repeats=1,
):
    """Run the thread scaling benchmark.

    Args:
        max_threads: Maximum number of threads to test.
        threshold: Identity threshold for clustering.
        max_per_family: Max sequences per Pfam family.
        repeats: Number of repeat runs per configuration (for variance).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine thread counts to test
    thread_counts = []
    t = 1
    while t <= max_threads:
        thread_counts.append(t)
        t *= 2
    # Always include max_threads if not already present
    if thread_counts[-1] != max_threads:
        thread_counts.append(max_threads)

    print("=" * 120)
    print(f"THREAD SCALING BENCHMARK — ClustKIT & MMseqs2 at threshold={threshold}")
    print(f"Thread counts: {thread_counts}")
    print(f"Repeats per config: {repeats}")
    print("=" * 120)
    print()

    # Load and mix families
    mixed_fasta, ground_truth = load_and_mix_families(DATA_DIR, max_per_family)
    n_sequences = len(ground_truth)
    print(f"Dataset: {n_sequences} sequences")
    print()

    # ── ClustKIT scaling ─────────────────────────────────────────────
    print("-" * 120)
    print("ClustKIT Thread Scaling")
    print("-" * 120)

    clustkit_results = []
    baseline_time = None

    for n_threads in thread_counts:
        times = []
        n_clusters = 0

        for rep in range(repeats):
            out_dir = OUTPUT_DIR / f"clustkit_t{threshold}_threads{n_threads}_rep{rep}"
            elapsed, n_clust = run_clustkit(
                mixed_fasta, out_dir, threshold, n_threads
            )
            times.append(elapsed)
            n_clusters = n_clust

        mean_time = np.mean(times)
        std_time = np.std(times) if repeats > 1 else 0.0

        if baseline_time is None:
            baseline_time = mean_time

        speedup = baseline_time / mean_time if mean_time > 0 else 0.0
        efficiency = speedup / n_threads if n_threads > 0 else 0.0
        throughput = n_sequences / mean_time if mean_time > 0 else 0.0

        result = {
            "tool": "ClustKIT",
            "threads": n_threads,
            "mean_time": round(mean_time, 2),
            "std_time": round(std_time, 2),
            "speedup": round(speedup, 2),
            "parallel_efficiency": round(efficiency, 4),
            "throughput_seqs_per_sec": round(throughput, 1),
            "n_clusters": n_clusters,
        }
        clustkit_results.append(result)

        if repeats > 1:
            print(
                f"  {n_threads:>3} threads: {mean_time:>8.2f}s +/- {std_time:.2f}s | "
                f"speedup={speedup:.2f}x | efficiency={efficiency:.1%} | "
                f"throughput={throughput:.0f} seq/s | {n_clusters} clusters"
            )
        else:
            print(
                f"  {n_threads:>3} threads: {mean_time:>8.2f}s | "
                f"speedup={speedup:.2f}x | efficiency={efficiency:.1%} | "
                f"throughput={throughput:.0f} seq/s | {n_clusters} clusters"
            )

    print()

    # ── MMseqs2 scaling ──────────────────────────────────────────────
    print("-" * 120)
    print("MMseqs2 Thread Scaling")
    print("-" * 120)

    mmseqs_results = []
    mmseqs_baseline_time = None

    for n_threads in thread_counts:
        times = []
        n_clusters = 0
        success = True

        for rep in range(repeats):
            mmseqs_prefix = OUTPUT_DIR / f"mmseqs_t{threshold}_threads{n_threads}_rep{rep}"
            elapsed, n_clust, ok = run_mmseqs(
                mixed_fasta, mmseqs_prefix, threshold, n_threads
            )
            if ok:
                times.append(elapsed)
                n_clusters = n_clust
            else:
                success = False
                break

        if not success or not times:
            print(f"  {n_threads:>3} threads: FAILED")
            mmseqs_results.append({
                "tool": "MMseqs2",
                "threads": n_threads,
                "error": "failed",
            })
            continue

        mean_time = np.mean(times)
        std_time = np.std(times) if repeats > 1 else 0.0

        if mmseqs_baseline_time is None:
            mmseqs_baseline_time = mean_time

        speedup = mmseqs_baseline_time / mean_time if mean_time > 0 else 0.0
        efficiency = speedup / n_threads if n_threads > 0 else 0.0
        throughput = n_sequences / mean_time if mean_time > 0 else 0.0

        result = {
            "tool": "MMseqs2",
            "threads": n_threads,
            "mean_time": round(mean_time, 2),
            "std_time": round(std_time, 2),
            "speedup": round(speedup, 2),
            "parallel_efficiency": round(efficiency, 4),
            "throughput_seqs_per_sec": round(throughput, 1),
            "n_clusters": n_clusters,
        }
        mmseqs_results.append(result)

        if repeats > 1:
            print(
                f"  {n_threads:>3} threads: {mean_time:>8.2f}s +/- {std_time:.2f}s | "
                f"speedup={speedup:.2f}x | efficiency={efficiency:.1%} | "
                f"throughput={throughput:.0f} seq/s | {n_clusters} clusters"
            )
        else:
            print(
                f"  {n_threads:>3} threads: {mean_time:>8.2f}s | "
                f"speedup={speedup:.2f}x | efficiency={efficiency:.1%} | "
                f"throughput={throughput:.0f} seq/s | {n_clusters} clusters"
            )

    print()

    # ── Summary table ────────────────────────────────────────────────
    print("=" * 120)
    print("SUMMARY — Thread Scaling")
    print("=" * 120)
    print()

    # ClustKIT summary
    print("ClustKIT:")
    print(
        f"  {'Threads':>8} {'Time(s)':>10} {'Speedup':>10} "
        f"{'Efficiency':>12} {'Throughput':>14} {'Clusters':>10}"
    )
    print("  " + "-" * 70)
    for res in clustkit_results:
        if "error" in res:
            print(f"  {res['threads']:>8} {'FAILED':>10}")
            continue
        print(
            f"  {res['threads']:>8} "
            f"{res['mean_time']:>10.2f} "
            f"{res['speedup']:>9.2f}x "
            f"{res['parallel_efficiency']:>11.1%} "
            f"{res['throughput_seqs_per_sec']:>11.0f} seq/s "
            f"{res['n_clusters']:>10}"
        )
    print()

    # MMseqs2 summary
    print("MMseqs2:")
    print(
        f"  {'Threads':>8} {'Time(s)':>10} {'Speedup':>10} "
        f"{'Efficiency':>12} {'Throughput':>14} {'Clusters':>10}"
    )
    print("  " + "-" * 70)
    for res in mmseqs_results:
        if "error" in res:
            print(f"  {res['threads']:>8} {'FAILED':>10}")
            continue
        print(
            f"  {res['threads']:>8} "
            f"{res['mean_time']:>10.2f} "
            f"{res['speedup']:>9.2f}x "
            f"{res['parallel_efficiency']:>11.1%} "
            f"{res['throughput_seqs_per_sec']:>11.0f} seq/s "
            f"{res['n_clusters']:>10}"
        )
    print()

    # ── Side-by-side comparison at each thread count ─────────────────
    print("Side-by-side comparison (time in seconds):")
    print(
        f"  {'Threads':>8} {'ClustKIT':>12} {'MMseqs2':>12} "
        f"{'CK Speedup':>12} {'MM Speedup':>12}"
    )
    print("  " + "-" * 62)

    mmseqs_by_threads = {}
    for res in mmseqs_results:
        if "error" not in res:
            mmseqs_by_threads[res["threads"]] = res

    for res in clustkit_results:
        if "error" in res:
            continue
        t = res["threads"]
        ck_time = res["mean_time"]
        ck_speedup = res["speedup"]

        mm = mmseqs_by_threads.get(t)
        if mm:
            mm_time = mm["mean_time"]
            mm_speedup = mm["speedup"]
            print(
                f"  {t:>8} {ck_time:>11.2f}s {mm_time:>11.2f}s "
                f"{ck_speedup:>11.2f}x {mm_speedup:>11.2f}x"
            )
        else:
            print(
                f"  {t:>8} {ck_time:>11.2f}s {'N/A':>12} "
                f"{ck_speedup:>11.2f}x {'N/A':>12}"
            )

    print()

    # Save results
    all_results = {
        "config": {
            "threshold": threshold,
            "max_per_family": max_per_family,
            "n_sequences": n_sequences,
            "thread_counts": thread_counts,
            "repeats": repeats,
        },
        "clustkit": clustkit_results,
        "mmseqs2": mmseqs_results,
    }

    results_file = OUTPUT_DIR / "thread_scaling_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Thread scaling benchmark for ClustKIT and MMseqs2 (C5)"
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=192,
        help="Maximum number of threads to test (default: 192).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Identity threshold for clustering (default: 0.5).",
    )
    parser.add_argument(
        "--max-per-family",
        type=int,
        default=500,
        help="Max sequences per Pfam family (default: 500).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeat runs per configuration (default: 1).",
    )
    args = parser.parse_args()

    run_thread_scaling(
        max_threads=args.max_threads,
        threshold=args.threshold,
        max_per_family=args.max_per_family,
        repeats=args.repeats,
    )
