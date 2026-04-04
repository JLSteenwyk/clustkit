"""Thread scaling benchmark for clustering tools.

Fixed Pfam dataset, varying thread count from 1 to max.
Measures scaling for ClustKIT and the external clustering baselines.

C5 from the publication plan.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.clustering_mode import resolve_clustering_mode
from clustkit.pipeline import run_pipeline

# Import data loading from the Pfam benchmark
from benchmark_pfam_concordance import (
    load_and_mix_families,
    run_cdhit as run_cdhit_clusters,
    run_deepclust as run_deepclust_clusters,
    run_mmseqs as run_mmseqs_clusters,
    run_mmseqs_linclust as run_linclust_clusters,
    run_vsearch as run_vsearch_clusters,
)


DATA_DIR = Path(__file__).resolve().parent / "data" / "pfam_families"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "thread_scaling_results"


def run_clustkit(mixed_fasta, out_dir, threshold, threads, clustkit_mode):
    """Run ClustKIT and return elapsed time."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sketch_size, sensitivity = resolve_clustering_mode(clustkit_mode, threshold)

    config = {
        "input": mixed_fasta,
        "output": out_dir,
        "threshold": threshold,
        "clustering_mode": clustkit_mode,
        "mode": "protein",
        "alignment": "align",
        "sketch_size": sketch_size,
        "kmer_size": 5,
        "sensitivity": sensitivity,
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


def _count_clusters(cluster_map):
    if not cluster_map:
        return 0
    return len(set(cluster_map.values()))


def run_external_tool(runner, fasta_path, output_prefix, threshold, threads):
    """Run an external clustering tool and return elapsed time and cluster count."""
    clusters, elapsed = runner(fasta_path, output_prefix, threshold, threads=threads)
    if clusters is None:
        return elapsed, 0, False
    return elapsed, _count_clusters(clusters), True


# ──────────────────────────────────────────────────────────────────────
# Main benchmark
# ──────────────────────────────────────────────────────────────────────

def run_thread_scaling(
    max_threads=192,
    threshold=0.5,
    max_per_family=500,
    repeats=1,
    clustkit_mode="balanced",
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
    print(
        "THREAD SCALING BENCHMARK — "
        f"ClustKIT[{clustkit_mode}] vs MMseqs2 vs Linclust vs DeepClust vs CD-HIT vs VSEARCH "
        f"at threshold={threshold}"
    )
    print(f"Thread counts: {thread_counts}")
    print(f"Repeats per config: {repeats}")
    print("=" * 120)
    print()

    # Load and mix families
    mixed_fasta, ground_truth = load_and_mix_families(DATA_DIR, max_per_family)
    n_sequences = len(ground_truth)
    print(f"Dataset: {n_sequences} sequences")
    print()

    tool_specs = [
        {
            "key": "clustkit",
            "label": f"ClustKIT-{clustkit_mode}",
            "runner": None,
        },
        {"key": "mmseqs2", "label": "MMseqs2", "runner": run_mmseqs_clusters},
        {"key": "linclust", "label": "Linclust", "runner": run_linclust_clusters},
        {"key": "deepclust", "label": "DeepClust", "runner": run_deepclust_clusters},
        {"key": "cdhit", "label": "CD-HIT", "runner": run_cdhit_clusters},
        {"key": "vsearch", "label": "VSEARCH", "runner": run_vsearch_clusters},
    ]

    results_by_key = {}

    for tool in tool_specs:
        print("-" * 120)
        print(f"{tool['label']} Thread Scaling")
        print("-" * 120)

        tool_results = []
        baseline_time = None

        for n_threads in thread_counts:
            times = []
            n_clusters = 0
            success = True

            for rep in range(repeats):
                if tool["key"] == "clustkit":
                    out_dir = (
                        OUTPUT_DIR
                        / f"clustkit_{clustkit_mode}_t{threshold}_threads{n_threads}_rep{rep}"
                    )
                    elapsed, n_clust = run_clustkit(
                        mixed_fasta, out_dir, threshold, n_threads, clustkit_mode
                    )
                    times.append(elapsed)
                    n_clusters = n_clust
                else:
                    output_prefix = (
                        OUTPUT_DIR / f"{tool['key']}_t{threshold}_threads{n_threads}_rep{rep}"
                    )
                    elapsed, n_clust, ok = run_external_tool(
                        tool["runner"], mixed_fasta, output_prefix, threshold, n_threads
                    )
                    if ok:
                        times.append(elapsed)
                        n_clusters = n_clust
                    else:
                        success = False
                        break

            if not success or not times:
                print(f"  {n_threads:>3} threads: FAILED")
                tool_results.append(
                    {
                        "tool": tool["label"],
                        "threads": n_threads,
                        "error": "failed",
                    }
                )
                continue

            mean_time = np.mean(times)
            std_time = np.std(times) if repeats > 1 else 0.0

            if baseline_time is None:
                baseline_time = mean_time

            speedup = baseline_time / mean_time if mean_time > 0 else 0.0
            efficiency = speedup / n_threads if n_threads > 0 else 0.0
            throughput = n_sequences / mean_time if mean_time > 0 else 0.0

            result = {
                "tool": tool["label"],
                "threads": n_threads,
                "mean_time": round(mean_time, 2),
                "std_time": round(std_time, 2),
                "speedup": round(speedup, 2),
                "parallel_efficiency": round(efficiency, 4),
                "throughput_seqs_per_sec": round(throughput, 1),
                "n_clusters": n_clusters,
            }
            tool_results.append(result)

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

        results_by_key[tool["key"]] = tool_results
        print()

    # ── Summary table ────────────────────────────────────────────────
    print("=" * 120)
    print("SUMMARY — Thread Scaling")
    print("=" * 120)
    print()

    for tool in tool_specs:
        tool_results = results_by_key[tool["key"]]
        print(f"{tool['label']}:")
        print(
            f"  {'Threads':>8} {'Time(s)':>10} {'Speedup':>10} "
            f"{'Efficiency':>12} {'Throughput':>14} {'Clusters':>10}"
        )
        print("  " + "-" * 70)
        for res in tool_results:
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

    print("Side-by-side comparison (time in seconds):")
    header = (
        f"  {'Threads':>8} {'ClustKIT':>12} {'MMseqs2':>12} {'Linclust':>12} "
        f"{'DeepClust':>12} {'CD-HIT':>12} {'VSEARCH':>12}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    by_tool_and_threads = {}
    for tool in tool_specs:
        by_thread = {}
        for res in results_by_key[tool["key"]]:
            if "error" not in res:
                by_thread[res["threads"]] = res
        by_tool_and_threads[tool["key"]] = by_thread

    for n_threads in thread_counts:
        row = [f"  {n_threads:>8}"]
        for tool_key in ["clustkit", "mmseqs2", "linclust", "deepclust", "cdhit", "vsearch"]:
            res = by_tool_and_threads[tool_key].get(n_threads)
            row.append(
                f"{res['mean_time']:>11.2f}s" if res else f"{'FAILED':>12}"
            )
        print(" ".join(row))

    print()

    # Save results
    all_results = {
        "config": {
            "threshold": threshold,
            "max_per_family": max_per_family,
            "n_sequences": n_sequences,
            "thread_counts": thread_counts,
            "repeats": repeats,
            "clustkit_mode": clustkit_mode,
        },
        "results": results_by_key,
    }

    results_file = OUTPUT_DIR / f"thread_scaling_results_{clustkit_mode}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Thread scaling benchmark for ClustKIT and external clustering baselines (C5)"
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
    parser.add_argument(
        "--clustkit-mode",
        type=str,
        default="balanced",
        choices=["balanced", "accurate", "fast"],
        help="ClustKIT clustering mode to benchmark.",
    )
    args = parser.parse_args()

    run_thread_scaling(
        max_threads=args.max_threads,
        threshold=args.threshold,
        max_per_family=args.max_per_family,
        repeats=args.repeats,
        clustkit_mode=args.clustkit_mode,
    )
