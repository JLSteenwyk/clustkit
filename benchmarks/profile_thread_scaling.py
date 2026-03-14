"""Thread scaling benchmark for ClustKIT Phase 3.

Runs alignment on a fixed subset of candidate pairs at different thread counts.
Uses a subset (default 500K pairs) to keep total runtime reasonable while still
capturing real scaling behavior.

Usage:
    python benchmarks/profile_thread_scaling.py --threshold 0.4 --sample 500000
"""

import argparse
import sys
import time
from pathlib import Path

import numba
import numpy as np
from numba import int32

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.io import read_sequences
from clustkit.sketch import compute_sketches
from clustkit.lsh import lsh_candidates
from clustkit.pairwise import _batch_align
from clustkit.utils import auto_kmer_for_lsh, auto_lsh_params

DATA_DIR = Path(__file__).resolve().parent / "data"


def main():
    parser = argparse.ArgumentParser(description="Thread scaling for ClustKIT Phase 3")
    parser.add_argument("--input", "-i", type=str,
                        default=str(DATA_DIR / "pfam_mixed.fasta"))
    parser.add_argument("--threshold", "-t", type=float, default=0.4)
    parser.add_argument("--mode", type=str, default="protein")
    parser.add_argument("--sensitivity", type=str, default="high")
    parser.add_argument("--sample", type=int, default=500_000,
                        help="Number of candidate pairs to sample (0 = all)")
    parser.add_argument("--max-threads", type=int, default=192)
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of trials per thread count (best-of-N)")
    args = parser.parse_args()

    input_path = Path(args.input)
    print(f"{'='*80}")
    print(f"  Thread Scaling Benchmark")
    print(f"{'='*80}")
    print(f"  Input:      {input_path}")
    print(f"  Threshold:  {args.threshold}")
    print(f"  Sample:     {args.sample if args.sample > 0 else 'all'}")
    print(f"  Trials:     {args.trials}")
    print()

    # Load data
    dataset = read_sequences(input_path, args.mode)
    n = dataset.num_sequences
    print(f"  Loaded {n:,} sequences")

    k_lsh = auto_kmer_for_lsh(args.threshold, args.mode, 5)
    sketches = compute_sketches(dataset.encoded_sequences, dataset.lengths,
                                k_lsh, 128, args.mode)

    lsh_params = auto_lsh_params(args.threshold, args.sensitivity, k=k_lsh)
    candidate_pairs = lsh_candidates(sketches,
                                     num_tables=lsh_params["num_tables"],
                                     num_bands=lsh_params["num_bands"])
    m = len(candidate_pairs)
    print(f"  LSH candidates: {m:,} pairs")

    # Sample if needed
    if args.sample > 0 and args.sample < m:
        rng = np.random.RandomState(42)
        idx = rng.choice(m, size=args.sample, replace=False)
        pairs = candidate_pairs[idx]
        print(f"  Sampled: {len(pairs):,} pairs")
    else:
        pairs = candidate_pairs
        print(f"  Using all {len(pairs):,} pairs")

    p95_len = int(np.percentile(dataset.lengths, 95))
    band_width = max(20, int(p95_len * 0.3))
    print(f"  Band width: {band_width}")
    print()

    # Thread counts
    system_threads = numba.config.NUMBA_NUM_THREADS
    thread_counts = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192]
    thread_counts = [t for t in thread_counts if t <= min(args.max_threads, system_threads)]

    # Warm up
    numba.set_num_threads(1)
    _batch_align(pairs[:100], dataset.encoded_sequences, dataset.lengths,
                 np.float32(args.threshold), int32(band_width))

    print(f"  {'Threads':>8} {'Time (s)':>10} {'Pairs/s':>14} {'Speedup':>10} {'Efficiency':>12}")
    print(f"  {'-'*8} {'-'*10} {'-'*14} {'-'*10} {'-'*12}")

    baseline_time = None
    results = []

    for nt in thread_counts:
        numba.set_num_threads(nt)

        best_time = float("inf")
        for trial in range(args.trials):
            t0 = time.perf_counter()
            _batch_align(pairs, dataset.encoded_sequences, dataset.lengths,
                         np.float32(args.threshold), int32(band_width))
            elapsed = time.perf_counter() - t0
            best_time = min(best_time, elapsed)

        if baseline_time is None:
            baseline_time = best_time

        throughput = len(pairs) / best_time
        speedup = baseline_time / best_time
        efficiency = speedup / nt * 100.0

        results.append({"threads": nt, "time": best_time, "throughput": throughput,
                         "speedup": speedup, "efficiency": efficiency})

        print(f"  {nt:>8} {best_time:>10.3f} {throughput:>14,.0f} {speedup:>9.2f}x {efficiency:>10.1f}%")

    print()

    # ASCII chart
    if len(results) > 1:
        max_speedup = max(r["speedup"] for r in results)
        chart_width = 50
        print(f"  Thread scaling (speedup relative to 1 thread):")
        print()
        for r in results:
            bar_len = int(r["speedup"] / max_speedup * chart_width)
            bar = "#" * bar_len
            print(f"  {r['threads']:>4}T  |{bar:<{chart_width}}| {r['speedup']:.2f}x")
        print()

    # Extrapolate full runtime at each thread count
    if args.sample > 0 and args.sample < m:
        print(f"  Estimated full runtime ({m:,} pairs) at each thread count:")
        for r in results:
            full_est = m / r["throughput"]
            print(f"    {r['threads']:>4} threads: {full_est:>8.1f}s ({full_est/60:.1f} min)")
        print()

    print(f"{'='*80}")
    print(f"  Done.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
