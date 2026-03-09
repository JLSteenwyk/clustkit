"""CPU vs GPU performance comparison for ClustKIT.

Tests each phase (sketch, LSH, pairwise) individually, plus the full pipeline,
on datasets of increasing size.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.io import read_sequences
from clustkit.sketch import compute_sketches
from clustkit.lsh import lsh_candidates
from clustkit.pairwise import compute_pairwise_jaccard, compute_pairwise_alignment
from clustkit.pipeline import run_pipeline


def generate_fasta(path, n_sequences, seq_length, n_families, seed=42):
    """Generate a synthetic FASTA with n_families of similar sequences."""
    rng = np.random.RandomState(seed)
    aa = "ACDEFGHIKLMNPQRSTVWY"

    with open(path, "w") as f:
        for i in range(n_sequences):
            fam = i % n_families
            # Each family has a conserved template; members differ by ~10%
            rng_fam = np.random.RandomState(seed + fam)
            template = [aa[rng_fam.randint(0, 20)] for _ in range(seq_length)]

            seq = list(template)
            n_mut = int(seq_length * 0.1)
            positions = rng.choice(seq_length, size=n_mut, replace=False)
            for p in positions:
                seq[p] = aa[rng.randint(0, 20)]

            f.write(f">seq{i:05d}_fam{fam}\n{''.join(seq)}\n")


def time_fn(fn, *args, warmup=0, repeats=1, **kwargs):
    """Time a function, returning (result, elapsed_seconds)."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - start)
    return result, min(times)


def benchmark_phases(dataset, device, k=5, sketch_size=128, threshold=0.9):
    """Benchmark each phase on a given device. Returns dict of timings."""
    timings = {}

    # Phase 1: Sketch
    sketches, t = time_fn(
        compute_sketches,
        dataset.encoded_sequences, dataset.lengths, k, sketch_size, "protein",
        device=device,
    )
    timings["sketch"] = t

    # Phase 2: LSH
    candidates, t = time_fn(
        lsh_candidates,
        sketches,
        num_tables=32, num_bands=2,
        device=device,
    )
    timings["lsh"] = t

    # Phase 3a: Pairwise Jaccard
    if len(candidates) > 0:
        kmer_threshold = threshold ** k
        _, t = time_fn(
            compute_pairwise_jaccard,
            candidates, sketches, kmer_threshold,
            device=device,
        )
        timings["pairwise_jaccard"] = t

    # Phase 3b: Pairwise Alignment (CPU only, for reference)
    if device == "cpu" and len(candidates) > 0:
        band_width = max(20, int(dataset.max_length * 0.2))
        _, t = time_fn(
            compute_pairwise_alignment,
            candidates, dataset.encoded_sequences, dataset.lengths,
            threshold, band_width=band_width,
            device=device,
        )
        timings["pairwise_align"] = t

    return timings


def benchmark_full_pipeline(fasta_path, device, output_dir, threshold=0.9):
    """Time the full pipeline end-to-end."""
    config = {
        "input": fasta_path,
        "output": output_dir,
        "threshold": threshold,
        "mode": "protein",
        "alignment": "kmer",
        "sketch_size": 128,
        "kmer_size": 5,
        "sensitivity": "medium",
        "cluster_method": "connected",
        "representative": "longest",
        "device": device,
        "threads": 1,
        "format": "tsv",
    }
    _, t = time_fn(run_pipeline, config)
    return t


def main():
    gpu_device = "1"  # Use GPU 1 (more free memory)
    tmp_dir = Path("/tmp/clustkit_gpu_bench")
    tmp_dir.mkdir(exist_ok=True)

    sizes = [500, 1000, 2000, 5000]

    print("=" * 90)
    print("ClustKIT CPU vs GPU PERFORMANCE BENCHMARK")
    print("=" * 90)

    # Warmup GPU
    print("\nWarming up GPU (JIT compilation)...")
    warmup_fasta = tmp_dir / "warmup.fasta"
    generate_fasta(warmup_fasta, 50, 100, 5)
    warmup_ds = read_sequences(warmup_fasta, "protein")
    benchmark_phases(warmup_ds, gpu_device)
    benchmark_phases(warmup_ds, "cpu")
    print("Warmup done.\n")

    all_results = {}

    for n in sizes:
        print("-" * 90)
        print(f"Dataset: {n} sequences x 200aa, 20 families")
        print("-" * 90)

        fasta_path = tmp_dir / f"bench_{n}.fasta"
        generate_fasta(fasta_path, n, 200, 20)
        dataset = read_sequences(fasta_path, "protein")

        # Per-phase benchmarks
        cpu_times = benchmark_phases(dataset, "cpu")
        gpu_times = benchmark_phases(dataset, gpu_device)

        print(f"\n  {'Phase':<22} {'CPU (s)':>10} {'GPU (s)':>10} {'Speedup':>10}")
        print(f"  {'-'*52}")
        for phase in ["sketch", "lsh", "pairwise_jaccard", "pairwise_align"]:
            cpu_t = cpu_times.get(phase)
            gpu_t = gpu_times.get(phase)
            if cpu_t is not None and gpu_t is not None:
                speedup = cpu_t / gpu_t if gpu_t > 0 else float("inf")
                print(f"  {phase:<22} {cpu_t:>10.4f} {gpu_t:>10.4f} {speedup:>9.1f}x")
            elif cpu_t is not None:
                print(f"  {phase:<22} {cpu_t:>10.4f} {'N/A':>10} {'N/A':>10}")

        # Full pipeline benchmark
        cpu_out = tmp_dir / f"cpu_out_{n}"
        gpu_out = tmp_dir / f"gpu_out_{n}"
        cpu_out.mkdir(exist_ok=True)
        gpu_out.mkdir(exist_ok=True)

        cpu_pipeline = benchmark_full_pipeline(fasta_path, "cpu", cpu_out)
        gpu_pipeline = benchmark_full_pipeline(fasta_path, gpu_device, gpu_out)
        speedup = cpu_pipeline / gpu_pipeline if gpu_pipeline > 0 else float("inf")

        print(f"\n  {'Full pipeline (kmer)':<22} {cpu_pipeline:>10.4f} {gpu_pipeline:>10.4f} {speedup:>9.1f}x")
        print()

        all_results[n] = {
            "cpu_phases": cpu_times,
            "gpu_phases": gpu_times,
            "cpu_pipeline": cpu_pipeline,
            "gpu_pipeline": gpu_pipeline,
        }

    # Summary
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"\n  {'N seqs':<10} {'CPU pipe (s)':>14} {'GPU pipe (s)':>14} {'Speedup':>10}")
    print(f"  {'-'*48}")
    for n in sizes:
        r = all_results[n]
        s = r["cpu_pipeline"] / r["gpu_pipeline"] if r["gpu_pipeline"] > 0 else float("inf")
        print(f"  {n:<10} {r['cpu_pipeline']:>14.4f} {r['gpu_pipeline']:>14.4f} {s:>9.1f}x")


if __name__ == "__main__":
    main()
