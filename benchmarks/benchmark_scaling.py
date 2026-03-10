"""ClustKIT Scaling Benchmark

Tests ClustKIT performance at 10K-500K sequences, capturing per-phase timings
and peak memory usage.
"""

import logging
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.pipeline import run_pipeline

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "scaling_results"

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def generate_synthetic_fasta(path, num_sequences, num_families=100,
                             seq_length=200, target_identity=0.85, seed=42):
    """Generate synthetic protein FASTA by mutating family ancestors.

    Much faster than pyvolve — writes directly to disk without holding
    all sequences in memory.
    """
    rng = np.random.RandomState(seed)
    members_per_family = num_sequences // num_families
    mutation_rate = 1.0 - target_identity

    with open(path, "w") as f:
        for fam in range(num_families):
            # Random ancestor
            ancestor = rng.choice(len(AMINO_ACIDS), size=seq_length)

            for mem in range(members_per_family):
                seq = ancestor.copy()
                # Mutate at mutation_rate
                mask = rng.random(seq_length) < mutation_rate
                seq[mask] = rng.choice(len(AMINO_ACIDS), size=mask.sum())
                seq_str = "".join(AMINO_ACIDS[aa] for aa in seq)
                f.write(f">fam{fam:04d}_s{mem}\n{seq_str}\n")

        # Fill remainder
        remainder = num_sequences - (num_families * members_per_family)
        for i in range(remainder):
            fam = rng.randint(0, num_families)
            seq = rng.choice(len(AMINO_ACIDS), size=seq_length)
            seq_str = "".join(AMINO_ACIDS[aa] for aa in seq)
            f.write(f">fam{fam:04d}_extra{i}\n{seq_str}\n")


class TimingHandler(logging.Handler):
    """Capture phase timings from ClustKIT's timer() context manager."""

    def __init__(self):
        super().__init__()
        self.timings = {}

    def emit(self, record):
        msg = record.getMessage()
        # Match "Phase N: Description done (X.XXs)"
        if " done (" in msg and "Phase" in msg:
            try:
                phase = msg.split(":")[0].strip()
                time_str = msg.split("(")[1].split("s)")[0]
                self.timings[phase] = float(time_str)
            except (IndexError, ValueError):
                pass


def run_benchmark():
    """Run scaling benchmark."""
    sizes = [10_000, 50_000, 100_000, 250_000, 500_000]
    threshold = 0.7
    timeout_minutes = 10

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    print("=" * 100)
    print("ClustKIT SCALING BENCHMARK")
    print("=" * 100)
    print(f"Config: threshold={threshold}, alignment mode, 100 families, 200aa, ~85% identity")
    print()

    for n in sizes:
        label = f"{n // 1000}K"
        fasta_path = OUTPUT_DIR / f"synthetic_{label}.fasta"
        out_dir = OUTPUT_DIR / f"clustkit_{label}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"--- {label} sequences ---")

        # Generate data
        print(f"  Generating {n} sequences...", end=" ", flush=True)
        t0 = time.perf_counter()
        generate_synthetic_fasta(fasta_path, n)
        gen_time = time.perf_counter() - t0
        fasta_size_mb = os.path.getsize(fasta_path) / (1024 * 1024)
        print(f"done ({gen_time:.1f}s, {fasta_size_mb:.1f} MB)")

        # Set up timing capture
        handler = TimingHandler()
        clustkit_logger = logging.getLogger("clustkit")
        clustkit_logger.addHandler(handler)

        config = {
            "input": fasta_path,
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
            "threads": 1,
            "format": "tsv",
        }

        # Run with resource tracking
        mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t0 = time.perf_counter()

        try:
            run_pipeline(config)
            total_time = time.perf_counter() - t0
            status = "OK"
        except Exception as e:
            total_time = time.perf_counter() - t0
            status = f"FAIL: {e}"

        mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mem_mb = mem_after / 1024  # ru_maxrss is in KB on Linux

        clustkit_logger.removeHandler(handler)

        # Count clusters
        n_clusters = "?"
        try:
            with open(out_dir / "clusters.tsv") as f:
                next(f)
                cluster_ids = set()
                for line in f:
                    cluster_ids.add(line.strip().split("\t")[1])
                n_clusters = len(cluster_ids)
        except Exception:
            pass

        result = {
            "size": label,
            "n": n,
            "status": status,
            "total": round(total_time, 2),
            "n_clusters": n_clusters,
            "peak_mem_mb": round(peak_mem_mb, 1),
            **{k: round(v, 2) for k, v in handler.timings.items()},
        }
        results.append(result)

        print(f"  Status: {status}")
        print(f"  Total: {total_time:.2f}s | Clusters: {n_clusters} | Peak mem: {peak_mem_mb:.0f} MB")
        for phase, t in sorted(handler.timings.items()):
            print(f"    {phase}: {t:.2f}s")
        print()

        # Skip larger sizes if this one took too long
        if total_time > timeout_minutes * 60:
            print(f"  Skipping larger sizes (>{timeout_minutes} min timeout)")
            break

    # Summary table
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()

    phases = ["Phase 0", "Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5"]
    phase_labels = ["Read", "Sketch", "LSH", "Pairwise", "Graph", "Cluster"]

    header = f"{'Size':<8} {'Status':<8} {'Total':>8}"
    for pl in phase_labels:
        header += f" {pl:>8}"
    header += f" {'Clusters':>8} {'Mem(MB)':>8}"
    print(header)
    print("-" * len(header))

    for r in results:
        line = f"{r['size']:<8} {r['status']:<8} {r['total']:>7.2f}s"
        for p in phases:
            val = r.get(p, None)
            line += f" {val:>7.2f}s" if val is not None else f" {'N/A':>8}"
        line += f" {str(r['n_clusters']):>8} {r['peak_mem_mb']:>7.1f}"
        print(line)

    print()
    print(f"Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    run_benchmark()
