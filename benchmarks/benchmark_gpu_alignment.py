"""Benchmark GPU vs CPU for alignment-based pairwise identity.

First verifies correctness (identical results), then benchmarks at scale.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.io import read_sequences
from clustkit.sketch import compute_sketches
from clustkit.lsh import lsh_candidates
from clustkit.pairwise import compute_pairwise_alignment, compute_pairwise_jaccard


def generate_fasta(path, n_sequences, seq_length, n_families, identity=0.85, seed=42):
    """Generate synthetic FASTA with families sharing given identity."""
    rng = np.random.RandomState(seed)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    divergence = 1.0 - identity

    with open(path, "w") as f:
        for i in range(n_sequences):
            fam = i % n_families
            rng_fam = np.random.RandomState(seed + fam)
            template = [aa[rng_fam.randint(0, 20)] for _ in range(seq_length)]

            seq = list(template)
            n_mut = int(seq_length * divergence)
            positions = rng.choice(seq_length, size=n_mut, replace=False)
            for p in positions:
                seq[p] = aa[rng.randint(0, 20)]

            f.write(f">seq{i:05d}_fam{fam}\n{''.join(seq)}\n")


def get_candidates(dataset, k=5, sketch_size=128, threshold=0.7):
    """Get candidate pairs via sketch + LSH."""
    sketches = compute_sketches(
        dataset.encoded_sequences, dataset.lengths, k, sketch_size, "protein",
    )
    candidates = lsh_candidates(sketches, num_tables=64, num_bands=1)
    return candidates


def main():
    gpu_device = "1"
    tmp_dir = Path("/tmp/clustkit_gpu_align_bench")
    tmp_dir.mkdir(exist_ok=True)

    # ── Step 1: Correctness verification ────────────────────────────
    print("=" * 80)
    print("STEP 1: CORRECTNESS VERIFICATION (CPU vs GPU alignment)")
    print("=" * 80)

    fasta_path = tmp_dir / "verify.fasta"
    generate_fasta(fasta_path, 500, 200, 20, identity=0.85)
    dataset = read_sequences(fasta_path, "protein")
    candidates = get_candidates(dataset, threshold=0.5)
    print(f"  {len(candidates)} candidate pairs")

    band_width = max(20, int(dataset.max_length * 0.2))

    # CPU
    cpu_pairs, cpu_sims = compute_pairwise_alignment(
        candidates, dataset.encoded_sequences, dataset.lengths,
        0.5, band_width=band_width, device="cpu",
    )

    # GPU
    gpu_pairs, gpu_sims = compute_pairwise_alignment(
        candidates, dataset.encoded_sequences, dataset.lengths,
        0.5, band_width=band_width, device=gpu_device,
    )

    print(f"  CPU: {len(cpu_pairs)} pairs above threshold")
    print(f"  GPU: {len(gpu_pairs)} pairs above threshold")

    if len(cpu_pairs) == len(gpu_pairs):
        # Sort both by pair indices for comparison
        cpu_order = np.lexsort((cpu_pairs[:, 1], cpu_pairs[:, 0]))
        gpu_order = np.lexsort((gpu_pairs[:, 1], gpu_pairs[:, 0]))

        pairs_match = np.array_equal(cpu_pairs[cpu_order], gpu_pairs[gpu_order])
        max_sim_diff = np.max(np.abs(cpu_sims[cpu_order] - gpu_sims[gpu_order]))

        print(f"  Pairs identical: {pairs_match}")
        print(f"  Max identity difference: {max_sim_diff:.6f}")

        if pairs_match and max_sim_diff < 1e-6:
            print("  PASSED: GPU and CPU produce identical results")
        else:
            print("  WARNING: Results differ!")
            if not pairs_match:
                print("    Pair sets are different")
            if max_sim_diff >= 1e-6:
                print(f"    Identity values differ by up to {max_sim_diff}")
    else:
        print("  FAILED: Different number of pairs!")
        # Show some diagnostics
        cpu_set = set(map(tuple, cpu_pairs.tolist()))
        gpu_set = set(map(tuple, gpu_pairs.tolist()))
        only_cpu = cpu_set - gpu_set
        only_gpu = gpu_set - cpu_set
        print(f"    Only in CPU: {len(only_cpu)}")
        print(f"    Only in GPU: {len(only_gpu)}")

    # ── Step 2: Performance benchmark ───────────────────────────────
    print()
    print("=" * 80)
    print("STEP 2: ALIGNMENT PERFORMANCE BENCHMARK (CPU vs GPU)")
    print("=" * 80)

    # Warmup
    print("\nWarming up...")
    compute_pairwise_alignment(
        candidates[:min(100, len(candidates))],
        dataset.encoded_sequences, dataset.lengths,
        0.5, band_width=band_width, device=gpu_device,
    )
    print("Done.\n")

    sizes = [1_000, 5_000, 10_000, 20_000, 50_000]
    threshold = 0.5

    print(f"{'N seqs':<10} {'Candidates':>12} {'CPU (s)':>10} {'GPU (s)':>10} {'Speedup':>10}")
    print("-" * 62)

    for n in sizes:
        fasta_path = tmp_dir / f"bench_{n}.fasta"
        generate_fasta(fasta_path, n, 200, 50, identity=0.85)
        dataset = read_sequences(fasta_path, "protein")
        candidates = get_candidates(dataset, threshold=threshold)
        band_width = max(20, int(dataset.max_length * 0.2))

        if len(candidates) == 0:
            print(f"{n:<10} {'0':>12} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            continue

        # CPU
        t0 = time.perf_counter()
        compute_pairwise_alignment(
            candidates, dataset.encoded_sequences, dataset.lengths,
            threshold, band_width=band_width, device="cpu",
        )
        cpu_time = time.perf_counter() - t0

        # GPU
        t0 = time.perf_counter()
        compute_pairwise_alignment(
            candidates, dataset.encoded_sequences, dataset.lengths,
            threshold, band_width=band_width, device=gpu_device,
        )
        gpu_time = time.perf_counter() - t0

        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
        print(f"{n:<10} {len(candidates):>12,} {cpu_time:>10.4f} {gpu_time:>10.4f} {speedup:>9.1f}x")

    # ── Step 3: Full pipeline comparison (align mode) ───────────────
    print()
    print("=" * 80)
    print("STEP 3: FULL PIPELINE (align mode) CPU vs GPU")
    print("=" * 80)

    from clustkit.pipeline import run_pipeline

    for n in [5_000, 10_000, 20_000]:
        fasta_path = tmp_dir / f"bench_{n}.fasta"
        if not fasta_path.exists():
            generate_fasta(fasta_path, n, 200, 50, identity=0.85)

        for device, label in [("cpu", "CPU"), (gpu_device, f"GPU:{gpu_device}")]:
            out_dir = tmp_dir / f"pipe_{n}_{device}"
            out_dir.mkdir(exist_ok=True)
            config = {
                "input": fasta_path,
                "output": out_dir,
                "threshold": 0.7,
                "mode": "protein",
                "alignment": "align",
                "sketch_size": 128,
                "kmer_size": 5,
                "sensitivity": "medium",
                "cluster_method": "connected",
                "representative": "longest",
                "device": device,
                "threads": 1,
                "format": "tsv",
            }
            t0 = time.perf_counter()
            run_pipeline(config)
            elapsed = time.perf_counter() - t0
            print(f"  N={n:>6,}  {label:<8}  {elapsed:.4f}s")
        print()


if __name__ == "__main__":
    main()
