"""Large-scale CPU vs GPU benchmark to find the crossover point."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.io import read_sequences
from clustkit.sketch import compute_sketches
from clustkit.lsh import lsh_candidates
from clustkit.pairwise import compute_pairwise_jaccard
from clustkit.pipeline import run_pipeline


def generate_fasta(path, n_sequences, seq_length, n_families, seed=42):
    """Generate synthetic FASTA with families sharing ~85% identity."""
    rng = np.random.RandomState(seed)
    aa = "ACDEFGHIKLMNPQRSTVWY"

    with open(path, "w") as f:
        for i in range(n_sequences):
            fam = i % n_families
            rng_fam = np.random.RandomState(seed + fam)
            template = [aa[rng_fam.randint(0, 20)] for _ in range(seq_length)]

            seq = list(template)
            # ~15% divergence so pairs actually survive filtering
            n_mut = int(seq_length * 0.15)
            positions = rng.choice(seq_length, size=n_mut, replace=False)
            for p in positions:
                seq[p] = aa[rng.randint(0, 20)]

            f.write(f">seq{i:05d}_fam{fam}\n{''.join(seq)}\n")


def main():
    gpu_device = "1"
    tmp_dir = Path("/tmp/clustkit_gpu_bench_large")
    tmp_dir.mkdir(exist_ok=True)

    # Warmup
    print("Warming up JIT/kernels...")
    warmup_fasta = tmp_dir / "warmup.fasta"
    generate_fasta(warmup_fasta, 100, 150, 10)
    ds = read_sequences(warmup_fasta, "protein")
    sk = compute_sketches(ds.encoded_sequences, ds.lengths, 5, 128, "protein", device="cpu")
    lsh_candidates(sk, 32, 2, device="cpu")
    sk = compute_sketches(ds.encoded_sequences, ds.lengths, 5, 128, "protein", device=gpu_device)
    lsh_candidates(sk, 32, 2, device=gpu_device)
    print("Done.\n")

    sizes = [10_000, 20_000, 50_000, 100_000]
    threshold = 0.7
    k = 5
    sketch_size = 128
    kmer_threshold = threshold ** k

    print("=" * 90)
    print("LARGE-SCALE CPU vs GPU BENCHMARK")
    print(f"Threshold: {threshold}, k-mer threshold: {kmer_threshold:.4f}")
    print("=" * 90)

    for n in sizes:
        print(f"\n{'='*90}")
        print(f"Dataset: {n:,} sequences x 200aa, 50 families, ~85% within-family identity")
        print(f"{'='*90}")

        fasta_path = tmp_dir / f"bench_{n}.fasta"
        print(f"Generating {n:,} sequences...")
        generate_fasta(fasta_path, n, 200, 50)
        dataset = read_sequences(fasta_path, "protein")
        print(f"  Loaded {dataset.num_sequences} sequences\n")

        for device, label in [("cpu", "CPU"), (gpu_device, f"GPU:{gpu_device}")]:
            print(f"  --- {label} ---")

            t0 = time.perf_counter()
            sketches = compute_sketches(
                dataset.encoded_sequences, dataset.lengths,
                k, sketch_size, "protein", device=device,
            )
            t_sketch = time.perf_counter() - t0
            print(f"    Sketch:           {t_sketch:.4f}s")

            t0 = time.perf_counter()
            candidates = lsh_candidates(sketches, 32, 2, device=device)
            t_lsh = time.perf_counter() - t0
            print(f"    LSH:              {t_lsh:.4f}s  ({len(candidates):,} candidates)")

            if len(candidates) > 0:
                t0 = time.perf_counter()
                filtered, sims = compute_pairwise_jaccard(
                    candidates, sketches, kmer_threshold, device=device,
                )
                t_pw = time.perf_counter() - t0
                print(f"    Pairwise Jaccard: {t_pw:.4f}s  ({len(filtered):,} pairs above threshold)")
            else:
                t_pw = 0.0
                print(f"    Pairwise Jaccard: 0.0000s  (no candidates)")

            total = t_sketch + t_lsh + t_pw
            print(f"    TOTAL (3 phases): {total:.4f}s")
            print()


if __name__ == "__main__":
    main()
