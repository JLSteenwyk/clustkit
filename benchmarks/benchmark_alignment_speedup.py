"""Benchmark alignment optimizations.

Measures the impact of:
1. Forward match counting (no traceback) — saves memory allocation
2. Early termination — skips DP rows when threshold is unreachable
3. K-mer Jaccard pre-filter — eliminates pairs before alignment

Compares alignment with early termination disabled (threshold=0) vs enabled.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.io import read_sequences
from clustkit.sketch import compute_sketches
from clustkit.lsh import lsh_candidates
from clustkit.pairwise import _batch_align, compute_pairwise_jaccard
from clustkit.pipeline import run_pipeline
from numba import int32


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


def generate_mixed_fasta(path, n_sequences, seq_length, n_families, seed=42):
    """Generate FASTA with a mix of identity levels (more realistic).

    50% of families at ~90% identity, 30% at ~70%, 20% at ~50%.
    This creates more false positive LSH candidates.
    """
    rng = np.random.RandomState(seed)
    aa = "ACDEFGHIKLMNPQRSTVWY"

    with open(path, "w") as f:
        for i in range(n_sequences):
            fam = i % n_families

            # Different divergence levels per family
            if fam % 10 < 5:
                divergence = 0.10  # ~90% identity
            elif fam % 10 < 8:
                divergence = 0.30  # ~70% identity
            else:
                divergence = 0.50  # ~50% identity

            rng_fam = np.random.RandomState(seed + fam)
            template = [aa[rng_fam.randint(0, 20)] for _ in range(seq_length)]

            seq = list(template)
            n_mut = int(seq_length * divergence)
            positions = rng.choice(seq_length, size=n_mut, replace=False)
            for p in positions:
                seq[p] = aa[rng.randint(0, 20)]

            f.write(f">seq{i:05d}_fam{fam}\n{''.join(seq)}\n")


def main():
    tmp_dir = Path("/tmp/clustkit_align_opt_bench2")
    tmp_dir.mkdir(exist_ok=True)

    # Warmup
    print("Warming up Numba JIT...")
    warmup_fasta = tmp_dir / "warmup.fasta"
    generate_fasta(warmup_fasta, 100, 100, 10)
    ds = read_sequences(warmup_fasta, "protein")
    sk = compute_sketches(ds.encoded_sequences, ds.lengths, 5, 128, "protein")
    cands = lsh_candidates(sk, 32, 2)
    if len(cands) > 0:
        _batch_align(cands, ds.encoded_sequences, ds.lengths, np.float32(0.5), int32(20))
        _batch_align(cands, ds.encoded_sequences, ds.lengths, np.float32(0.0), int32(20))
    print("Done.\n")

    print("=" * 90)
    print("ALIGNMENT OPTIMIZATION BENCHMARK")
    print("=" * 90)

    scenarios = [
        # Uniform families
        {"gen": "uniform", "n": 5_000, "families": 50, "identity": 0.85,
         "threshold": 0.7, "label": "5K, t=0.7, uniform 85%"},
        {"gen": "uniform", "n": 10_000, "families": 50, "identity": 0.85,
         "threshold": 0.7, "label": "10K, t=0.7, uniform 85%"},
        {"gen": "uniform", "n": 20_000, "families": 50, "identity": 0.85,
         "threshold": 0.7, "label": "20K, t=0.7, uniform 85%"},
        # Mixed identity (more realistic)
        {"gen": "mixed", "n": 10_000, "families": 50,
         "threshold": 0.7, "label": "10K, t=0.7, mixed identity"},
        {"gen": "mixed", "n": 10_000, "families": 50,
         "threshold": 0.5, "label": "10K, t=0.5, mixed identity"},
    ]

    for scenario in scenarios:
        n = scenario["n"]
        families = scenario["families"]
        threshold = scenario["threshold"]
        label = scenario["label"]
        k = 5

        print(f"\n{'─' * 90}")
        print(f"Scenario: {label}")
        print(f"{'─' * 90}")

        fasta_path = tmp_dir / f"bench_{label.replace(' ', '_').replace(',', '')}.fasta"
        if scenario["gen"] == "mixed":
            generate_mixed_fasta(fasta_path, n, 200, families)
        else:
            generate_fasta(fasta_path, n, 200, families, identity=scenario["identity"])
        dataset = read_sequences(fasta_path, "protein")

        sketches = compute_sketches(
            dataset.encoded_sequences, dataset.lengths, k, 128, "protein",
        )
        lsh_t = 64 if threshold < 0.6 else 32
        candidates = lsh_candidates(sketches, lsh_t, 1)
        band_width = max(20, int(dataset.max_length * 0.2))

        print(f"  {len(candidates):,} candidate pairs from LSH")
        if len(candidates) == 0:
            continue

        # ── A: No early termination (threshold=0) ──
        t0 = time.perf_counter()
        sims_a, mask_a = _batch_align(
            candidates, dataset.encoded_sequences, dataset.lengths,
            np.float32(0.0), int32(band_width),
        )
        time_no_et = time.perf_counter() - t0
        n_above_a = int(np.sum(sims_a >= threshold))

        # ── B: With early termination ──
        t0 = time.perf_counter()
        sims_b, mask_b = _batch_align(
            candidates, dataset.encoded_sequences, dataset.lengths,
            np.float32(threshold), int32(band_width),
        )
        time_with_et = time.perf_counter() - t0
        n_above_b = int(np.sum(mask_b))

        # Verify early-terminated results match
        above_a = sims_a >= threshold
        above_b = mask_b
        # For pairs above threshold, results should match exactly
        pairs_match = np.array_equal(above_a, above_b)
        # For pairs above threshold, identity values should match
        if n_above_a > 0 and pairs_match:
            max_diff = float(np.max(np.abs(sims_a[above_a] - sims_b[above_b])))
        else:
            max_diff = 0.0

        et_speedup = time_no_et / time_with_et if time_with_et > 0 else float("inf")

        print(f"\n  A) No early termination: {time_no_et:.4f}s  ({n_above_a:,} pairs ≥ threshold)")
        print(f"  B) With early termination: {time_with_et:.4f}s  ({n_above_b:,} pairs ≥ threshold)")
        print(f"  Early termination speedup: {et_speedup:.2f}x")
        print(f"  Results match: {pairs_match}  (max identity diff: {max_diff:.6f})")

        # ── C: Pre-filter + early termination ──
        expected_kmer_sim = threshold ** k
        if expected_kmer_sim >= 0.1:
            kmer_prefilter = expected_kmer_sim * 0.1

            t0 = time.perf_counter()
            prefiltered, _ = compute_pairwise_jaccard(
                candidates, sketches, kmer_prefilter,
            )
            pf_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            sims_c, mask_c = _batch_align(
                prefiltered, dataset.encoded_sequences, dataset.lengths,
                np.float32(threshold), int32(band_width),
            )
            align_time = time.perf_counter() - t0
            total_c = pf_time + align_time
            n_above_c = int(np.sum(mask_c))

            filter_rate = 1.0 - len(prefiltered) / len(candidates)
            combined_speedup = time_no_et / total_c if total_c > 0 else float("inf")

            print(f"\n  C) Pre-filter ({kmer_prefilter:.4f}) + early termination:")
            print(f"     Filtered: {len(candidates):,} → {len(prefiltered):,} ({filter_rate:.1%} removed) [{pf_time:.4f}s]")
            print(f"     Alignment: {n_above_c:,} pairs ≥ threshold [{align_time:.4f}s]")
            print(f"     Total: {total_c:.4f}s")
            print(f"     Combined speedup vs A: {combined_speedup:.2f}x")
            if n_above_c != n_above_a:
                print(f"     WARNING: lost {n_above_a - n_above_c} pairs vs no-filter!")
        else:
            print(f"\n  C) Pre-filter skipped (k-mer Jaccard {expected_kmer_sim:.4f} < 0.1, unreliable)")


if __name__ == "__main__":
    main()
