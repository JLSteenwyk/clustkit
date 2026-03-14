"""Correctness validation: ClustKIT banded NW vs full NW.

Samples pairs and compares banded NW identity to full (unbanded) NW.
C7 from the publication plan.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.pairwise import _nw_identity
from clustkit.io import read_sequences

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("bench_correctness")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_h)


# ===================================================================
# Pair sampling
# ===================================================================

def sample_pairs(n: int, num_pairs: int, seed: int = 42) -> np.ndarray:
    """Sample random (i, j) pairs with i < j from n sequences."""
    rng = np.random.RandomState(seed)
    pairs = set()
    attempts = 0
    max_attempts = num_pairs * 20

    while len(pairs) < num_pairs and attempts < max_attempts:
        batch_size = min(num_pairs * 3, max_attempts - attempts)
        aa = rng.randint(0, n, size=batch_size)
        bb = rng.randint(0, n, size=batch_size)
        for a, b in zip(aa, bb):
            if a != b:
                pairs.add((min(a, b), max(a, b)))
                if len(pairs) >= num_pairs:
                    break
        attempts += batch_size

    pairs = np.array(sorted(pairs), dtype=np.int32)
    return pairs


# ===================================================================
# Identity computation helpers
# ===================================================================

def compute_full_identity(seq_a, len_a, seq_b, len_b) -> float:
    """Compute identity using full (unbanded) NW."""
    # band_width=0 triggers full DP in _nw_identity
    # threshold=0.0 disables early termination
    return float(_nw_identity(seq_a, len_a, seq_b, len_b, 0, 0.0))


def compute_banded_identity(seq_a, len_a, seq_b, len_b, band_width) -> float:
    """Compute identity using banded NW with a fixed band width."""
    return float(_nw_identity(seq_a, len_a, seq_b, len_b, band_width, 0.0))


def compute_adaptive_identity(seq_a, len_a, seq_b, len_b, max_band_width) -> float:
    """Compute identity using adaptive banded NW (same logic as pipeline)."""
    len_diff = abs(int(len_a) - int(len_b))
    adaptive_band = min(max_band_width, max(10, len_diff + 10))
    return float(_nw_identity(seq_a, len_a, seq_b, len_b, adaptive_band, 0.0))


def _try_biopython_identity(seq_a_str: str, seq_b_str: str) -> float | None:
    """Attempt to compute identity using BioPython pairwise2 (global alignment).

    Returns None if BioPython is not available.
    """
    try:
        from Bio.Align import PairwiseAligner
    except ImportError:
        return None

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1

    alignments = aligner.align(seq_a_str, seq_b_str)
    if not alignments:
        return 0.0

    aln = alignments[0]
    # Count matches from alignment
    aligned = aln.format().split("\n")
    # PairwiseAligner format: seq1 / middle / seq2
    if len(aligned) >= 3:
        seq1_aln = aligned[0]
        seq2_aln = aligned[2]
        matches = sum(
            1 for c1, c2 in zip(seq1_aln, seq2_aln)
            if c1 == c2 and c1 != "-"
        )
        shorter = min(len(seq_a_str), len(seq_b_str))
        return matches / shorter if shorter > 0 else 0.0
    return None


# ===================================================================
# Main benchmark
# ===================================================================

def run_benchmark(
    input_fasta: Path,
    mode: str,
    num_pairs: int,
    seed: int,
    output_json: Path | None = None,
):
    """Run the correctness validation benchmark."""
    log.info(f"Loading sequences from {input_fasta} ...")
    dataset = read_sequences(input_fasta, mode)
    n = dataset.num_sequences
    log.info(f"  Loaded {n} sequences (max length {dataset.max_length})")

    if n < 2:
        log.error("Need at least 2 sequences.")
        return

    # Adjust num_pairs if dataset is small
    max_possible = n * (n - 1) // 2
    if num_pairs > max_possible:
        log.warning(f"Requested {num_pairs} pairs but only {max_possible} possible; using all.")
        num_pairs = max_possible

    log.info(f"Sampling {num_pairs} pairs ...")
    pairs = sample_pairs(n, num_pairs, seed)
    actual_pairs = len(pairs)
    log.info(f"  Sampled {actual_pairs} unique pairs")

    seqs = dataset.encoded_sequences
    lens = dataset.lengths

    # Pipeline default band width: p95_len * 0.3
    p95_len = int(np.percentile(lens, 95))
    default_band = max(20, int(p95_len * 0.3))
    log.info(f"  p95 length = {p95_len}, default band_width = {default_band}")

    # ---------------------------------------------------------------------------
    # Compute identities
    # ---------------------------------------------------------------------------
    full_ids = np.zeros(actual_pairs, dtype=np.float64)
    banded_fixed_ids = np.zeros(actual_pairs, dtype=np.float64)
    banded_adaptive_ids = np.zeros(actual_pairs, dtype=np.float64)

    log.info("Computing full NW identities ...")
    t0 = time.perf_counter()
    for idx in range(actual_pairs):
        i, j = pairs[idx]
        # Ensure shorter is seq_a (as the pipeline does)
        if lens[i] <= lens[j]:
            full_ids[idx] = compute_full_identity(seqs[i], lens[i], seqs[j], lens[j])
        else:
            full_ids[idx] = compute_full_identity(seqs[j], lens[j], seqs[i], lens[i])
    full_time = time.perf_counter() - t0
    log.info(f"  Full NW: {full_time:.2f}s")

    log.info(f"Computing banded NW identities (fixed band={default_band}) ...")
    t0 = time.perf_counter()
    for idx in range(actual_pairs):
        i, j = pairs[idx]
        if lens[i] <= lens[j]:
            banded_fixed_ids[idx] = compute_banded_identity(
                seqs[i], lens[i], seqs[j], lens[j], default_band
            )
        else:
            banded_fixed_ids[idx] = compute_banded_identity(
                seqs[j], lens[j], seqs[i], lens[i], default_band
            )
    banded_fixed_time = time.perf_counter() - t0
    log.info(f"  Banded fixed: {banded_fixed_time:.2f}s")

    log.info(f"Computing banded NW identities (adaptive, max_band={default_band}) ...")
    t0 = time.perf_counter()
    for idx in range(actual_pairs):
        i, j = pairs[idx]
        if lens[i] <= lens[j]:
            banded_adaptive_ids[idx] = compute_adaptive_identity(
                seqs[i], lens[i], seqs[j], lens[j], default_band
            )
        else:
            banded_adaptive_ids[idx] = compute_adaptive_identity(
                seqs[j], lens[j], seqs[i], lens[i], default_band
            )
    adaptive_time = time.perf_counter() - t0
    log.info(f"  Banded adaptive: {adaptive_time:.2f}s")

    # ---------------------------------------------------------------------------
    # BioPython comparison (optional, on a small subset)
    # ---------------------------------------------------------------------------
    biopython_available = False
    biopython_ids = None
    try:
        from Bio.Align import PairwiseAligner  # noqa: F401
        biopython_available = True
    except ImportError:
        pass

    bio_subset_size = min(500, actual_pairs)
    if biopython_available:
        log.info(f"Computing BioPython identities on {bio_subset_size} pairs ...")
        biopython_ids = np.zeros(bio_subset_size, dtype=np.float64)
        t0 = time.perf_counter()
        for idx in range(bio_subset_size):
            i, j = pairs[idx]
            seq_a_str = dataset.records[i].sequence
            seq_b_str = dataset.records[j].sequence
            result = _try_biopython_identity(seq_a_str, seq_b_str)
            biopython_ids[idx] = result if result is not None else np.nan
        bio_time = time.perf_counter() - t0
        log.info(f"  BioPython: {bio_time:.2f}s")
    else:
        log.info("BioPython not available; skipping BioPython comparison.")

    # ---------------------------------------------------------------------------
    # Analysis: banded fixed vs full
    # ---------------------------------------------------------------------------
    diff_fixed = banded_fixed_ids - full_ids
    abs_diff_fixed = np.abs(diff_fixed)

    corr_fixed = np.corrcoef(full_ids, banded_fixed_ids)[0, 1] if actual_pairs > 1 else 1.0
    mae_fixed = float(np.mean(abs_diff_fixed))
    max_ae_fixed = float(np.max(abs_diff_fixed))
    underest_fixed = int(np.sum(banded_fixed_ids < full_ids - 1e-6))

    # ---------------------------------------------------------------------------
    # Analysis: banded adaptive vs full
    # ---------------------------------------------------------------------------
    diff_adaptive = banded_adaptive_ids - full_ids
    abs_diff_adaptive = np.abs(diff_adaptive)

    corr_adaptive = np.corrcoef(full_ids, banded_adaptive_ids)[0, 1] if actual_pairs > 1 else 1.0
    mae_adaptive = float(np.mean(abs_diff_adaptive))
    max_ae_adaptive = float(np.max(abs_diff_adaptive))
    underest_adaptive = int(np.sum(banded_adaptive_ids < full_ids - 1e-6))

    # ---------------------------------------------------------------------------
    # Histogram of differences
    # ---------------------------------------------------------------------------
    bins = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    hist_fixed, _ = np.histogram(abs_diff_fixed, bins=bins)
    hist_adaptive, _ = np.histogram(abs_diff_adaptive, bins=bins)

    # ---------------------------------------------------------------------------
    # Print report
    # ---------------------------------------------------------------------------
    print()
    print("=" * 100)
    print("CORRECTNESS VALIDATION: ClustKIT Banded NW vs Full NW (C7)")
    print("=" * 100)
    print(f"Dataset: {input_fasta} ({n} sequences, max length {dataset.max_length})")
    print(f"Pairs sampled: {actual_pairs}")
    print(f"Band width (default): {default_band}  (p95 length {p95_len} x 0.3)")
    print()

    # Banded fixed
    print("-" * 100)
    print("BANDED FIXED vs FULL NW")
    print("-" * 100)
    print(f"  Pearson correlation:       {corr_fixed:.8f}")
    print(f"  Mean absolute error:       {mae_fixed:.8f}")
    print(f"  Max absolute error:        {max_ae_fixed:.8f}")
    print(f"  Pairs where banded < full: {underest_fixed} / {actual_pairs} "
          f"({100.0 * underest_fixed / actual_pairs:.2f}%)")
    print(f"  Timing: full={full_time:.2f}s, banded_fixed={banded_fixed_time:.2f}s "
          f"(speedup={full_time / banded_fixed_time:.2f}x)")

    print(f"\n  Absolute difference histogram (banded fixed):")
    print(f"  {'Bin':>20}  {'Count':>8}  {'Fraction':>10}")
    for k in range(len(hist_fixed)):
        lo, hi = bins[k], bins[k + 1]
        frac = hist_fixed[k] / actual_pairs if actual_pairs > 0 else 0
        print(f"  {f'[{lo:.3f}, {hi:.3f})':>20}  {hist_fixed[k]:>8}  {frac:>10.4f}")

    # Banded adaptive
    print()
    print("-" * 100)
    print("BANDED ADAPTIVE vs FULL NW")
    print("-" * 100)
    print(f"  Pearson correlation:       {corr_adaptive:.8f}")
    print(f"  Mean absolute error:       {mae_adaptive:.8f}")
    print(f"  Max absolute error:        {max_ae_adaptive:.8f}")
    print(f"  Pairs where banded < full: {underest_adaptive} / {actual_pairs} "
          f"({100.0 * underest_adaptive / actual_pairs:.2f}%)")
    print(f"  Timing: full={full_time:.2f}s, banded_adaptive={adaptive_time:.2f}s "
          f"(speedup={full_time / adaptive_time:.2f}x)")

    print(f"\n  Absolute difference histogram (banded adaptive):")
    print(f"  {'Bin':>20}  {'Count':>8}  {'Fraction':>10}")
    for k in range(len(hist_adaptive)):
        lo, hi = bins[k], bins[k + 1]
        frac = hist_adaptive[k] / actual_pairs if actual_pairs > 0 else 0
        print(f"  {f'[{lo:.3f}, {hi:.3f})':>20}  {hist_adaptive[k]:>8}  {frac:>10.4f}")

    # BioPython comparison
    bio_results = {}
    if biopython_available and biopython_ids is not None:
        valid_mask = ~np.isnan(biopython_ids)
        valid_bio = biopython_ids[valid_mask]
        valid_full = full_ids[:bio_subset_size][valid_mask]
        valid_banded = banded_adaptive_ids[:bio_subset_size][valid_mask]

        if len(valid_bio) > 1:
            corr_bio_full = np.corrcoef(valid_bio, valid_full)[0, 1]
            corr_bio_banded = np.corrcoef(valid_bio, valid_banded)[0, 1]
            mae_bio_full = float(np.mean(np.abs(valid_bio - valid_full)))
            mae_bio_banded = float(np.mean(np.abs(valid_bio - valid_banded)))
        else:
            corr_bio_full = corr_bio_banded = float("nan")
            mae_bio_full = mae_bio_banded = float("nan")

        print()
        print("-" * 100)
        print(f"BIOPYTHON COMPARISON ({len(valid_bio)} valid pairs)")
        print("-" * 100)
        print(f"  BioPython vs Full NW:     corr={corr_bio_full:.6f}, MAE={mae_bio_full:.6f}")
        print(f"  BioPython vs Adaptive NW: corr={corr_bio_banded:.6f}, MAE={mae_bio_banded:.6f}")

        bio_results = {
            "n_valid_pairs": int(len(valid_bio)),
            "corr_bio_vs_full": round(float(corr_bio_full), 8),
            "corr_bio_vs_adaptive": round(float(corr_bio_banded), 8),
            "mae_bio_vs_full": round(float(mae_bio_full), 8),
            "mae_bio_vs_adaptive": round(float(mae_bio_banded), 8),
        }

    # Pipeline-realistic accuracy: only pairs that pass the length pre-filters
    # In the pipeline, pairs are skipped if:
    #   1. min(len_i, len_j) / max(len_i, len_j) < threshold (length-ratio pre-filter)
    #   2. abs(len_i - len_j) > band_width (band reach pre-filter)
    # We analyze accuracy at several thresholds for pairs that would enter alignment.
    print()
    print("-" * 100)
    print("PIPELINE-REALISTIC ACCURACY (pairs passing length pre-filters)")
    print("-" * 100)
    print("  Pairs are filtered by: len_ratio >= threshold AND len_diff <= band_width")
    print()
    print(f"  {'Threshold':>10} {'Eligible':>10} {'MAE':>12} {'MaxAE':>12} {'Underest':>10} {'Exact(<.001)':>14}")
    print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*14}")

    pipeline_realistic_results = {}
    for test_t in [0.3, 0.4, 0.5, 0.7, 0.9]:
        # Apply the same pre-filters the pipeline uses
        eligible = np.zeros(actual_pairs, dtype=bool)
        for idx in range(actual_pairs):
            i, j = pairs[idx]
            len_i, len_j = int(lens[i]), int(lens[j])
            shorter = min(len_i, len_j)
            longer = max(len_i, len_j)
            len_diff = abs(len_i - len_j)
            if longer > 0 and shorter / longer >= test_t and len_diff <= default_band:
                eligible[idx] = True

        n_eligible = int(np.sum(eligible))
        if n_eligible > 0:
            e_abs_diff = np.abs(banded_adaptive_ids[eligible] - full_ids[eligible])
            e_mae = float(np.mean(e_abs_diff))
            e_max = float(np.max(e_abs_diff))
            e_underest = int(np.sum(banded_adaptive_ids[eligible] < full_ids[eligible] - 1e-6))
            e_exact = int(np.sum(e_abs_diff < 0.001))
            exact_frac = e_exact / n_eligible

            pipeline_realistic_results[str(test_t)] = {
                "n_eligible": n_eligible,
                "mae": round(e_mae, 8),
                "max_ae": round(e_max, 8),
                "underestimate_count": e_underest,
                "exact_fraction": round(exact_frac, 6),
            }

            print(f"  {test_t:>10.1f} {n_eligible:>10} {e_mae:>12.8f} {e_max:>12.8f} "
                  f"{e_underest:>10} {exact_frac:>13.1%}")
        else:
            print(f"  {test_t:>10.1f} {0:>10} {'N/A':>12} {'N/A':>12} {'N/A':>10} {'N/A':>14}")

    # Identity distribution
    print()
    print("-" * 100)
    print("IDENTITY DISTRIBUTION (full NW)")
    print("-" * 100)
    id_percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    for p in id_percentiles:
        val = np.percentile(full_ids, p)
        print(f"  P{p:>3}: {val:.4f}")
    print(f"  Mean: {np.mean(full_ids):.4f}")
    nonzero_frac = np.sum(full_ids > 0) / actual_pairs if actual_pairs > 0 else 0
    print(f"  Non-zero: {np.sum(full_ids > 0)} / {actual_pairs} ({100 * nonzero_frac:.1f}%)")

    print()
    print("=" * 100)

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    results = {
        "dataset": str(input_fasta),
        "n_sequences": n,
        "n_pairs": actual_pairs,
        "band_width": default_band,
        "p95_length": p95_len,
        "banded_fixed_vs_full": {
            "correlation": round(float(corr_fixed), 8),
            "mean_abs_error": round(mae_fixed, 8),
            "max_abs_error": round(max_ae_fixed, 8),
            "underestimate_count": underest_fixed,
            "underestimate_fraction": round(underest_fixed / actual_pairs, 6) if actual_pairs > 0 else 0,
            "timing_full_s": round(full_time, 3),
            "timing_banded_s": round(banded_fixed_time, 3),
        },
        "banded_adaptive_vs_full": {
            "correlation": round(float(corr_adaptive), 8),
            "mean_abs_error": round(mae_adaptive, 8),
            "max_abs_error": round(max_ae_adaptive, 8),
            "underestimate_count": underest_adaptive,
            "underestimate_fraction": round(underest_adaptive / actual_pairs, 6) if actual_pairs > 0 else 0,
            "timing_full_s": round(full_time, 3),
            "timing_banded_s": round(adaptive_time, 3),
        },
        "histogram_bins": bins,
        "histogram_fixed": hist_fixed.tolist(),
        "histogram_adaptive": hist_adaptive.tolist(),
    }

    if bio_results:
        results["biopython_comparison"] = bio_results

    if pipeline_realistic_results:
        results["pipeline_realistic"] = pipeline_realistic_results

    if output_json is None:
        output_json = Path(__file__).resolve().parent / "data" / "correctness_results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info(f"Results saved to {output_json}")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Correctness validation: ClustKIT banded NW vs full NW (C7).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Input FASTA file (Pfam dataset or any protein FASTA).",
    )
    parser.add_argument(
        "--mode", type=str, default="protein",
        choices=["protein", "nucleotide"],
        help="Sequence type (default: protein).",
    )
    parser.add_argument(
        "--num-pairs", type=int, default=10_000,
        help="Number of random pairs to sample (default: 10000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for pair sampling (default: 42).",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output JSON file for results (default: benchmarks/data/correctness_results.json).",
    )

    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")

    run_benchmark(
        input_fasta=args.input,
        mode=args.mode,
        num_pairs=args.num_pairs,
        seed=args.seed,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
