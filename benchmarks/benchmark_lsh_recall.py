"""LSH recall benchmark: measures what fraction of true pairs LSH discovers.

Varies num_tables to show recall vs candidate count trade-off.
C8 from the publication plan.
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

from clustkit.lsh import lsh_candidates
from clustkit.sketch import compute_sketches
from clustkit.pairwise import _nw_identity
from clustkit.io import read_sequences
from clustkit.clustering_mode import resolve_clustering_mode
from clustkit.utils import auto_kmer_for_lsh, auto_lsh_params

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("bench_lsh_recall")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_h)


# ===================================================================
# Pair sampling and ground truth
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


def compute_ground_truth(
    pairs: np.ndarray,
    encoded_sequences: np.ndarray,
    lengths: np.ndarray,
    threshold: float,
    band_width: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute true alignment identity for sampled pairs.

    Returns:
        identities: (M,) float array of identities for each pair.
        true_mask: (M,) boolean mask where identity >= threshold (full NW).
        pipeline_mask: (M,) boolean mask where identity >= threshold AND
            the pair passes the pipeline's length pre-filters (length-ratio
            and length-diff).  This represents pairs that the pipeline would
            actually process and return as true positives.
    """
    m = len(pairs)
    identities = np.zeros(m, dtype=np.float64)
    pipeline_eligible = np.zeros(m, dtype=np.bool_)

    for idx in range(m):
        i, j = pairs[idx]
        len_i, len_j = int(lengths[i]), int(lengths[j])

        # Check pipeline length pre-filters
        shorter = min(len_i, len_j)
        longer = max(len_i, len_j)
        len_diff = abs(len_i - len_j)
        if longer > 0 and shorter / longer >= threshold and (band_width <= 0 or len_diff <= band_width):
            pipeline_eligible[idx] = True

        # Shorter as seq_a for efficiency
        if len_i <= len_j:
            identities[idx] = float(_nw_identity(
                encoded_sequences[i], len_i,
                encoded_sequences[j], len_j,
                0, 0.0,  # band_width=0 -> full DP, threshold=0 -> no early term
            ))
        else:
            identities[idx] = float(_nw_identity(
                encoded_sequences[j], len_j,
                encoded_sequences[i], len_i,
                0, 0.0,
            ))

    true_mask = identities >= threshold
    pipeline_mask = true_mask & pipeline_eligible
    return identities, true_mask, pipeline_mask


# ===================================================================
# LSH candidate set lookup
# ===================================================================

def pairs_to_set(pairs: np.ndarray, n: int) -> set[int]:
    """Convert (M, 2) pair array to a set of packed int64 keys for fast lookup."""
    packed = set()
    for idx in range(len(pairs)):
        i, j = int(pairs[idx, 0]), int(pairs[idx, 1])
        if i > j:
            i, j = j, i
        packed.add(i * n + j)
    return packed


# ===================================================================
# Main benchmark
# ===================================================================

def run_benchmark(
    input_fasta: Path,
    threshold: float,
    threads: int,
    sample_pairs_count: int,
    num_tables_list: list[int],
    clustkit_mode: str = "balanced",
    seed: int = 42,
    output_json: Path | None = None,
):
    """Run the LSH recall benchmark."""
    import numba
    numba.set_num_threads(threads)

    log.info(f"Loading sequences from {input_fasta} ...")
    dataset = read_sequences(input_fasta, "protein")
    n = dataset.num_sequences
    log.info(f"  Loaded {n} sequences (max length {dataset.max_length})")

    if n < 2:
        log.error("Need at least 2 sequences.")
        return

    # Adjust sample count
    max_possible = n * (n - 1) // 2
    if sample_pairs_count > max_possible:
        log.warning(
            f"Requested {sample_pairs_count} pairs but only {max_possible} "
            f"possible; using all."
        )
        sample_pairs_count = max_possible

    # ---------------------------------------------------------------------------
    # Compute sketches (using the same k and parameters the pipeline would use)
    # ---------------------------------------------------------------------------
    k_user = 5
    k_lsh = auto_kmer_for_lsh(threshold, "protein", k_user)
    sketch_size, sensitivity = resolve_clustering_mode(clustkit_mode, threshold)
    lsh_params = auto_lsh_params(threshold, sensitivity, k=k_lsh)
    default_num_bands = lsh_params["num_bands"]

    log.info(
        f"  mode = {clustkit_mode}, k_lsh = {k_lsh}, sketch_size = {sketch_size}, "
        f"sensitivity = {sensitivity}, num_bands = {default_num_bands}"
    )

    log.info("Computing sketches ...")
    t0 = time.perf_counter()
    sketches = compute_sketches(
        dataset.encoded_sequences,
        dataset.lengths,
        k_lsh,
        sketch_size,
        "protein",
    )
    sketch_time = time.perf_counter() - t0
    log.info(f"  Sketches computed in {sketch_time:.2f}s")

    # ---------------------------------------------------------------------------
    # Establish ground truth
    # ---------------------------------------------------------------------------
    log.info(f"Sampling {sample_pairs_count} random pairs for ground truth ...")
    gt_pairs = sample_pairs(n, sample_pairs_count, seed)
    actual_sample = len(gt_pairs)
    log.info(f"  Sampled {actual_sample} unique pairs")

    # Compute pipeline band_width for length pre-filter check
    p95_len = int(np.percentile(dataset.lengths, 95))
    pipeline_band_width = max(20, int(p95_len * 0.3))

    log.info("Computing ground truth identities (full NW) ...")
    t0 = time.perf_counter()
    gt_identities, gt_true_mask, gt_pipeline_mask = compute_ground_truth(
        gt_pairs, dataset.encoded_sequences, dataset.lengths, threshold,
        band_width=pipeline_band_width,
    )
    gt_time = time.perf_counter() - t0

    n_true_pairs = int(np.sum(gt_true_mask))
    n_pipeline_pairs = int(np.sum(gt_pipeline_mask))
    log.info(
        f"  Ground truth computed in {gt_time:.2f}s: "
        f"{n_true_pairs} true pairs (full NW), "
        f"{n_pipeline_pairs} pipeline-realistic true pairs "
        f"(pass length pre-filters) out of {actual_sample} sampled"
    )

    if n_true_pairs == 0:
        log.warning(
            "No true pairs found at this threshold. Consider lowering --threshold "
            "or using a dataset with more similar sequences."
        )

    # Build set of true pair keys for fast lookup
    true_pair_keys = set()
    pipeline_pair_keys = set()
    for idx in range(actual_sample):
        i, j = int(gt_pairs[idx, 0]), int(gt_pairs[idx, 1])
        if i > j:
            i, j = j, i
        key = i * n + j
        if gt_true_mask[idx]:
            true_pair_keys.add(key)
        if gt_pipeline_mask[idx]:
            pipeline_pair_keys.add(key)

    # ---------------------------------------------------------------------------
    # Run LSH with varying num_tables
    # ---------------------------------------------------------------------------
    total_possible_pairs = n * (n - 1) // 2
    results_per_tables = []

    print()
    print("=" * 120)
    print(f"LSH RECALL BENCHMARK (C8) | threshold={threshold} | "
          f"n={n} | true_pairs={n_true_pairs} (pipeline-realistic: {n_pipeline_pairs}) / {actual_sample}")
    print("=" * 120)
    print(
        f"{'Tables':>8} {'Bands':>6} {'Candidates':>12} {'Cand_Rate':>12} "
        f"{'True_Found':>12} {'Recall':>10} {'PipeFound':>10} {'PipeRecall':>10} "
        f"{'LSH_Time':>10}"
    )
    print("-" * 120)

    for num_tables in sorted(num_tables_list):
        log.info(f"Running LSH with num_tables={num_tables} ...")

        t0 = time.perf_counter()
        candidate_pairs = lsh_candidates(
            sketches,
            num_tables=num_tables,
            num_bands=default_num_bands,
        )
        lsh_time = time.perf_counter() - t0

        n_candidates = len(candidate_pairs)
        candidate_rate = n_candidates / total_possible_pairs if total_possible_pairs > 0 else 0

        # Build candidate set for lookup
        cand_set = pairs_to_set(candidate_pairs, n)

        # Check how many true pairs are in the candidate set
        true_found = 0
        for key in true_pair_keys:
            if key in cand_set:
                true_found += 1

        # Pipeline-realistic recall: how many length-compatible true pairs does LSH find?
        pipeline_found = 0
        for key in pipeline_pair_keys:
            if key in cand_set:
                pipeline_found += 1

        recall = true_found / n_true_pairs if n_true_pairs > 0 else 1.0
        pipeline_recall = pipeline_found / n_pipeline_pairs if n_pipeline_pairs > 0 else 1.0

        # False positives among candidates (candidates that are not true pairs)
        # Note: we can only estimate this w.r.t. the sampled pairs
        sampled_cand_keys = set()
        for idx in range(actual_sample):
            i, j = int(gt_pairs[idx, 0]), int(gt_pairs[idx, 1])
            if i > j:
                i, j = j, i
            key = i * n + j
            if key in cand_set:
                sampled_cand_keys.add(key)

        fp_in_sample = len(sampled_cand_keys) - true_found
        total_neg_in_sample = actual_sample - n_true_pairs
        fp_rate = fp_in_sample / total_neg_in_sample if total_neg_in_sample > 0 else 0

        row = {
            "num_tables": num_tables,
            "num_bands": default_num_bands,
            "n_candidates": n_candidates,
            "candidate_rate": round(candidate_rate, 8),
            "true_found": true_found,
            "recall": round(recall, 6),
            "pipeline_found": pipeline_found,
            "pipeline_recall": round(pipeline_recall, 6),
            "fp_in_sample": fp_in_sample,
            "fp_rate": round(fp_rate, 6),
            "lsh_time_s": round(lsh_time, 3),
        }
        results_per_tables.append(row)

        print(
            f"{num_tables:>8} {default_num_bands:>6} {n_candidates:>12} "
            f"{candidate_rate:>12.6f} {true_found:>12} {recall:>10.4f} "
            f"{pipeline_found:>10} {pipeline_recall:>10.4f} {lsh_time:>10.2f}s"
        )

    print()
    print("=" * 120)

    # ---------------------------------------------------------------------------
    # Identity distribution of ground truth pairs
    # ---------------------------------------------------------------------------
    print()
    print("GROUND TRUTH IDENTITY DISTRIBUTION")
    print("-" * 60)
    percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    for p in percentiles:
        val = np.percentile(gt_identities, p)
        print(f"  P{p:>3}: {val:.4f}")
    print(f"  Mean: {np.mean(gt_identities):.4f}")
    nonzero = np.sum(gt_identities > 0)
    print(f"  Non-zero: {nonzero}/{actual_sample} ({100 * nonzero / actual_sample:.1f}%)")
    print(f"  >= threshold ({threshold}): {n_true_pairs}/{actual_sample} "
          f"({100 * n_true_pairs / actual_sample:.1f}%)")

    # Identity distribution of found vs missed true pairs
    if n_true_pairs > 0:
        # Build the candidate set for the highest table count ONCE
        last_cand_set = pairs_to_set(
            lsh_candidates(
                sketches,
                num_tables=max(num_tables_list),
                num_bands=default_num_bands,
            ),
            n,
        )
        found_ids = []
        missed_ids = []
        for idx in range(actual_sample):
            if gt_true_mask[idx]:
                i, j = int(gt_pairs[idx, 0]), int(gt_pairs[idx, 1])
                if i > j:
                    i, j = j, i
                key = i * n + j
                if key in last_cand_set:
                    found_ids.append(gt_identities[idx])
                else:
                    missed_ids.append(gt_identities[idx])

        print()
        print(f"TRUE PAIRS: found vs missed (at max tables={max(num_tables_list)})")
        print("-" * 60)
        if found_ids:
            found_ids = np.array(found_ids)
            print(f"  Found ({len(found_ids)}): mean={np.mean(found_ids):.4f}, "
                  f"min={np.min(found_ids):.4f}, max={np.max(found_ids):.4f}")
        if missed_ids:
            missed_ids = np.array(missed_ids)
            print(f"  Missed ({len(missed_ids)}): mean={np.mean(missed_ids):.4f}, "
                  f"min={np.min(missed_ids):.4f}, max={np.max(missed_ids):.4f}")
        else:
            print("  Missed: 0 (perfect recall)")

    print()

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    full_results = {
        "dataset": str(input_fasta),
        "n_sequences": n,
        "threshold": threshold,
        "clustkit_mode": clustkit_mode,
        "k_lsh": k_lsh,
        "sketch_size": sketch_size,
        "sensitivity": sensitivity,
        "num_bands": default_num_bands,
        "n_sampled_pairs": actual_sample,
        "n_true_pairs": n_true_pairs,
        "n_pipeline_realistic_true_pairs": n_pipeline_pairs,
        "pipeline_band_width": pipeline_band_width,
        "sketch_time_s": round(sketch_time, 3),
        "gt_compute_time_s": round(gt_time, 3),
        "results": results_per_tables,
    }

    if output_json is None:
        output_json = Path(__file__).resolve().parent / "data" / "lsh_recall_results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as fh:
        json.dump(full_results, fh, indent=2)
    log.info(f"Results saved to {output_json}")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LSH recall benchmark: recall vs candidate count trade-off (C8).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Input FASTA file (Pfam dataset or similar, ~22K sequences).",
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.5,
        help="Identity threshold for defining true pairs (default: 0.5).",
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="Number of CPU threads (default: 4).",
    )
    parser.add_argument(
        "--sample-pairs", type=int, default=50_000,
        help="Number of random pairs to sample for ground truth (default: 50000).",
    )
    parser.add_argument(
        "--num-tables", type=int, nargs="+",
        default=[8, 16, 32, 64, 128, 256, 512],
        help="LSH num_tables values to sweep (default: 8 16 32 64 128 256 512).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output JSON file for results (default: benchmarks/data/lsh_recall_results.json).",
    )
    parser.add_argument(
        "--clustkit-mode", type=str, default="balanced",
        choices=["balanced", "accurate", "fast"],
        help="ClustKIT clustering mode to benchmark.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")

    run_benchmark(
        input_fasta=args.input,
        threshold=args.threshold,
        threads=args.threads,
        sample_pairs_count=args.sample_pairs,
        num_tables_list=sorted(args.num_tables),
        clustkit_mode=args.clustkit_mode,
        seed=args.seed,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
