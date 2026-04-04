"""Ablation study benchmark for ClustKIT.

Tests the contribution of each ClustKIT optimization by disabling them
one at a time on Pfam data at t=0.4 (where differences are largest).

Variants tested:
  1. Full ClustKIT          — baseline with all optimizations
  2. No adaptive k          — fixed k=5 instead of auto_kmer_for_lsh(t=0.4) -> k=3
  3. No adaptive band       — fixed large band width for all pairs
  4. No early termination   — threshold=0.001 in DP (near-zero early exit)
  5. No length pre-filter   — skip length-ratio and length-diff checks
  6. k-mer mode only        — alignment="kmer" with k=5 instead of NW alignment
  7. Fixed LSH params       — 32 tables, 2 bands regardless of threshold

C6 from the publication plan.
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.pipeline import run_pipeline
from clustkit.clustering_mode import resolve_clustering_mode

# Import the benchmark_pfam_concordance helpers for data loading
from benchmark_pfam_concordance import load_and_mix_families


DATA_DIR = Path(__file__).resolve().parent / "data" / "pfam_families"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "ablation_results"


# ──────────────────────────────────────────────────────────────────────
# Evaluation helpers (same as in benchmark_pfam_concordance.py)
# ──────────────────────────────────────────────────────────────────────

def pairwise_precision_recall_f1(true_labels, pred_labels):
    """Compute pairwise precision, recall, and F1."""
    n = len(true_labels)

    if n > 5000:
        num_samples = 2_000_000
        rng = np.random.RandomState(42)
        idx_a = rng.randint(0, n, size=num_samples)
        idx_b = rng.randint(0, n, size=num_samples)
        valid = idx_a != idx_b
        idx_a = idx_a[valid]
        idx_b = idx_b[valid]
    else:
        idx_a, idx_b = [], []
        for i in range(n):
            for j in range(i + 1, n):
                idx_a.append(i)
                idx_b.append(j)
        idx_a = np.array(idx_a)
        idx_b = np.array(idx_b)

    same_pred = pred_labels[idx_a] == pred_labels[idx_b]
    same_true = true_labels[idx_a] == true_labels[idx_b]

    tp = int(np.sum(same_pred & same_true))
    fp = int(np.sum(same_pred & ~same_true))
    fn = int(np.sum(~same_pred & same_true))
    tn = int(np.sum(~same_pred & ~same_true))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "pairwise_precision": round(precision, 4),
        "pairwise_recall": round(recall, 4),
        "pairwise_F1": round(f1, 4),
    }


def _normalize_id(seq_id):
    """Extract UniProt accession from sp|ACC|NAME format, or return as-is."""
    parts = seq_id.split("|")
    if len(parts) >= 2:
        return parts[1]
    return seq_id


def evaluate_clusters(ground_truth, pred_clusters):
    """Evaluate predicted clusters against ground truth Pfam labels."""
    common_ids = sorted(set(ground_truth.keys()) & set(pred_clusters.keys()))
    if not common_ids:
        gt_norm = {_normalize_id(k): v for k, v in ground_truth.items()}
        pred_norm = {_normalize_id(k): v for k, v in pred_clusters.items()}
        common_ids = sorted(set(gt_norm.keys()) & set(pred_norm.keys()))
        if not common_ids:
            return {"error": "no common sequences"}
        ground_truth = gt_norm
        pred_clusters = pred_norm

    true_label_list = [ground_truth[sid] for sid in common_ids]
    pred_cluster_list = [pred_clusters[sid] for sid in common_ids]

    label_to_int = {f: i for i, f in enumerate(sorted(set(true_label_list)))}
    true_labels = np.array(
        [label_to_int[f] for f in true_label_list], dtype=np.int32
    )
    pred_labels = np.array(pred_cluster_list, dtype=np.int32)

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        true_labels, pred_labels
    )
    pw = pairwise_precision_recall_f1(true_labels, pred_labels)

    return {
        "n_sequences": len(common_ids),
        "n_true_families": len(set(true_label_list)),
        "n_predicted_clusters": len(set(pred_cluster_list)),
        "ARI": round(ari, 4),
        "NMI": round(nmi, 4),
        "homogeneity": round(homogeneity, 4),
        "completeness": round(completeness, 4),
        "V_measure": round(v_measure, 4),
        **pw,
    }


# ──────────────────────────────────────────────────────────────────────
# Ablation variant definitions
# ──────────────────────────────────────────────────────────────────────

def _base_config(mixed_fasta, out_dir, threshold, threads, clustkit_mode):
    """Return the baseline ClustKIT config."""
    sketch_size, sensitivity = resolve_clustering_mode(clustkit_mode, threshold)
    return {
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


def _run_variant_and_evaluate(
    variant_name,
    mixed_fasta,
    ground_truth,
    threshold,
    threads,
    clustkit_mode,
    config_overrides=None,
    mock_patches=None,
):
    """Run a single ablation variant and evaluate.

    Args:
        variant_name: Human-readable variant name.
        mixed_fasta: Path to the mixed FASTA file.
        ground_truth: dict mapping seq_id -> family.
        threshold: Identity threshold.
        threads: Thread count.
        config_overrides: dict of config key overrides.
        mock_patches: list of (target, replacement) for unittest.mock.patch.

    Returns:
        dict of evaluation metrics + runtime.
    """
    out_dir = OUTPUT_DIR / variant_name.lower().replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = _base_config(mixed_fasta, out_dir, threshold, threads, clustkit_mode)
    if config_overrides:
        config.update(config_overrides)

    print(f"  {variant_name}...", end=" ", flush=True)

    start = time.perf_counter()
    try:
        if mock_patches:
            # Apply all patches as a nested context manager stack
            import contextlib

            with contextlib.ExitStack() as stack:
                for target, replacement in mock_patches:
                    stack.enter_context(patch(target, replacement))
                run_pipeline(config)
        else:
            run_pipeline(config)
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"ERROR: {e} ({elapsed:.2f}s)")
        return {
            "variant": variant_name,
            "error": str(e),
            "runtime_seconds": round(elapsed, 2),
        }

    elapsed = time.perf_counter() - start

    # Parse ClustKIT output
    clusters = {}
    tsv_path = out_dir / "clusters.tsv"
    if tsv_path.exists():
        with open(tsv_path) as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    clusters[parts[0]] = int(parts[1])

    if not clusters:
        print(f"NO OUTPUT ({elapsed:.2f}s)")
        return {
            "variant": variant_name,
            "error": "no clusters produced",
            "runtime_seconds": round(elapsed, 2),
        }

    metrics = evaluate_clusters(ground_truth, clusters)
    metrics["variant"] = variant_name
    metrics["runtime_seconds"] = round(elapsed, 2)

    if "error" not in metrics:
        print(
            f"{metrics['n_predicted_clusters']} clusters, "
            f"ARI={metrics['ARI']}, "
            f"F1={metrics['pairwise_F1']}, "
            f"{elapsed:.2f}s"
        )
    else:
        print(f"EVAL ERROR: {metrics['error']} ({elapsed:.2f}s)")

    return metrics


# ──────────────────────────────────────────────────────────────────────
# Monkey-patch factories for ablation
# ──────────────────────────────────────────────────────────────────────

def _make_fixed_kmer_for_lsh(k_value):
    """Return a replacement for auto_kmer_for_lsh that always returns k_value."""
    def fixed_kmer_for_lsh(threshold, mode, user_k):
        return k_value
    return fixed_kmer_for_lsh


def _make_fixed_lsh_params(num_tables, num_bands):
    """Return a replacement for auto_lsh_params with fixed values."""
    def fixed_lsh_params(threshold, sensitivity, k=5):
        return {"num_tables": num_tables, "num_bands": num_bands}
    return fixed_lsh_params


def _make_no_length_prefilter_batch_align():
    """Return replacements for _batch_align and _batch_align_compact that skip length pre-filters.

    Imports the original Numba kernels and wraps with length pre-filter disabled.
    Returns a tuple: (batch_align, batch_align_compact).
    """
    from numba import njit, prange, int32, float32
    from clustkit.pairwise import _nw_identity

    @njit(parallel=True, cache=False)
    def _batch_align_no_prefilter(pairs, sequences, lengths, threshold, band_width):
        m = pairs.shape[0]
        sims = np.empty(m, dtype=np.float32)
        mask = np.empty(m, dtype=np.bool_)

        for idx in prange(m):
            i = pairs[idx, 0]
            j = pairs[idx, 1]
            len_i = lengths[i]
            len_j = lengths[j]

            # Skip length-ratio pre-filter (disabled for ablation)
            # Skip length-diff pre-filter (disabled for ablation)

            len_diff = abs(int32(len_i) - int32(len_j))

            # Adaptive band still active (this variant only disables length filters)
            adaptive_band = min(band_width, max(int32(10), len_diff + int32(10)))

            if len_i <= len_j:
                identity = _nw_identity(
                    sequences[i], len_i,
                    sequences[j], len_j,
                    adaptive_band, threshold,
                )
            else:
                identity = _nw_identity(
                    sequences[j], len_j,
                    sequences[i], len_i,
                    adaptive_band, threshold,
                )
            sims[idx] = identity
            mask[idx] = identity >= threshold

        return sims, mask

    @njit(parallel=True, cache=False)
    def _batch_align_compact_no_prefilter(pairs, flat_sequences, offsets, lengths, threshold, band_width):
        m = pairs.shape[0]
        sims = np.empty(m, dtype=np.float32)
        mask = np.empty(m, dtype=np.bool_)

        for idx in prange(m):
            i = pairs[idx, 0]
            j = pairs[idx, 1]
            len_i = lengths[i]
            len_j = lengths[j]

            # Skip length-ratio pre-filter (disabled for ablation)
            # Skip length-diff pre-filter (disabled for ablation)

            len_diff = abs(int32(len_i) - int32(len_j))
            adaptive_band = min(band_width, max(int32(10), len_diff + int32(10)))

            seq_i = flat_sequences[offsets[i]:offsets[i] + len_i]
            seq_j = flat_sequences[offsets[j]:offsets[j] + len_j]

            if len_i <= len_j:
                identity = _nw_identity(seq_i, len_i, seq_j, len_j, adaptive_band, threshold)
            else:
                identity = _nw_identity(seq_j, len_j, seq_i, len_i, adaptive_band, threshold)
            sims[idx] = identity
            mask[idx] = identity >= threshold

        return sims, mask

    return _batch_align_no_prefilter, _batch_align_compact_no_prefilter


def _make_no_adaptive_band_batch_align():
    """Return replacements for _batch_align and _batch_align_compact that skip per-pair
    adaptive band narrowing. Always uses the full band_width instead of
    min(band_width, max(10, len_diff + 10)).

    Returns a tuple: (batch_align, batch_align_compact).
    """
    from numba import njit, prange, int32, float32
    from clustkit.pairwise import _nw_identity

    @njit(parallel=True, cache=False)
    def _batch_align_no_adaptive(pairs, sequences, lengths, threshold, band_width):
        m = pairs.shape[0]
        sims = np.empty(m, dtype=np.float32)
        mask = np.empty(m, dtype=np.bool_)

        for idx in prange(m):
            i = pairs[idx, 0]
            j = pairs[idx, 1]
            len_i = lengths[i]
            len_j = lengths[j]

            # Keep length pre-filters
            shorter = min(len_i, len_j)
            longer = max(len_i, len_j)
            if longer > 0 and float32(shorter) / float32(longer) < threshold:
                sims[idx] = 0.0
                mask[idx] = False
                continue
            len_diff = abs(int32(len_i) - int32(len_j))
            if len_diff > band_width:
                sims[idx] = 0.0
                mask[idx] = False
                continue

            # NO adaptive band: always use band_width directly
            if len_i <= len_j:
                identity = _nw_identity(
                    sequences[i], len_i,
                    sequences[j], len_j,
                    band_width, threshold,
                )
            else:
                identity = _nw_identity(
                    sequences[j], len_j,
                    sequences[i], len_i,
                    band_width, threshold,
                )
            sims[idx] = identity
            mask[idx] = identity >= threshold

        return sims, mask

    @njit(parallel=True, cache=False)
    def _batch_align_compact_no_adaptive(pairs, flat_sequences, offsets, lengths, threshold, band_width):
        m = pairs.shape[0]
        sims = np.empty(m, dtype=np.float32)
        mask = np.empty(m, dtype=np.bool_)

        for idx in prange(m):
            i = pairs[idx, 0]
            j = pairs[idx, 1]
            len_i = lengths[i]
            len_j = lengths[j]

            # Keep length pre-filters
            shorter = min(len_i, len_j)
            longer = max(len_i, len_j)
            if longer > 0 and float32(shorter) / float32(longer) < threshold:
                sims[idx] = 0.0
                mask[idx] = False
                continue
            len_diff = abs(int32(len_i) - int32(len_j))
            if len_diff > band_width:
                sims[idx] = 0.0
                mask[idx] = False
                continue

            seq_i = flat_sequences[offsets[i]:offsets[i] + len_i]
            seq_j = flat_sequences[offsets[j]:offsets[j] + len_j]

            # NO adaptive band: always use band_width directly
            if len_i <= len_j:
                identity = _nw_identity(seq_i, len_i, seq_j, len_j, band_width, threshold)
            else:
                identity = _nw_identity(seq_j, len_j, seq_i, len_i, band_width, threshold)
            sims[idx] = identity
            mask[idx] = identity >= threshold

        return sims, mask

    return _batch_align_no_adaptive, _batch_align_compact_no_adaptive


# ──────────────────────────────────────────────────────────────────────
# Main benchmark
# ──────────────────────────────────────────────────────────────────────

def run_ablation(threshold=0.4, max_per_family=500, threads=4, clustkit_mode="balanced"):
    """Run the ablation study.

    Args:
        threshold: Identity threshold (default 0.4, where differences are largest).
        max_per_family: Max sequences per Pfam family.
        threads: Number of CPU threads.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print(f"ABLATION STUDY — ClustKIT[{clustkit_mode}] at threshold={threshold} ({threads} threads)")
    print("=" * 120)
    print()

    # Load and mix families
    mixed_fasta, ground_truth = load_and_mix_families(DATA_DIR, max_per_family)
    print()

    all_results = []

    # ── Variant 1: Full ClustKIT (baseline) ──────────────────────────
    result = _run_variant_and_evaluate(
        "Full ClustKIT",
        mixed_fasta, ground_truth, threshold, threads, clustkit_mode,
    )
    all_results.append(result)

    # ── Variant 2: No adaptive k ─────────────────────────────────────
    # At t=0.4, auto_kmer_for_lsh returns k=3. Force k=5 instead.
    result = _run_variant_and_evaluate(
        "No adaptive k",
        mixed_fasta, ground_truth, threshold, threads, clustkit_mode,
        mock_patches=[
            ("clustkit.pipeline.auto_kmer_for_lsh", _make_fixed_kmer_for_lsh(5)),
        ],
    )
    all_results.append(result)

    # ── Variant 3: No adaptive band ──────────────────────────────────
    # Disable per-pair adaptive band narrowing: always use band_width (458)
    # instead of min(band_width, max(10, len_diff + 10)).
    # This shows the value of the per-pair adaptive band optimization.
    no_adaptive_batch, no_adaptive_batch_compact = _make_no_adaptive_band_batch_align()
    result = _run_variant_and_evaluate(
        "No adaptive band",
        mixed_fasta, ground_truth, threshold, threads, clustkit_mode,
        mock_patches=[
            ("clustkit.pairwise._batch_align", no_adaptive_batch),
            ("clustkit.pairwise._batch_align_compact", no_adaptive_batch_compact),
        ],
    )
    all_results.append(result)

    # ── Variant 4: No early termination ──────────────────────────────
    # Pass threshold=0.001 to the pipeline so the DP never terminates early,
    # then evaluate using the real threshold.
    # We achieve this by running with a near-zero threshold, then post-filtering
    # the clusters. Since clustering uses the graph edges (which would all pass
    # at t=0.001), we instead patch the pairwise stage to use threshold=0.001
    # but keep the real threshold for graph edge filtering.
    #
    # Simplest approach: run with t=0.001, parse output, then re-cluster
    # at the real threshold. But this changes the pipeline too much.
    #
    # Cleanest approach: monkey-patch _nw_identity threshold parameter.
    # Actually, the threshold is passed through _batch_align -> _nw_identity.
    # We can monkey-patch compute_pairwise_alignment to pass threshold=0.001
    # to _batch_align, but post-filter results at the real threshold.
    from clustkit import pairwise as pw_module

    original_compute_pairwise_alignment = pw_module.compute_pairwise_alignment

    def compute_pairwise_alignment_no_early_term(
        candidate_pairs, encoded_sequences, lengths, threshold_arg,
        band_width=50, device="cpu", mode="protein", sketches=None,
        flat_sequences=None, offsets=None,
    ):
        """Wrapper that disables early termination by using threshold=0.001."""
        # Run alignment with near-zero threshold (no early termination)
        filtered_pairs, sims = original_compute_pairwise_alignment(
            candidate_pairs, encoded_sequences, lengths,
            threshold=0.001,  # disables early exit
            band_width=band_width,
            device=device,
            mode=mode,
            sketches=sketches,
            flat_sequences=flat_sequences,
            offsets=offsets,
        )
        # Post-filter at the real threshold
        if len(filtered_pairs) > 0:
            real_mask = sims >= threshold_arg
            return filtered_pairs[real_mask], sims[real_mask]
        return filtered_pairs, sims

    result = _run_variant_and_evaluate(
        "No early termination",
        mixed_fasta, ground_truth, threshold, threads, clustkit_mode,
        mock_patches=[
            (
                "clustkit.pipeline.compute_pairwise_alignment",
                compute_pairwise_alignment_no_early_term,
            ),
        ],
    )
    all_results.append(result)

    # ── Variant 5: No length pre-filter ──────────────────────────────
    # Disable both length-ratio and length-diff checks in _batch_align
    # and _batch_align_compact (pipeline uses compact format).
    no_prefilter_batch, no_prefilter_batch_compact = _make_no_length_prefilter_batch_align()

    result = _run_variant_and_evaluate(
        "No length pre-filter",
        mixed_fasta, ground_truth, threshold, threads, clustkit_mode,
        mock_patches=[
            ("clustkit.pairwise._batch_align", no_prefilter_batch),
            ("clustkit.pairwise._batch_align_compact", no_prefilter_batch_compact),
        ],
    )
    all_results.append(result)

    # ── Variant 6: k-mer mode only ───────────────────────────────────
    # Use alignment="kmer" with k=5 instead of NW alignment.
    result = _run_variant_and_evaluate(
        "k-mer mode only",
        mixed_fasta, ground_truth, threshold, threads, clustkit_mode,
        config_overrides={
            "alignment": "kmer",
            "kmer_size": 5,
        },
    )
    all_results.append(result)

    # ── Variant 7: Fixed LSH params ──────────────────────────────────
    # Force 32 tables, 2 bands regardless of threshold.
    result = _run_variant_and_evaluate(
        "Fixed LSH params",
        mixed_fasta, ground_truth, threshold, threads, clustkit_mode,
        mock_patches=[
            ("clustkit.pipeline.auto_lsh_params", _make_fixed_lsh_params(32, 2)),
        ],
    )
    all_results.append(result)

    # ── Summary table ────────────────────────────────────────────────
    print()
    print("=" * 120)
    print("ABLATION SUMMARY")
    print("=" * 120)
    print()
    print(
        f"{'Variant':<25} {'Clust':>6} {'ARI':>8} {'NMI':>8} "
        f"{'P(pw)':>8} {'R(pw)':>8} {'F1(pw)':>8} {'Time':>10}"
    )
    print("-" * 90)

    baseline_time = None
    for res in all_results:
        name = res.get("variant", "?")
        if "error" in res and "ARI" not in res:
            rt = res.get("runtime_seconds", "?")
            print(
                f"{name:<25} {'FAIL':>6} {'':>8} {'':>8} "
                f"{'':>8} {'':>8} {'':>8} {rt:>9}s"
            )
            continue

        rt = res["runtime_seconds"]
        if baseline_time is None:
            baseline_time = rt

        speedup = ""
        if baseline_time and baseline_time > 0:
            ratio = rt / baseline_time
            speedup = f" ({ratio:.2f}x)"

        print(
            f"{name:<25} "
            f"{res.get('n_predicted_clusters', '?'):>6} "
            f"{res.get('ARI', 0):>8.4f} "
            f"{res.get('NMI', 0):>8.4f} "
            f"{res.get('pairwise_precision', 0):>8.4f} "
            f"{res.get('pairwise_recall', 0):>8.4f} "
            f"{res.get('pairwise_F1', 0):>8.4f} "
            f"{rt:>8.2f}s{speedup}"
        )

    print()

    # Delta table (difference from baseline)
    baseline = all_results[0] if all_results else None
    if baseline and "ARI" in baseline:
        print("Deltas from baseline (Full ClustKIT):")
        print(
            f"{'Variant':<25} {'dARI':>8} {'dNMI':>8} "
            f"{'dF1(pw)':>8} {'dTime':>10}"
        )
        print("-" * 70)
        for res in all_results[1:]:
            if "ARI" not in res:
                continue
            name = res.get("variant", "?")
            d_ari = res["ARI"] - baseline["ARI"]
            d_nmi = res["NMI"] - baseline["NMI"]
            d_f1 = res["pairwise_F1"] - baseline["pairwise_F1"]
            d_time = res["runtime_seconds"] - baseline["runtime_seconds"]
            print(
                f"{name:<25} "
                f"{d_ari:>+8.4f} "
                f"{d_nmi:>+8.4f} "
                f"{d_f1:>+8.4f} "
                f"{d_time:>+9.2f}s"
            )
        print()

    # Save results
    results_file = OUTPUT_DIR / f"ablation_results_{clustkit_mode}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ablation study benchmark for ClustKIT (C6)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Identity threshold (default: 0.4, where differences are largest).",
    )
    parser.add_argument(
        "--max-per-family",
        type=int,
        default=500,
        help="Max sequences per Pfam family (default: 500).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads (default: 4).",
    )
    parser.add_argument(
        "--clustkit-mode",
        type=str,
        default="balanced",
        choices=["balanced", "accurate", "fast"],
        help="ClustKIT clustering mode to benchmark.",
    )
    args = parser.parse_args()

    run_ablation(
        threshold=args.threshold,
        max_per_family=args.max_per_family,
        threads=args.threads,
        clustkit_mode=args.clustkit_mode,
    )
