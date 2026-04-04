"""Phase 3 Profiling Script for ClustKIT.

Measures pre-filter effectiveness, adaptive band width distribution,
early termination rate, per-pair timing, and thread scaling.

Usage:
    python benchmarks/profile_phase3.py --input data.fasta --threshold 0.4
    python benchmarks/profile_phase3.py --input data.fasta --threshold 0.4 --max-threads 64
"""

import argparse
import sys
import time
from pathlib import Path

import numba
import numpy as np
from numba import njit, prange, int32, float32

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.io import read_sequences
from clustkit.sketch import compute_sketches
from clustkit.lsh import lsh_candidates
from clustkit.pairwise import _batch_align, _nw_identity
from clustkit.clustering_mode import resolve_clustering_mode
from clustkit.utils import auto_kmer_for_lsh, auto_lsh_params


DATA_DIR = Path(__file__).resolve().parent / "data"

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _separator(char="=", width=100):
    print(char * width)


def _header(title, char="=", width=100):
    print()
    _separator(char, width)
    print(f"  {title}")
    _separator(char, width)


def _fmt_count(n):
    """Format a large integer with commas."""
    return f"{n:,}"


def _fmt_pct(num, denom):
    """Format a percentage safely."""
    if denom == 0:
        return "N/A"
    return f"{100.0 * num / denom:.2f}%"


# ──────────────────────────────────────────────────────────────────────
# Section 1: Pre-filter effectiveness
# ──────────────────────────────────────────────────────────────────────

def profile_prefilters(candidate_pairs, lengths, threshold, band_width):
    """Analyze how many pairs each pre-filter removes.

    Reproduces the logic from _batch_align to count:
      - Pairs removed by the length-ratio pre-filter
      - Pairs removed by the length-diff pre-filter (after length-ratio)
      - Pairs that enter the DP alignment
    """
    _header("PRE-FILTER EFFECTIVENESS")

    m = candidate_pairs.shape[0]
    len_i = lengths[candidate_pairs[:, 0]]
    len_j = lengths[candidate_pairs[:, 1]]
    shorter = np.minimum(len_i, len_j)
    longer = np.maximum(len_i, len_j)
    len_diff = np.abs(len_i.astype(np.int32) - len_j.astype(np.int32))

    # B1: length-ratio pre-filter: shorter/longer < threshold
    ratio = np.where(longer > 0, shorter.astype(np.float32) / longer.astype(np.float32), 0.0)
    ratio_reject = ratio < threshold
    n_ratio_reject = int(np.sum(ratio_reject))

    # B2: length-diff pre-filter (applied to pairs surviving ratio filter)
    surviving_ratio = ~ratio_reject
    diff_reject = surviving_ratio & (len_diff > band_width)
    n_diff_reject = int(np.sum(diff_reject))

    # Pairs entering DP
    n_dp = m - n_ratio_reject - n_diff_reject

    print(f"\n  Total candidate pairs:          {_fmt_count(m)}")
    print(f"  Rejected by length-ratio filter: {_fmt_count(n_ratio_reject):>12}  ({_fmt_pct(n_ratio_reject, m)})")
    print(f"  Rejected by length-diff filter:  {_fmt_count(n_diff_reject):>12}  ({_fmt_pct(n_diff_reject, m)})")
    print(f"  Entering DP alignment:           {_fmt_count(n_dp):>12}  ({_fmt_pct(n_dp, m)})")
    print()

    # Additional statistics on the rejected pairs
    if n_ratio_reject > 0:
        rejected_ratios = ratio[ratio_reject]
        print(f"  Length-ratio rejected pairs:")
        print(f"    min ratio:  {rejected_ratios.min():.4f}")
        print(f"    mean ratio: {rejected_ratios.mean():.4f}")
        print(f"    max ratio:  {rejected_ratios.max():.4f}")
        print()

    if n_diff_reject > 0:
        rejected_diffs = len_diff[diff_reject]
        print(f"  Length-diff rejected pairs:")
        print(f"    min diff:   {int(rejected_diffs.min())}")
        print(f"    mean diff:  {rejected_diffs.mean():.1f}")
        print(f"    max diff:   {int(rejected_diffs.max())}")
        print(f"    band_width: {band_width}")
        print()

    return {
        "total": m,
        "ratio_reject": n_ratio_reject,
        "diff_reject": n_diff_reject,
        "entering_dp": n_dp,
    }


# ──────────────────────────────────────────────────────────────────────
# Section 2: Adaptive band width distribution
# ──────────────────────────────────────────────────────────────────────

def profile_band_widths(candidate_pairs, lengths, band_width):
    """Compute histogram of adaptive band widths for pairs entering DP."""
    _header("ADAPTIVE BAND WIDTH DISTRIBUTION")

    m = candidate_pairs.shape[0]
    len_i = lengths[candidate_pairs[:, 0]]
    len_j = lengths[candidate_pairs[:, 1]]
    shorter = np.minimum(len_i, len_j)
    longer = np.maximum(len_i, len_j)
    len_diff = np.abs(len_i.astype(np.int32) - len_j.astype(np.int32))

    # Compute adaptive_band = min(band_width, max(10, len_diff + 10))
    adaptive_band = np.minimum(band_width, np.maximum(10, len_diff + 10))

    # For short sequences (max(len_a, len_b) <= 50), full matrix: bw = max(len_a, len_b)
    max_lens = np.maximum(len_i, len_j)
    short_mask = max_lens <= 50
    adaptive_band[short_mask] = max_lens[short_mask]

    print(f"\n  Pairs analyzed: {_fmt_count(m)}")
    print(f"  Configured band_width (max): {band_width}")
    print()

    # Statistics
    print(f"  Adaptive band width statistics:")
    print(f"    min:     {int(adaptive_band.min()):>6}")
    print(f"    p25:     {int(np.percentile(adaptive_band, 25)):>6}")
    print(f"    median:  {int(np.median(adaptive_band)):>6}")
    print(f"    p75:     {int(np.percentile(adaptive_band, 75)):>6}")
    print(f"    p95:     {int(np.percentile(adaptive_band, 95)):>6}")
    print(f"    max:     {int(adaptive_band.max()):>6}")
    print(f"    mean:    {adaptive_band.mean():>9.1f}")
    print()

    # Histogram
    n_short = int(np.sum(short_mask))
    print(f"  Short-sequence pairs (full DP, max_len <= 50): {_fmt_count(n_short)} ({_fmt_pct(n_short, m)})")
    print()

    # Bin the adaptive band widths
    bw_values = adaptive_band.astype(np.int64)
    bin_edges = [0, 10, 15, 20, 30, 50, 75, 100, 150, 200, 500, int(bw_values.max()) + 1]
    # Remove bins beyond max value
    bin_edges = sorted(set(e for e in bin_edges if e <= int(bw_values.max()) + 1))
    if bin_edges[-1] <= int(bw_values.max()):
        bin_edges.append(int(bw_values.max()) + 1)

    counts, edges = np.histogram(bw_values, bins=bin_edges)
    max_bar = 50  # max bar length in characters
    max_count = max(counts) if max(counts) > 0 else 1

    print(f"  {'Band Width Range':<20} {'Count':>10} {'%':>8}  Bar")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 8}  {'-' * max_bar}")

    for i in range(len(counts)):
        lo = int(edges[i])
        hi = int(edges[i + 1]) - 1
        if lo == hi:
            label = f"  {lo}"
        else:
            label = f"  {lo}-{hi}"
        bar_len = int(counts[i] / max_count * max_bar)
        bar = "#" * bar_len
        print(f"  {label:<20} {counts[i]:>10} {_fmt_pct(counts[i], m):>8}  {bar}")

    print()
    return adaptive_band


# ──────────────────────────────────────────────────────────────────────
# Section 3: Early termination rate
# ──────────────────────────────────────────────────────────────────────

def profile_early_termination(candidate_pairs, encoded_sequences, lengths, threshold, band_width):
    """Measure early termination rate.

    Runs alignment on all pairs and counts:
      - Pairs that return identity 0.0 (early terminated or pre-filtered)
      - Pairs that return identity > 0.0 but < threshold
      - Pairs that return identity >= threshold
    """
    _header("EARLY TERMINATION RATE")

    m = candidate_pairs.shape[0]
    print(f"\n  Running alignment on {_fmt_count(m)} pairs (threshold={threshold})...")
    print(f"  (This measures the _batch_align kernel directly)\n")

    # Warm up numba
    warmup_n = min(10, m)
    if warmup_n > 0:
        _batch_align(
            candidate_pairs[:warmup_n],
            encoded_sequences,
            lengths,
            np.float32(threshold),
            int32(band_width),
        )

    t0 = time.perf_counter()
    sims, mask = _batch_align(
        candidate_pairs,
        encoded_sequences,
        lengths,
        np.float32(threshold),
        int32(band_width),
    )
    elapsed = time.perf_counter() - t0

    n_zero = int(np.sum(sims == 0.0))
    n_below = int(np.sum((sims > 0.0) & (sims < threshold)))
    n_above = int(np.sum(sims >= threshold))

    # Distinguish pre-filter zeros from early-termination zeros
    # Pre-filter: length-ratio or length-diff
    len_i = lengths[candidate_pairs[:, 0]]
    len_j = lengths[candidate_pairs[:, 1]]
    shorter = np.minimum(len_i, len_j)
    longer = np.maximum(len_i, len_j)
    len_diff = np.abs(len_i.astype(np.int32) - len_j.astype(np.int32))
    ratio = np.where(longer > 0, shorter.astype(np.float32) / longer.astype(np.float32), 0.0)

    prefilter_mask = (ratio < threshold) | (len_diff > band_width)
    n_prefiltered = int(np.sum(prefilter_mask))
    n_early_terminated = n_zero - n_prefiltered  # zeros that came from DP early termination

    print(f"  Alignment completed in {elapsed:.3f}s ({m / elapsed:,.0f} pairs/s)")
    print()
    print(f"  {'Category':<40} {'Count':>10} {'%':>8}")
    print(f"  {'-' * 40} {'-' * 10} {'-' * 8}")
    print(f"  {'Identity = 0.0 (pre-filtered)':<40} {_fmt_count(n_prefiltered):>10} {_fmt_pct(n_prefiltered, m):>8}")
    print(f"  {'Identity = 0.0 (DP early termination)':<40} {_fmt_count(n_early_terminated):>10} {_fmt_pct(n_early_terminated, m):>8}")
    print(f"  {'Identity 0.0 < id < threshold':<40} {_fmt_count(n_below):>10} {_fmt_pct(n_below, m):>8}")
    print(f"  {'Identity >= threshold (kept)':<40} {_fmt_count(n_above):>10} {_fmt_pct(n_above, m):>8}")
    print()

    total_zero = n_prefiltered + n_early_terminated
    dp_pairs = m - n_prefiltered
    print(f"  Summary:")
    print(f"    Total pairs returning 0.0:      {_fmt_count(total_zero)} ({_fmt_pct(total_zero, m)})")
    if dp_pairs > 0:
        print(f"    DP early termination rate:      {_fmt_pct(n_early_terminated, dp_pairs)} of DP-entered pairs")
    print(f"    Useful pairs (>= threshold):    {_fmt_count(n_above)} ({_fmt_pct(n_above, m)})")
    print()

    # Distribution of non-zero identities
    nonzero_sims = sims[sims > 0.0]
    if len(nonzero_sims) > 0:
        print(f"  Non-zero identity distribution ({_fmt_count(len(nonzero_sims))} pairs):")
        print(f"    min:     {nonzero_sims.min():.4f}")
        print(f"    p25:     {np.percentile(nonzero_sims, 25):.4f}")
        print(f"    median:  {np.median(nonzero_sims):.4f}")
        print(f"    p75:     {np.percentile(nonzero_sims, 75):.4f}")
        print(f"    max:     {nonzero_sims.max():.4f}")
        print()

    return {
        "elapsed": elapsed,
        "n_prefiltered": n_prefiltered,
        "n_early_terminated": n_early_terminated,
        "n_below_threshold": n_below,
        "n_above_threshold": n_above,
        "sims": sims,
    }


# ──────────────────────────────────────────────────────────────────────
# Section 4: Per-pair time distribution
# ──────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _time_single_pair(seq_a, len_a, seq_b, len_b, band_width, threshold):
    """Time a single NW alignment call (returns identity, no timing in njit).

    We call this from Python with perf_counter bracketing for timing.
    """
    return _nw_identity(seq_a, len_a, seq_b, len_b, band_width, threshold)


def profile_pair_timing(candidate_pairs, encoded_sequences, lengths, threshold, band_width,
                        sample_size=2000):
    """Measure per-pair timing as a function of work estimate.

    Work estimate: min(len_a, len_b) * adaptive_band
    Samples pairs and times each individually.
    """
    _header("PER-PAIR TIME DISTRIBUTION")

    m = candidate_pairs.shape[0]
    n_sample = min(sample_size, m)

    # Sample pairs deterministically
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(m, size=n_sample, replace=False)
    sample_pairs = candidate_pairs[sample_idx]

    print(f"\n  Sampling {_fmt_count(n_sample)} pairs from {_fmt_count(m)} total")
    print(f"  (Timing each pair individually with perf_counter)\n")

    # Warm up
    if n_sample > 0:
        i0, j0 = sample_pairs[0]
        li, lj = int(lengths[i0]), int(lengths[j0])
        if li <= lj:
            _nw_identity(encoded_sequences[i0], int32(li),
                         encoded_sequences[j0], int32(lj),
                         int32(band_width), np.float32(threshold))
        else:
            _nw_identity(encoded_sequences[j0], int32(lj),
                         encoded_sequences[i0], int32(li),
                         int32(band_width), np.float32(threshold))

    times = np.empty(n_sample, dtype=np.float64)
    work_estimates = np.empty(n_sample, dtype=np.float64)
    identities = np.empty(n_sample, dtype=np.float32)

    for s in range(n_sample):
        idx_i = sample_pairs[s, 0]
        idx_j = sample_pairs[s, 1]
        li = int(lengths[idx_i])
        lj = int(lengths[idx_j])

        shorter_len = min(li, lj)
        longer_len = max(li, lj)
        len_diff = abs(li - lj)
        adaptive_bw = min(band_width, max(10, len_diff + 10))
        if max(li, lj) <= 50:
            adaptive_bw = max(li, lj)

        work_estimates[s] = shorter_len * adaptive_bw

        # Swap so shorter is seq_a
        if li <= lj:
            t0 = time.perf_counter()
            identity = _nw_identity(
                encoded_sequences[idx_i], int32(li),
                encoded_sequences[idx_j], int32(lj),
                int32(adaptive_bw), np.float32(threshold),
            )
            times[s] = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            identity = _nw_identity(
                encoded_sequences[idx_j], int32(lj),
                encoded_sequences[idx_i], int32(li),
                int32(adaptive_bw), np.float32(threshold),
            )
            times[s] = time.perf_counter() - t0

        identities[s] = identity

    # Convert times to microseconds
    times_us = times * 1e6

    print(f"  Per-pair time statistics (microseconds):")
    print(f"    min:     {times_us.min():>10.1f} us")
    print(f"    p25:     {np.percentile(times_us, 25):>10.1f} us")
    print(f"    median:  {np.median(times_us):>10.1f} us")
    print(f"    p75:     {np.percentile(times_us, 75):>10.1f} us")
    print(f"    p95:     {np.percentile(times_us, 95):>10.1f} us")
    print(f"    max:     {times_us.max():>10.1f} us")
    print(f"    mean:    {times_us.mean():>10.1f} us")
    print()

    # Bin by work estimate and show time per bin
    nonzero_work = work_estimates > 0
    if np.any(nonzero_work):
        valid_work = work_estimates[nonzero_work]
        valid_times = times_us[nonzero_work]

        work_pcts = [0, 10, 25, 50, 75, 90, 100]
        work_edges = np.percentile(valid_work, work_pcts)
        # Ensure unique edges
        work_edges = np.unique(work_edges)

        if len(work_edges) >= 2:
            print(f"  Time vs work estimate (work = min_len * adaptive_band):")
            print(f"  {'Work Range':<25} {'Pairs':>7} {'Mean Time':>12} {'Median Time':>12} {'P95 Time':>12}")
            print(f"  {'-' * 25} {'-' * 7} {'-' * 12} {'-' * 12} {'-' * 12}")

            for b in range(len(work_edges) - 1):
                lo = work_edges[b]
                hi = work_edges[b + 1]
                if b == len(work_edges) - 2:
                    in_bin = (valid_work >= lo) & (valid_work <= hi)
                else:
                    in_bin = (valid_work >= lo) & (valid_work < hi)
                bin_times = valid_times[in_bin]
                n_bin = len(bin_times)
                if n_bin > 0:
                    label = f"  {lo:>10.0f} - {hi:>9.0f}"
                    print(f"  {label:<25} {n_bin:>7} {bin_times.mean():>10.1f}us {np.median(bin_times):>10.1f}us {np.percentile(bin_times, 95):>10.1f}us")

            print()

    # Early-terminated pairs timing
    zero_id = identities == 0.0
    nonzero_id = identities > 0.0
    if np.any(zero_id) and np.any(nonzero_id):
        print(f"  Timing by outcome:")
        print(f"    Early term / pre-filtered (id=0.0): mean={times_us[zero_id].mean():.1f}us, "
              f"median={np.median(times_us[zero_id]):.1f}us ({int(np.sum(zero_id))} pairs)")
        print(f"    Computed fully (id>0.0):             mean={times_us[nonzero_id].mean():.1f}us, "
              f"median={np.median(times_us[nonzero_id]):.1f}us ({int(np.sum(nonzero_id))} pairs)")
        print()

    return {
        "times_us": times_us,
        "work_estimates": work_estimates,
        "identities": identities,
    }


# ──────────────────────────────────────────────────────────────────────
# Section 5: Thread scaling curve
# ──────────────────────────────────────────────────────────────────────

def profile_thread_scaling(candidate_pairs, encoded_sequences, lengths, threshold, band_width,
                           max_threads=192):
    """Measure throughput at different thread counts.

    Uses numba.set_num_threads() to control parallelism.
    """
    _header("THREAD SCALING CURVE")

    m = candidate_pairs.shape[0]
    system_threads = numba.config.NUMBA_NUM_THREADS
    thread_counts = [1, 2, 4, 8, 16, 32, 64, 128, 192]
    thread_counts = [t for t in thread_counts if t <= min(max_threads, system_threads)]
    if not thread_counts:
        thread_counts = [1]

    # Add max_threads if it's not already in the list and within system limits
    if max_threads <= system_threads and max_threads not in thread_counts:
        thread_counts.append(max_threads)
        thread_counts.sort()

    print(f"\n  Total candidate pairs: {_fmt_count(m)}")
    print(f"  System max threads (NUMBA_NUM_THREADS): {system_threads}")
    print(f"  Testing thread counts: {thread_counts}")
    print()

    # Warm up with 1 thread
    numba.set_num_threads(1)
    warmup_n = min(100, m)
    if warmup_n > 0:
        _batch_align(
            candidate_pairs[:warmup_n],
            encoded_sequences,
            lengths,
            np.float32(threshold),
            int32(band_width),
        )

    results = []
    baseline_time = None

    print(f"  {'Threads':>8} {'Time (s)':>10} {'Pairs/s':>14} {'Speedup':>10} {'Efficiency':>12}")
    print(f"  {'-' * 8} {'-' * 10} {'-' * 14} {'-' * 10} {'-' * 12}")

    for n_threads in thread_counts:
        numba.set_num_threads(n_threads)

        # Run 2 trials and take the best to reduce variance
        best_time = float("inf")
        n_trials = 2
        for trial in range(n_trials):
            t0 = time.perf_counter()
            _batch_align(
                candidate_pairs,
                encoded_sequences,
                lengths,
                np.float32(threshold),
                int32(band_width),
            )
            elapsed = time.perf_counter() - t0
            best_time = min(best_time, elapsed)

        if baseline_time is None:
            baseline_time = best_time

        throughput = m / best_time
        speedup = baseline_time / best_time
        efficiency = speedup / n_threads * 100.0

        results.append({
            "threads": n_threads,
            "time": best_time,
            "throughput": throughput,
            "speedup": speedup,
            "efficiency": efficiency,
        })

        print(f"  {n_threads:>8} {best_time:>10.3f} {throughput:>14,.0f} {speedup:>9.2f}x {efficiency:>10.1f}%")

    print()

    # ASCII scaling chart
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

    # Restore a sensible default
    numba.set_num_threads(min(system_threads, max_threads))

    return results


# ──────────────────────────────────────────────────────────────────────
# Section 6: Memory analysis
# ──────────────────────────────────────────────────────────────────────

def profile_memory(encoded_sequences, lengths):
    """Analyze memory usage of the sequence matrix and suggest compact storage."""
    _header("MEMORY ANALYSIS")

    n_seqs, max_len = encoded_sequences.shape
    dtype = encoded_sequences.dtype
    bytes_per_elem = dtype.itemsize
    total_bytes = encoded_sequences.nbytes
    total_mb = total_bytes / (1024 * 1024)

    print(f"\n  Encoded sequence matrix:")
    print(f"    Shape:       {n_seqs} x {max_len}")
    print(f"    Dtype:       {dtype}")
    print(f"    Total size:  {total_mb:.2f} MB ({_fmt_count(total_bytes)} bytes)")
    print()

    # Actual data vs padding
    actual_residues = int(np.sum(lengths))
    padded_cells = n_seqs * max_len
    actual_bytes = actual_residues * bytes_per_elem
    padding_bytes = total_bytes - actual_bytes
    padding_pct = 100.0 * padding_bytes / total_bytes if total_bytes > 0 else 0.0

    print(f"  Utilization:")
    print(f"    Actual residues:   {_fmt_count(actual_residues)}")
    print(f"    Padded cells:      {_fmt_count(padded_cells)}")
    print(f"    Padding waste:     {padding_pct:.1f}% ({padding_bytes / (1024 * 1024):.2f} MB)")
    print()

    # Length statistics
    print(f"  Sequence length distribution:")
    print(f"    min:     {int(lengths.min()):>6}")
    print(f"    p25:     {int(np.percentile(lengths, 25)):>6}")
    print(f"    median:  {int(np.median(lengths)):>6}")
    print(f"    p75:     {int(np.percentile(lengths, 75)):>6}")
    print(f"    p95:     {int(np.percentile(lengths, 95)):>6}")
    print(f"    max:     {int(lengths.max()):>6}")
    print(f"    mean:    {lengths.mean():>9.1f}")
    print(f"    std:     {lengths.std():>9.1f}")
    print()

    # Compact storage suggestions
    print(f"  Compact storage options:")
    print()

    # Option 1: Concatenated array with offsets
    concat_bytes = actual_residues * bytes_per_elem
    offsets_bytes = (n_seqs + 1) * 4  # int32 offsets
    concat_total = concat_bytes + offsets_bytes
    concat_mb = concat_total / (1024 * 1024)
    savings_1 = total_bytes - concat_total
    print(f"    1. Concatenated array + offsets (CSR-style):")
    print(f"       Data:      {concat_bytes / (1024 * 1024):.2f} MB (concatenated residues)")
    print(f"       Offsets:   {offsets_bytes / (1024 * 1024):.4f} MB ({n_seqs + 1} int32)")
    print(f"       Total:     {concat_mb:.2f} MB")
    print(f"       Savings:   {savings_1 / (1024 * 1024):.2f} MB ({100.0 * savings_1 / total_bytes:.1f}%)")
    print()

    # Option 2: Bucketed by length (group sequences with similar lengths)
    bucket_size = 50  # group into buckets of 50 residue-length range
    length_vals = lengths.astype(np.int64)
    bucket_ids = length_vals // bucket_size
    unique_buckets = np.unique(bucket_ids)
    bucket_total = 0
    for b in unique_buckets:
        b_mask = bucket_ids == b
        b_count = int(np.sum(b_mask))
        b_max_len = int(length_vals[b_mask].max())
        bucket_total += b_count * b_max_len * bytes_per_elem
    bucket_total_mb = bucket_total / (1024 * 1024)
    savings_2 = total_bytes - bucket_total
    print(f"    2. Bucketed by length (bucket size = {bucket_size}):")
    print(f"       {len(unique_buckets)} buckets, each padded to local max length")
    print(f"       Total:     {bucket_total_mb:.2f} MB")
    if savings_2 > 0:
        print(f"       Savings:   {savings_2 / (1024 * 1024):.2f} MB ({100.0 * savings_2 / total_bytes:.1f}%)")
    else:
        print(f"       Savings:   None (overhead exceeds savings for this dataset)")
    print()

    # Option 3: 4-bit packing for protein (20 amino acids fit in 5 bits, 2 per byte)
    if dtype == np.uint8:
        packed_bytes = (actual_residues + 1) // 2  # 2 residues per byte
        packed_total = packed_bytes + offsets_bytes
        packed_mb = packed_total / (1024 * 1024)
        savings_3 = total_bytes - packed_total
        print(f"    3. 4-bit packed + concatenated (2 residues per byte):")
        print(f"       Total:     {packed_mb:.2f} MB")
        print(f"       Savings:   {savings_3 / (1024 * 1024):.2f} MB ({100.0 * savings_3 / total_bytes:.1f}%)")
        print(f"       Note: Requires unpack in alignment kernel; may reduce SIMD throughput")
        print()

    return {
        "total_mb": total_mb,
        "padding_pct": padding_pct,
        "concat_mb": concat_mb,
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 profiling for ClustKIT: pre-filters, band widths, "
                    "early termination, per-pair timing, thread scaling, and memory.",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(DATA_DIR / "pfam_mixed.fasta"),
        help="Input FASTA file (default: benchmarks/data/pfam_mixed.fasta)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.4,
        help="Identity threshold (default: 0.4)",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=192,
        help="Maximum thread count for scaling test (default: 192)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="protein",
        choices=["protein", "nucleotide"],
        help="Sequence type (default: protein)",
    )
    parser.add_argument(
        "--clustkit-mode",
        type=str,
        default="balanced",
        choices=["balanced", "accurate", "fast"],
        help="ClustKIT clustering mode to profile.",
    )
    parser.add_argument(
        "--sensitivity",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help="LSH sensitivity override (default: from --clustkit-mode)",
    )
    parser.add_argument(
        "--sketch-size",
        type=int,
        default=None,
        help="Sketch size override for MinHash (default: from --clustkit-mode)",
    )
    parser.add_argument(
        "--kmer-size", "-k",
        type=int,
        default=5,
        help="K-mer size (default: 5 for protein)",
    )
    parser.add_argument(
        "--timing-sample",
        type=int,
        default=2000,
        help="Number of pairs to sample for per-pair timing (default: 2000)",
    )
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="Skip per-pair timing (Section 4) for faster profiling",
    )
    parser.add_argument(
        "--skip-scaling",
        action="store_true",
        help="Skip thread scaling (Section 5) for faster profiling",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print(f"Hint: Run the Pfam download/mix script first, or specify --input")
        sys.exit(1)

    resolved_sketch_size, resolved_sensitivity = resolve_clustering_mode(
        args.clustkit_mode, args.threshold, args.sketch_size, args.sensitivity
    )
    args.sketch_size = resolved_sketch_size
    args.sensitivity = resolved_sensitivity

    _separator("=")
    print(f"  ClustKIT Phase 3 Profiler")
    _separator("=")
    print(f"  Input:       {input_path}")
    print(f"  Threshold:   {args.threshold}")
    print(f"  Mode:        {args.mode}")
    print(f"  CK mode:     {args.clustkit_mode}")
    print(f"  Sensitivity: {args.sensitivity}")
    print(f"  Sketch size: {args.sketch_size}")
    print(f"  Max threads: {args.max_threads}")
    print()

    # ── Phase 0: Read sequences ──────────────────────────────────────
    _header("PHASE 0: Reading Sequences", char="-")
    t0 = time.perf_counter()
    dataset = read_sequences(input_path, args.mode)
    t_read = time.perf_counter() - t0
    n = dataset.num_sequences
    print(f"\n  Loaded {_fmt_count(n)} sequences (max length {dataset.max_length}) in {t_read:.2f}s")
    print()

    if n < 2:
        print("  Error: Need at least 2 sequences for pairwise profiling.")
        sys.exit(1)

    # ── Phase 1: Sketch ──────────────────────────────────────────────
    _header("PHASE 1: Computing Sketches", char="-")
    k_lsh = auto_kmer_for_lsh(args.threshold, args.mode, args.kmer_size)
    print(f"\n  k-mer size for LSH: {k_lsh} (user k={args.kmer_size})")

    t0 = time.perf_counter()
    sketches = compute_sketches(
        dataset.encoded_sequences,
        dataset.lengths,
        k_lsh,
        args.sketch_size,
        args.mode,
    )
    t_sketch = time.perf_counter() - t0
    print(f"  Computed {_fmt_count(n)} sketches in {t_sketch:.2f}s")
    print()

    # ── Phase 2: LSH Candidates ──────────────────────────────────────
    _header("PHASE 2: LSH Candidate Generation", char="-")
    lsh_params = auto_lsh_params(args.threshold, args.sensitivity, k=k_lsh)
    print(f"\n  LSH params: {lsh_params['num_tables']} tables, {lsh_params['num_bands']} bands/table")

    t0 = time.perf_counter()
    candidate_pairs = lsh_candidates(
        sketches,
        num_tables=lsh_params["num_tables"],
        num_bands=lsh_params["num_bands"],
    )
    t_lsh = time.perf_counter() - t0
    print(f"  Found {_fmt_count(len(candidate_pairs))} candidate pairs in {t_lsh:.2f}s")
    print()

    if len(candidate_pairs) == 0:
        print("  No candidate pairs found. Cannot profile Phase 3.")
        sys.exit(1)

    # ── Compute band_width (same logic as pipeline.py) ───────────────
    p95_len = int(np.percentile(dataset.lengths, 95))
    band_width = max(20, int(p95_len * 0.3))
    print(f"  Band width: {band_width} (from p95 length = {p95_len})")
    print()

    # ── Section 1: Pre-filter effectiveness ──────────────────────────
    prefilter_stats = profile_prefilters(
        candidate_pairs, dataset.lengths, args.threshold, band_width
    )

    # ── Section 2: Adaptive band width distribution ──────────────────
    adaptive_bands = profile_band_widths(
        candidate_pairs, dataset.lengths, band_width
    )

    # ── Section 3: Early termination rate ────────────────────────────
    # Use max threads for the full alignment run
    numba.set_num_threads(min(numba.config.NUMBA_NUM_THREADS, args.max_threads))
    early_term_stats = profile_early_termination(
        candidate_pairs, dataset.encoded_sequences, dataset.lengths,
        args.threshold, band_width,
    )

    # ── Section 4: Per-pair timing ───────────────────────────────────
    if not args.skip_timing:
        # Single-threaded for accurate per-pair measurement
        numba.set_num_threads(1)
        timing_stats = profile_pair_timing(
            candidate_pairs, dataset.encoded_sequences, dataset.lengths,
            args.threshold, band_width,
            sample_size=args.timing_sample,
        )

    # ── Section 5: Thread scaling ────────────────────────────────────
    if not args.skip_scaling:
        scaling_results = profile_thread_scaling(
            candidate_pairs, dataset.encoded_sequences, dataset.lengths,
            args.threshold, band_width,
            max_threads=args.max_threads,
        )

    # ── Section 6: Memory analysis ───────────────────────────────────
    mem_stats = profile_memory(dataset.encoded_sequences, dataset.lengths)

    # ── Final summary ────────────────────────────────────────────────
    _header("SUMMARY")
    m = len(candidate_pairs)
    print(f"\n  Dataset:     {_fmt_count(n)} sequences, max length {dataset.max_length}")
    print(f"  Candidates:  {_fmt_count(m)} pairs")
    print(f"  Threshold:   {args.threshold}")
    print(f"  Band width:  {band_width}")
    print()
    print(f"  Pre-filter rejection:    {_fmt_pct(prefilter_stats['ratio_reject'] + prefilter_stats['diff_reject'], m)} of pairs never enter DP")
    print(f"  DP early termination:    {_fmt_count(early_term_stats['n_early_terminated'])} pairs")
    print(f"  Useful pairs:            {_fmt_count(early_term_stats['n_above_threshold'])} ({_fmt_pct(early_term_stats['n_above_threshold'], m)})")
    print(f"  Alignment throughput:    {m / early_term_stats['elapsed']:,.0f} pairs/s (all threads)")
    print(f"  Matrix memory:           {mem_stats['total_mb']:.2f} MB ({mem_stats['padding_pct']:.1f}% padding)")
    print(f"  Compact estimate:        {mem_stats['concat_mb']:.2f} MB (concatenated)")
    print()
    _separator("=")
    print("  Profiling complete.")
    _separator("=")
    print()


if __name__ == "__main__":
    main()
