"""Metagenomic-scale clustering benchmark.

Tests ClustKIT, MMseqs2 (easy-cluster), and Linclust on large metagenomic
datasets to characterize scaling behavior at millions of sequences.

Inspired by Linclust paper's 1.6B sequence clustering claim.
CP8 from the publication plan.
"""

import argparse
import json
import logging
import math
import os
import random
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Add project root to path
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.pipeline import run_pipeline
from clustkit.utils import logger as clustkit_logger

# ---------------------------------------------------------------------------
# Tool binary paths
# ---------------------------------------------------------------------------
MMSEQS_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/mmseqs/bin/mmseqs"
CDHIT_SIF = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/cd-hit.sif"

# CD-HIT is skipped above this many sequences (too slow at low thresholds)
CDHIT_MAX_SEQS = 1_000_000

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("bench_metagenomic")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    log.addHandler(_h)


# ===================================================================
# Subsampling
# ===================================================================

def load_sequence_ids(fasta_path: Path) -> list[str]:
    """Read all sequence IDs from a FASTA file (memory-light scan)."""
    ids = []
    with open(fasta_path) as fh:
        for line in fh:
            if line.startswith(">"):
                sid = line[1:].split()[0]
                ids.append(sid)
    return ids


def subsample_fasta(fasta_path: Path, selected_ids: set[str], out_path: Path):
    """Write a subset of sequences from *fasta_path* to *out_path*.

    Only sequences whose IDs are in *selected_ids* are written.
    """
    writing = False
    with open(fasta_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            if line.startswith(">"):
                sid = line[1:].split()[0]
                writing = sid in selected_ids
            if writing:
                fout.write(line)


def create_subsample(fasta_path: Path, n: int, seed: int, tmp_dir: Path) -> Path:
    """Create a deterministic subsample of *n* sequences and return its path."""
    all_ids = load_sequence_ids(fasta_path)
    total = len(all_ids)
    if n >= total:
        log.warning(
            f"Requested {n} sequences but input has only {total}; "
            f"using all {total} sequences."
        )
        return fasta_path

    rng = random.Random(seed)
    selected = set(rng.sample(all_ids, n))
    out_path = tmp_dir / f"subsample_{n}.fasta"
    subsample_fasta(fasta_path, selected, out_path)
    return out_path


# ===================================================================
# ClustKIT runner (in-process, with per-phase timing capture)
# ===================================================================

class PhaseTimingHandler(logging.Handler):
    """Capture ClustKIT per-phase timing from log messages."""

    def __init__(self):
        super().__init__()
        self.phase_timings: dict[str, float] = {}

    def emit(self, record):
        msg = record.getMessage()
        # ClustKIT logs lines like "Phase N: ... done (X.XXs)"
        if "done (" in msg and "Phase" in msg:
            try:
                label = msg.split("...")[0].strip()
                seconds_str = msg.split("(")[1].split("s)")[0]
                self.phase_timings[label] = float(seconds_str)
            except (IndexError, ValueError):
                pass


def run_clustkit(fasta_path: Path, threshold: float, threads: int) -> dict:
    """Run ClustKIT in-process. Returns dict with timing / memory / clusters."""
    out_dir = tempfile.mkdtemp(prefix="clustkit_meta_")
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
        "threads": threads,
        "format": "tsv",
    }

    # Attach phase-timing handler
    handler = PhaseTimingHandler()
    clustkit_logger.addHandler(handler)

    # Measure peak RSS before
    r_before = resource.getrusage(resource.RUSAGE_SELF)
    rss_before_kb = r_before.ru_maxrss

    t0 = time.perf_counter()
    try:
        run_pipeline(config)
    except Exception as exc:
        clustkit_logger.removeHandler(handler)
        shutil.rmtree(out_dir, ignore_errors=True)
        return {"tool": "ClustKIT", "error": str(exc)}

    wall_time = time.perf_counter() - t0

    r_after = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in KB on Linux
    peak_rss_mb = r_after.ru_maxrss / 1024.0

    # Count clusters from output
    n_clusters = 0
    tsv_path = Path(out_dir) / "clusters.tsv"
    if tsv_path.exists():
        cluster_ids = set()
        with open(tsv_path) as fh:
            next(fh)  # header
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    cluster_ids.add(parts[1])
        n_clusters = len(cluster_ids)

    clustkit_logger.removeHandler(handler)
    shutil.rmtree(out_dir, ignore_errors=True)

    return {
        "tool": "ClustKIT",
        "wall_time_s": round(wall_time, 3),
        "peak_rss_mb": round(peak_rss_mb, 1),
        "n_clusters": n_clusters,
        "phase_timings": handler.phase_timings,
    }


# ===================================================================
# External tool runners (wall-clock + /usr/bin/time -v for peak RSS)
# ===================================================================

def _parse_gnu_time_stderr(stderr: str) -> float | None:
    """Parse peak RSS (MB) from /usr/bin/time -v stderr output."""
    for line in stderr.splitlines():
        line = line.strip()
        if "Maximum resident set size" in line:
            try:
                # Value is in KB
                kb = int(line.split(":")[-1].strip())
                return round(kb / 1024.0, 1)
            except ValueError:
                pass
    return None


def _run_external(cmd: list[str], timeout: int = 43200) -> dict:
    """Run an external command wrapped with /usr/bin/time -v.

    Returns dict with wall_time_s, peak_rss_mb (from /usr/bin/time), returncode.
    Default timeout is 12 hours for large metagenomic runs.
    """
    time_cmd = ["/usr/bin/time", "-v"] + cmd
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            time_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout ({timeout}s)", "wall_time_s": timeout}
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    wall_time = time.perf_counter() - t0

    # /usr/bin/time writes to stderr
    peak_rss_mb = _parse_gnu_time_stderr(result.stderr)

    return {
        "wall_time_s": round(wall_time, 3),
        "peak_rss_mb": peak_rss_mb,
        "returncode": result.returncode,
        "stderr": result.stderr[-2000:] if result.stderr else "",
    }


def _count_mmseqs_clusters(tsv_path: str) -> int:
    """Count clusters from an MMseqs2 cluster TSV."""
    reps = set()
    try:
        with open(tsv_path) as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    reps.add(parts[0])
    except FileNotFoundError:
        return 0
    return len(reps)


def _count_cdhit_clusters(clstr_path: str) -> int:
    """Count clusters from a CD-HIT .clstr file."""
    count = 0
    try:
        with open(clstr_path) as fh:
            for line in fh:
                if line.startswith(">Cluster"):
                    count += 1
    except FileNotFoundError:
        return 0
    return count


def run_mmseqs_cluster(fasta_path: Path, threshold: float, threads: int) -> dict:
    """Run MMseqs2 easy-cluster."""
    out_prefix = tempfile.mktemp(prefix="mmseqs_meta_")
    tmp_dir = tempfile.mkdtemp(prefix="mmseqs_tmp_meta_")

    cmd = [
        MMSEQS_BIN, "easy-cluster",
        str(fasta_path), out_prefix, tmp_dir,
        "--min-seq-id", str(threshold),
        "--threads", str(threads),
        "-c", "0.8",
        "--cov-mode", "0",
    ]

    info = _run_external(cmd)
    n_clusters = _count_mmseqs_clusters(out_prefix + "_cluster.tsv")

    # Cleanup
    for suffix in ["_cluster.tsv", "_all_seqs.fasta", "_rep_seq.fasta"]:
        try:
            os.remove(out_prefix + suffix)
        except OSError:
            pass
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "tool": "MMseqs2",
        "wall_time_s": info.get("wall_time_s"),
        "peak_rss_mb": info.get("peak_rss_mb"),
        "n_clusters": n_clusters,
        "error": info.get("error"),
    }


def run_linclust(fasta_path: Path, threshold: float, threads: int) -> dict:
    """Run MMseqs2 easy-linclust."""
    out_prefix = tempfile.mktemp(prefix="linclust_meta_")
    tmp_dir = tempfile.mkdtemp(prefix="linclust_tmp_meta_")

    cmd = [
        MMSEQS_BIN, "easy-linclust",
        str(fasta_path), out_prefix, tmp_dir,
        "--min-seq-id", str(threshold),
        "--threads", str(threads),
        "-c", "0.8",
        "--cov-mode", "0",
    ]

    info = _run_external(cmd)
    n_clusters = _count_mmseqs_clusters(out_prefix + "_cluster.tsv")

    for suffix in ["_cluster.tsv", "_all_seqs.fasta", "_rep_seq.fasta"]:
        try:
            os.remove(out_prefix + suffix)
        except OSError:
            pass
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "tool": "Linclust",
        "wall_time_s": info.get("wall_time_s"),
        "peak_rss_mb": info.get("peak_rss_mb"),
        "n_clusters": n_clusters,
        "error": info.get("error"),
    }


def run_cdhit(fasta_path: Path, threshold: float, threads: int) -> dict:
    """Run CD-HIT via Singularity."""
    out_prefix = tempfile.mktemp(prefix="cdhit_meta_")

    # Word size selection (CD-HIT requirement)
    if threshold >= 0.7:
        word_size = "5"
    elif threshold >= 0.6:
        word_size = "4"
    elif threshold >= 0.5:
        word_size = "3"
    else:
        word_size = "2"

    cmd = [
        "singularity", "exec", CDHIT_SIF,
        "cd-hit",
        "-i", str(fasta_path),
        "-o", out_prefix,
        "-c", str(threshold),
        "-T", str(threads),
        "-M", "0",
        "-d", "0",
        "-n", word_size,
    ]

    info = _run_external(cmd, timeout=7200)  # 2-hour timeout for CD-HIT
    n_clusters = _count_cdhit_clusters(out_prefix + ".clstr")

    for suffix in ["", ".clstr"]:
        try:
            os.remove(out_prefix + suffix)
        except OSError:
            pass

    return {
        "tool": "CD-HIT",
        "wall_time_s": info.get("wall_time_s"),
        "peak_rss_mb": info.get("peak_rss_mb"),
        "n_clusters": n_clusters,
        "error": info.get("error"),
    }


# ===================================================================
# Scaling analysis
# ===================================================================

def _log_log_slope(sizes: list[int], times: list[float]) -> float | None:
    """Compute log-log slope (scaling exponent) via least-squares fit.

    Returns the exponent alpha where time ~ N^alpha.
    Returns None if fewer than 2 valid data points.
    """
    valid = [(s, t) for s, t in zip(sizes, times) if t is not None and t > 0]
    if len(valid) < 2:
        return None
    log_n = np.array([math.log10(s) for s, _ in valid])
    log_t = np.array([math.log10(t) for _, t in valid])
    # Least-squares fit: log_t = alpha * log_n + c
    A = np.vstack([log_n, np.ones(len(log_n))]).T
    result = np.linalg.lstsq(A, log_t, rcond=None)
    alpha = result[0][0]
    return round(float(alpha), 3)


def _extrapolate_runtime(
    sizes: list[int], times: list[float], target_n: int
) -> float | None:
    """Extrapolate runtime to *target_n* sequences using log-log fit.

    Returns estimated runtime in seconds, or None if insufficient data.
    """
    valid = [(s, t) for s, t in zip(sizes, times) if t is not None and t > 0]
    if len(valid) < 2:
        return None
    log_n = np.array([math.log10(s) for s, _ in valid])
    log_t = np.array([math.log10(t) for _, t in valid])
    A = np.vstack([log_n, np.ones(len(log_n))]).T
    result = np.linalg.lstsq(A, log_t, rcond=None)
    alpha, c = result[0]
    log_target = alpha * math.log10(target_n) + c
    return round(10 ** log_target, 1)


def _find_crossover(
    sizes_a: list[int],
    times_a: list[float],
    sizes_b: list[int],
    times_b: list[float],
    label_a: str,
    label_b: str,
) -> str:
    """Identify approximate crossover point where tool B becomes faster than tool A.

    Returns a human-readable description.
    """
    valid_a = {s: t for s, t in zip(sizes_a, times_a) if t is not None and t > 0}
    valid_b = {s: t for s, t in zip(sizes_b, times_b) if t is not None and t > 0}

    common_sizes = sorted(set(valid_a.keys()) & set(valid_b.keys()))
    if len(common_sizes) < 2:
        return "insufficient data for crossover estimation"

    # Check if B is always faster
    b_always_faster = all(valid_b[s] < valid_a[s] for s in common_sizes)
    if b_always_faster:
        return f"{label_b} is faster at all tested sizes"

    # Check if A is always faster
    a_always_faster = all(valid_a[s] < valid_b[s] for s in common_sizes)
    if a_always_faster:
        return f"{label_a} is faster at all tested sizes"

    # Find where the crossover happens
    for i in range(len(common_sizes) - 1):
        s1, s2 = common_sizes[i], common_sizes[i + 1]
        diff1 = valid_a[s1] - valid_b[s1]  # positive = A slower
        diff2 = valid_a[s2] - valid_b[s2]
        if diff1 * diff2 < 0:
            # Sign change: crossover between s1 and s2
            # Linear interpolation on log scale
            log_s1, log_s2 = math.log10(s1), math.log10(s2)
            frac = abs(diff1) / (abs(diff1) + abs(diff2))
            crossover_log = log_s1 + frac * (log_s2 - log_s1)
            crossover_n = int(10 ** crossover_log)
            if diff1 < 0:
                return (
                    f"{label_b} becomes faster than {label_a} "
                    f"around ~{crossover_n:,} sequences"
                )
            else:
                return (
                    f"{label_a} becomes faster than {label_b} "
                    f"around ~{crossover_n:,} sequences"
                )

    return "no clear crossover detected"


def summarize_scaling(all_results: list[dict], sizes: list[int]):
    """Print scaling analysis: exponents, extrapolations, crossover points.

    Args:
        all_results: List of result dicts from the benchmark.
        sizes: The sizes that were actually tested.
    """
    tools = ["ClustKIT", "MMseqs2", "Linclust", "CD-HIT"]
    extrapolation_targets = [10_000_000, 50_000_000, 100_000_000]

    # Build per-tool time and memory arrays
    tool_sizes: dict[str, list[int]] = {t: [] for t in tools}
    tool_times: dict[str, list[float]] = {t: [] for t in tools}
    tool_memory: dict[str, list[float]] = {t: [] for t in tools}

    for r in all_results:
        tool = r.get("tool")
        if tool not in tools:
            continue
        n = r.get("n_sequences")
        wt = r.get("wall_time_s")
        rss = r.get("peak_rss_mb")
        if n is not None:
            tool_sizes[tool].append(n)
            tool_times[tool].append(wt)
            tool_memory[tool].append(rss)

    print()
    print("=" * 100)
    print("SCALING ANALYSIS")
    print("=" * 100)

    # --- Scaling exponents ---
    print()
    print("Estimated scaling exponents (time ~ N^alpha):")
    print("-" * 60)
    print(f"  {'Tool':<12} {'alpha (time)':>14} {'alpha (memory)':>16}")
    print(f"  {'----':<12} {'------------':>14} {'---------------':>16}")
    for tool in tools:
        time_alpha = _log_log_slope(tool_sizes[tool], tool_times[tool])
        mem_alpha = _log_log_slope(
            tool_sizes[tool],
            [m if m is not None else 0 for m in tool_memory[tool]],
        )
        time_str = f"{time_alpha:.3f}" if time_alpha is not None else "N/A"
        mem_str = f"{mem_alpha:.3f}" if mem_alpha is not None else "N/A"
        print(f"  {tool:<12} {time_str:>14} {mem_str:>16}")

    # --- Extrapolations ---
    print()
    print("Estimated runtime extrapolations (seconds):")
    print("-" * 80)
    header = f"  {'Tool':<12}"
    for target in extrapolation_targets:
        header += f" {target / 1e6:.0f}M seqs".rjust(16)
    print(header)
    header_sep = f"  {'----':<12}"
    for _ in extrapolation_targets:
        header_sep += " " + "-" * 15
    print(header_sep)

    for tool in tools:
        row = f"  {tool:<12}"
        for target in extrapolation_targets:
            est = _extrapolate_runtime(tool_sizes[tool], tool_times[tool], target)
            if est is not None:
                if est > 86400:
                    row += f" {est / 3600:>12.1f} h"
                elif est > 3600:
                    row += f" {est / 3600:>12.2f} h"
                elif est > 60:
                    row += f" {est / 60:>11.1f} min"
                else:
                    row += f" {est:>13.1f} s"
            else:
                row += " " * 13 + "N/A"
        print(row)

    # --- Crossover analysis ---
    print()
    print("Crossover analysis:")
    print("-" * 80)
    # ClustKIT vs Linclust
    msg = _find_crossover(
        tool_sizes["ClustKIT"], tool_times["ClustKIT"],
        tool_sizes["Linclust"], tool_times["Linclust"],
        "ClustKIT", "Linclust",
    )
    print(f"  ClustKIT vs Linclust: {msg}")

    # ClustKIT vs MMseqs2
    msg = _find_crossover(
        tool_sizes["ClustKIT"], tool_times["ClustKIT"],
        tool_sizes["MMseqs2"], tool_times["MMseqs2"],
        "ClustKIT", "MMseqs2",
    )
    print(f"  ClustKIT vs MMseqs2:  {msg}")

    # MMseqs2 vs Linclust
    msg = _find_crossover(
        tool_sizes["MMseqs2"], tool_times["MMseqs2"],
        tool_sizes["Linclust"], tool_times["Linclust"],
        "MMseqs2", "Linclust",
    )
    print(f"  MMseqs2 vs Linclust:  {msg}")

    # --- Identify practicality limits ---
    print()
    print("Practicality assessment:")
    print("-" * 80)
    for tool in tools:
        est_10m = _extrapolate_runtime(tool_sizes[tool], tool_times[tool], 10_000_000)
        if est_10m is None:
            print(f"  {tool}: insufficient data to assess")
        elif est_10m > 86400:
            print(
                f"  {tool}: IMPRACTICAL at 10M sequences "
                f"(estimated {est_10m / 3600:.1f} hours)"
            )
        elif est_10m > 3600:
            print(
                f"  {tool}: SLOW at 10M sequences "
                f"(estimated {est_10m / 60:.0f} minutes)"
            )
        else:
            print(
                f"  {tool}: FEASIBLE at 10M sequences "
                f"(estimated {est_10m:.0f} seconds)"
            )


# ===================================================================
# Main benchmark loop
# ===================================================================

def run_benchmark(
    input_fasta: Path,
    sizes: list[int],
    threshold: float,
    threads: int,
    skip_clustkit_sizes: set[int] | None = None,
    seed: int = 42,
    output_json: Path | None = None,
):
    """Run the metagenomic-scale clustering benchmark.

    Args:
        input_fasta: Path to large metagenomic FASTA.
        sizes: List of subset sizes to benchmark.
        threshold: Sequence identity threshold for clustering.
        threads: Number of CPU threads.
        skip_clustkit_sizes: Set of sizes at which to skip ClustKIT.
        seed: Random seed for deterministic subsampling.
        output_json: Path to save results JSON.
    """
    if skip_clustkit_sizes is None:
        skip_clustkit_sizes = set()

    tmp_dir = Path(tempfile.mkdtemp(prefix="bench_metagenomic_"))
    all_results = []

    total_seqs = len(load_sequence_ids(input_fasta))
    log.info(f"Input FASTA: {input_fasta} ({total_seqs:,} sequences)")
    log.info(f"Sizes: {sizes}")
    log.info(f"Threshold: {threshold}")
    log.info(f"Threads: {threads}")
    if skip_clustkit_sizes:
        log.info(f"Skipping ClustKIT at sizes: {sorted(skip_clustkit_sizes)}")
    log.info("")

    print("=" * 140)
    print("METAGENOMIC-SCALE CLUSTERING BENCHMARK (CP8)")
    print("=" * 140)
    print(f"Input: {input_fasta} ({total_seqs:,} sequences)")
    print(f"Threshold: {threshold}, Threads: {threads}")
    print()
    print(
        f"{'Size':>12} {'Tool':<12} "
        f"{'Wall(s)':>12} {'RSS(MB)':>10} {'Clusters':>10} {'Status':<20}"
    )
    print("-" * 140)

    for n_seqs in sizes:
        # Create subsample once per size
        log.info(f"Subsampling {n_seqs:,} sequences from {total_seqs:,} ...")
        sub_fasta = create_subsample(input_fasta, n_seqs, seed, tmp_dir)
        actual_n = len(load_sequence_ids(sub_fasta))
        log.info(f"  Created subsample: {actual_n:,} sequences")

        # Determine which tools to run at this size
        tool_runners = []

        # ClustKIT
        if n_seqs not in skip_clustkit_sizes:
            tool_runners.append(
                ("ClustKIT", lambda fp=sub_fasta: run_clustkit(fp, threshold, threads))
            )
        else:
            log.info(f"  Skipping ClustKIT at n={actual_n:,} (--skip-clustkit)")

        # MMseqs2 easy-cluster
        tool_runners.append(
            ("MMseqs2", lambda fp=sub_fasta: run_mmseqs_cluster(fp, threshold, threads))
        )

        # Linclust
        tool_runners.append(
            ("Linclust", lambda fp=sub_fasta: run_linclust(fp, threshold, threads))
        )

        # CD-HIT: skip for sizes > 1M (too slow at low thresholds)
        if actual_n <= CDHIT_MAX_SEQS:
            tool_runners.append(
                ("CD-HIT", lambda fp=sub_fasta: run_cdhit(fp, threshold, threads))
            )
        else:
            log.info(
                f"  Skipping CD-HIT at n={actual_n:,} "
                f"(>{CDHIT_MAX_SEQS:,}, too slow at threshold={threshold})"
            )

        for tool_name, runner in tool_runners:
            log.info(f"  Running {tool_name} | n={actual_n:,} | t={threshold} ...")
            try:
                result = runner()
            except Exception as exc:
                result = {"tool": tool_name, "error": str(exc)}

            result["n_sequences"] = actual_n
            result["threshold"] = threshold
            all_results.append(result)

            status = result.get("error", "OK") or "OK"
            wt = result.get("wall_time_s", "?")
            rss = result.get("peak_rss_mb", "?")
            nc = result.get("n_clusters", "?")
            wt_str = f"{wt:>12.2f}" if isinstance(wt, (int, float)) else f"{wt:>12}"
            rss_str = f"{rss:>10.1f}" if isinstance(rss, (int, float)) else f"{rss:>10}"
            nc_str = f"{nc:>10}" if nc != "?" else f"{'?':>10}"

            print(
                f"{actual_n:>12,} {tool_name:<12} "
                f"{wt_str} {rss_str} {nc_str} {status:<20}"
            )

            # Print per-phase timings for ClustKIT
            if tool_name == "ClustKIT" and "phase_timings" in result:
                for phase, secs in result["phase_timings"].items():
                    print(f"{'':>12} {'':>12} {phase}: {secs:.2f}s")

        # Record skipped tools as explicit entries
        if n_seqs in skip_clustkit_sizes:
            all_results.append({
                "tool": "ClustKIT",
                "n_sequences": actual_n,
                "threshold": threshold,
                "skipped": True,
                "reason": "user skip (--skip-clustkit)",
            })
            print(
                f"{actual_n:>12,} {'ClustKIT':<12} "
                f"{'---':>12} {'---':>10} {'---':>10} {'SKIPPED':<20}"
            )

        if actual_n > CDHIT_MAX_SEQS:
            all_results.append({
                "tool": "CD-HIT",
                "n_sequences": actual_n,
                "threshold": threshold,
                "skipped": True,
                "reason": f"too slow (>{CDHIT_MAX_SEQS:,} seqs at threshold={threshold})",
            })
            print(
                f"{actual_n:>12,} {'CD-HIT':<12} "
                f"{'---':>12} {'---':>10} {'---':>10} {'SKIPPED (>1M)' :<20}"
            )

        print()

    # Cleanup temp subsamples
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # --- Summary tables ---
    _print_summary(all_results, sizes, threshold)

    # --- Scaling analysis ---
    actual_sizes = sorted(
        set(r["n_sequences"] for r in all_results if not r.get("skipped"))
    )
    summarize_scaling(all_results, actual_sizes)

    # --- Save results ---
    if output_json is None:
        output_json = (
            Path(__file__).resolve().parent / "data" / "metagenomic_results.json"
        )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    log.info(f"Results saved to {output_json}")
    print(f"\nResults saved to {output_json}")


def _print_summary(all_results: list[dict], sizes: list[int], threshold: float):
    """Print summary tables of wall-clock time and memory across sizes and tools."""
    tools = ["ClustKIT", "MMseqs2", "Linclust", "CD-HIT"]

    print()
    print("=" * 100)
    print(f"WALL-CLOCK TIME SUMMARY (seconds) | threshold={threshold}")
    print("=" * 100)

    header = f"{'Size':>12}"
    for t in tools:
        header += f" {t:>14}"
    print(header)
    print("-" * (12 + 15 * len(tools)))

    for size in sizes:
        row = f"{size:>12,}"
        for tool in tools:
            match = [
                r for r in all_results
                if r.get("tool") == tool
                and r.get("n_sequences") == size
                and not r.get("skipped")
            ]
            if match and match[0].get("wall_time_s") is not None:
                wt = match[0]["wall_time_s"]
                if wt > 3600:
                    row += f" {wt / 3600:>11.2f} h"
                elif wt > 60:
                    row += f" {wt / 60:>10.1f} min"
                else:
                    row += f" {wt:>12.2f} s"
            elif any(
                r.get("tool") == tool
                and r.get("n_sequences") == size
                and r.get("skipped")
                for r in all_results
            ):
                row += f" {'SKIPPED':>14}"
            else:
                row += f" {'FAIL':>14}"
        print(row)

    print()
    print("=" * 100)
    print(f"PEAK MEMORY SUMMARY (MB) | threshold={threshold}")
    print("=" * 100)

    header = f"{'Size':>12}"
    for t in tools:
        header += f" {t:>14}"
    print(header)
    print("-" * (12 + 15 * len(tools)))

    for size in sizes:
        row = f"{size:>12,}"
        for tool in tools:
            match = [
                r for r in all_results
                if r.get("tool") == tool
                and r.get("n_sequences") == size
                and not r.get("skipped")
            ]
            if match and match[0].get("peak_rss_mb") is not None:
                row += f" {match[0]['peak_rss_mb']:>12.1f} M"
            elif any(
                r.get("tool") == tool
                and r.get("n_sequences") == size
                and r.get("skipped")
                for r in all_results
            ):
                row += f" {'SKIPPED':>14}"
            else:
                row += f" {'?':>14}"
        print(row)

    print()
    print("=" * 100)
    print(f"CLUSTER COUNT SUMMARY | threshold={threshold}")
    print("=" * 100)

    header = f"{'Size':>12}"
    for t in tools:
        header += f" {t:>14}"
    print(header)
    print("-" * (12 + 15 * len(tools)))

    for size in sizes:
        row = f"{size:>12,}"
        for tool in tools:
            match = [
                r for r in all_results
                if r.get("tool") == tool
                and r.get("n_sequences") == size
                and not r.get("skipped")
            ]
            if match and match[0].get("n_clusters") is not None:
                row += f" {match[0]['n_clusters']:>14,}"
            elif any(
                r.get("tool") == tool
                and r.get("n_sequences") == size
                and r.get("skipped")
                for r in all_results
            ):
                row += f" {'SKIPPED':>14}"
            else:
                row += f" {'?':>14}"
        print(row)


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Metagenomic-scale clustering benchmark (CP8).\n\n"
            "Tests ClustKIT, MMseqs2 (easy-cluster), and Linclust on large\n"
            "metagenomic datasets to characterize scaling behavior at millions\n"
            "of sequences."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Input FASTA file (large metagenomic dataset).",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[100_000, 500_000, 1_000_000, 5_000_000, 10_000_000],
        help=(
            "Dataset sizes to test "
            "(default: 100000 500000 1000000 5000000 10000000)."
        ),
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.3,
        help="Sequence identity threshold (default: 0.3).",
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="Number of CPU threads for all tools (default: 4).",
    )
    parser.add_argument(
        "--skip-clustkit", type=int, nargs="*", default=None, metavar="SIZE",
        help=(
            "Sizes at which to skip ClustKIT (e.g., --skip-clustkit 5000000 10000000). "
            "Use without arguments to skip ClustKIT at all sizes."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic subsampling (default: 42).",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help=(
            "Output JSON file for results "
            "(default: benchmarks/data/metagenomic_results.json)."
        ),
    )

    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")

    # Handle --skip-clustkit: no arguments means skip all sizes
    if args.skip_clustkit is not None:
        if len(args.skip_clustkit) == 0:
            skip_clustkit_sizes = set(args.sizes)
        else:
            skip_clustkit_sizes = set(args.skip_clustkit)
    else:
        skip_clustkit_sizes = set()

    run_benchmark(
        input_fasta=args.input,
        sizes=sorted(args.sizes),
        threshold=args.threshold,
        threads=args.threads,
        skip_clustkit_sizes=skip_clustkit_sizes,
        seed=args.seed,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
