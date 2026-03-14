"""Large-scale clustering benchmark: runtime and memory scaling.

Tests ClustKIT, MMseqs2, Linclust, and CD-HIT on increasing dataset sizes.
Inspired by Linclust paper Fig 2.

CP1 from the publication plan.
"""

import argparse
import json
import logging
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.io import read_sequences
from clustkit.pipeline import run_pipeline
from clustkit.utils import logger as clustkit_logger

# ---------------------------------------------------------------------------
# Tool binary paths
# ---------------------------------------------------------------------------
MMSEQS_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/mmseqs/bin/mmseqs"
CDHIT_SIF = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/cd-hit.sif"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("bench_large_scale")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
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
    if n > len(all_ids):
        log.warning(
            f"Requested {n} sequences but input has only {len(all_ids)}; "
            f"using all {len(all_ids)} sequences."
        )
        n = len(all_ids)

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
    import numba

    out_dir = tempfile.mkdtemp(prefix="clustkit_bench_")
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


def _run_external(cmd: list[str], timeout: int = 14400) -> dict:
    """Run an external command wrapped with /usr/bin/time -v.

    Returns dict with wall_time_s, peak_rss_mb (from /usr/bin/time), returncode.
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


def run_mmseqs_cluster(fasta_path: Path, threshold: float, threads: int) -> dict:
    """Run MMseqs2 easy-cluster."""
    out_prefix = tempfile.mktemp(prefix="mmseqs_bench_")
    tmp_dir = tempfile.mkdtemp(prefix="mmseqs_tmp_")

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
    out_prefix = tempfile.mktemp(prefix="linclust_bench_")
    tmp_dir = tempfile.mkdtemp(prefix="linclust_tmp_")

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
    out_prefix = tempfile.mktemp(prefix="cdhit_bench_")

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

    info = _run_external(cmd)
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
# Main benchmark loop
# ===================================================================

def run_benchmark(
    input_fasta: Path,
    sizes: list[int],
    thresholds: list[float],
    threads: int,
    seed: int = 42,
    output_json: Path | None = None,
):
    """Run the large-scale scaling benchmark."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="bench_largescale_"))
    all_results = []

    total_seqs = len(load_sequence_ids(input_fasta))
    log.info(f"Input FASTA: {input_fasta} ({total_seqs} sequences)")
    log.info(f"Sizes: {sizes}")
    log.info(f"Thresholds: {thresholds}")
    log.info(f"Threads: {threads}")
    log.info("")

    print("=" * 130)
    print("LARGE-SCALE CLUSTERING BENCHMARK (CP1)")
    print("=" * 130)
    print(
        f"{'Size':>10} {'Thresh':>8} {'Tool':<12} "
        f"{'Wall(s)':>10} {'RSS(MB)':>10} {'Clusters':>10} {'Status':<20}"
    )
    print("-" * 130)

    for n_seqs in sizes:
        # Create subsample once per size
        log.info(f"Subsampling {n_seqs} sequences from {total_seqs} ...")
        sub_fasta = create_subsample(input_fasta, n_seqs, seed, tmp_dir)
        actual_n = len(load_sequence_ids(sub_fasta))
        log.info(f"  Created subsample: {actual_n} sequences")

        for threshold in thresholds:
            tools = [
                ("ClustKIT", lambda: run_clustkit(sub_fasta, threshold, threads)),
                ("MMseqs2", lambda: run_mmseqs_cluster(sub_fasta, threshold, threads)),
                ("Linclust", lambda: run_linclust(sub_fasta, threshold, threads)),
                ("CD-HIT", lambda: run_cdhit(sub_fasta, threshold, threads)),
            ]

            for tool_name, runner in tools:
                log.info(f"  Running {tool_name} | n={actual_n} | t={threshold} ...")
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
                wt_str = f"{wt:>10.2f}" if isinstance(wt, (int, float)) else f"{wt:>10}"
                rss_str = f"{rss:>10.1f}" if isinstance(rss, (int, float)) else f"{rss:>10}"
                nc_str = f"{nc:>10}" if nc != "?" else f"{'?':>10}"

                print(
                    f"{actual_n:>10} {threshold:>8.2f} {tool_name:<12} "
                    f"{wt_str} {rss_str} {nc_str} {status:<20}"
                )

                # Print per-phase timings for ClustKIT
                if tool_name == "ClustKIT" and "phase_timings" in result:
                    for phase, secs in result["phase_timings"].items():
                        print(f"{'':>10} {'':>8} {'':>12} {phase}: {secs:.2f}s")

        print()

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Save results
    if output_json is None:
        output_json = Path(__file__).resolve().parent / "data" / "large_scale_results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    log.info(f"Results saved to {output_json}")

    # Print summary table
    _print_summary(all_results, sizes, thresholds)


def _print_summary(all_results: list[dict], sizes: list[int], thresholds: list[float]):
    """Print a summary table of wall-clock time across sizes and tools."""
    print()
    print("=" * 130)
    print("WALL-CLOCK TIME SUMMARY (seconds)")
    print("=" * 130)

    tools = ["ClustKIT", "MMseqs2", "Linclust", "CD-HIT"]

    for threshold in thresholds:
        print(f"\nThreshold = {threshold}")
        header = f"{'Size':>10}"
        for t in tools:
            header += f" {t:>12}"
        print(header)
        print("-" * (10 + 13 * len(tools)))

        for size in sizes:
            row = f"{size:>10}"
            for tool in tools:
                match = [
                    r for r in all_results
                    if r.get("tool") == tool
                    and r.get("n_sequences") == size
                    and r.get("threshold") == threshold
                ]
                if match and match[0].get("wall_time_s") is not None:
                    row += f" {match[0]['wall_time_s']:>12.2f}"
                else:
                    row += f" {'FAIL':>12}"
            print(row)

    print()
    print("=" * 130)
    print("PEAK MEMORY SUMMARY (MB)")
    print("=" * 130)

    for threshold in thresholds:
        print(f"\nThreshold = {threshold}")
        header = f"{'Size':>10}"
        for t in tools:
            header += f" {t:>12}"
        print(header)
        print("-" * (10 + 13 * len(tools)))

        for size in sizes:
            row = f"{size:>10}"
            for tool in tools:
                match = [
                    r for r in all_results
                    if r.get("tool") == tool
                    and r.get("n_sequences") == size
                    and r.get("threshold") == threshold
                ]
                if match and match[0].get("peak_rss_mb") is not None:
                    row += f" {match[0]['peak_rss_mb']:>12.1f}"
                else:
                    row += f" {'?':>12}"
            print(row)


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Large-scale clustering benchmark: runtime and memory scaling (CP1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Input FASTA file (large SwissProt/UniProt).",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000],
        help="Dataset sizes to test (default: 10000 50000 100000 250000 500000 1000000).",
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.3, 0.5, 0.7, 0.9],
        help="Identity thresholds to test (default: 0.3 0.5 0.7 0.9).",
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="Number of CPU threads for all tools (default: 4).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic subsampling (default: 42).",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output JSON file for results (default: benchmarks/data/large_scale_results.json).",
    )

    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")

    run_benchmark(
        input_fasta=args.input,
        sizes=sorted(args.sizes),
        thresholds=args.thresholds,
        threads=args.threads,
        seed=args.seed,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
