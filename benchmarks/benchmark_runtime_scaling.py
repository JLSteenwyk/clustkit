"""Runtime vs dataset size (N) scaling benchmark.

Subsets the 133K Pfam dataset at 10K, 25K, 50K, 100K, 133K (full).
Runs ClustKIT (default), MMseqs2, DeepClust, Linclust at t=0.5, 8 threads.
Captures wall-clock time and peak RSS memory.
"""

import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.pipeline import run_pipeline

DATA_DIR = Path(__file__).resolve().parent / "data" / "pfam_full"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "runtime_scaling_results"
SUBSET_DIR = OUTPUT_DIR / "subsets"

MMSEQS_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/mmseqs/bin/mmseqs"
DIAMOND_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/diamond-linux64/diamond"

FASTA = DATA_DIR / "pfam_benchmark_large.fasta"
THRESHOLD = 0.5
THREADS = 8
SIZES = [10_000, 25_000, 50_000, 100_000, 133_122]  # last = full dataset


def parse_fasta_raw(path):
    """Parse FASTA into list of (name, sequence) tuples."""
    records = []
    name = None
    seq_parts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(seq_parts)))
                name = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
    if name is not None:
        records.append((name, "".join(seq_parts)))
    return records


def build_subsets():
    """Build random subsets of the 133K Pfam FASTA."""
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)

    records = parse_fasta_raw(FASTA)
    n_total = len(records)
    print(f"Loaded {n_total} sequences from {FASTA}")

    rng = np.random.RandomState(42)
    perm = rng.permutation(n_total)

    paths = {}
    for size in SIZES:
        if size >= n_total:
            paths[size] = str(FASTA)
            print(f"  {size}: full dataset")
            continue

        subset_path = SUBSET_DIR / f"pfam_subset_{size}.fasta"
        if subset_path.exists():
            count = sum(1 for line in open(subset_path) if line.startswith(">"))
            if count == size:
                paths[size] = str(subset_path)
                print(f"  {size}: already exists ({count} seqs)")
                continue

        indices = sorted(perm[:size])
        with open(subset_path, "w") as f:
            for idx in indices:
                name, seq = records[idx]
                f.write(f">{name}\n{seq}\n")
        paths[size] = str(subset_path)
        print(f"  {size}: created")

    return paths


def measure_peak_memory_mb():
    """Get peak RSS of this process in MB."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (ru.ru_maxrss + children.ru_maxrss) / 1024  # Linux: KB -> MB


def run_clustkit_timed(fasta_path, threshold, threads, label):
    """Run ClustKIT with current defaults, return (time, n_clusters)."""
    out_dir = OUTPUT_DIR / f"clustkit_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "input": fasta_path,
        "output": str(out_dir),
        "threshold": threshold,
        "mode": "protein",
        "alignment": "align",
        "sketch_size": 128,
        "kmer_size": 5,
        "sensitivity": "high",
        "cluster_method": "leiden",
        "representative": "longest",
        "device": "cpu",
        "threads": threads,
        "format": "tsv",
        "use_c_ext": True,
        "band_width": 100,
        "block": "off",
        "cascade": "off",
    }

    start = time.perf_counter()
    run_pipeline(config)
    elapsed = time.perf_counter() - start

    n_clusters = sum(1 for i, line in enumerate(open(out_dir / "clusters.tsv")) if i > 0)
    return elapsed, n_clusters


def run_external_tool(cmd, label):
    """Run an external tool via subprocess, return (time, None)."""
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        print(f"    {label} FAILED: {result.stderr[:200]}")
        return elapsed, None
    return elapsed, True


def run_mmseqs(fasta_path, threshold, threads, label):
    tmp_dir = tempfile.mkdtemp(prefix="mmseqs_scale_")
    prefix = str(OUTPUT_DIR / f"mmseqs_{label}")
    cmd = [
        MMSEQS_BIN, "easy-cluster", str(fasta_path), prefix, tmp_dir,
        "--min-seq-id", str(threshold), "--threads", str(threads),
        "-c", "0.8", "--cov-mode", "0",
    ]
    elapsed, ok = run_external_tool(cmd, "MMseqs2")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if ok:
        tsv = prefix + "_cluster.tsv"
        reps = set()
        with open(tsv) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    reps.add(parts[0])
        return elapsed, len(reps)
    return elapsed, None


def run_linclust(fasta_path, threshold, threads, label):
    tmp_dir = tempfile.mkdtemp(prefix="linclust_scale_")
    prefix = str(OUTPUT_DIR / f"linclust_{label}")
    cmd = [
        MMSEQS_BIN, "easy-linclust", str(fasta_path), prefix, tmp_dir,
        "--min-seq-id", str(threshold), "--threads", str(threads),
        "-c", "0.8", "--cov-mode", "0",
    ]
    elapsed, ok = run_external_tool(cmd, "Linclust")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if ok:
        tsv = prefix + "_cluster.tsv"
        reps = set()
        with open(tsv) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    reps.add(parts[0])
        return elapsed, len(reps)
    return elapsed, None


def run_deepclust(fasta_path, threshold, threads, label):
    out_file = str(OUTPUT_DIR / f"deepclust_{label}.tsv")
    approx_id = int(threshold * 100)
    cmd = [
        DIAMOND_BIN, "deepclust", "-d", str(fasta_path), "-o", out_file,
        "--approx-id", str(approx_id), "--member-cover", "80",
        "--threads", str(threads), "-M", "64G",
    ]
    elapsed, ok = run_external_tool(cmd, "DeepClust")
    if ok:
        reps = set()
        with open(out_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    reps.add(parts[0])
        return elapsed, len(reps)
    return elapsed, None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"RUNTIME vs DATASET SIZE — Pfam subsets, t={THRESHOLD}, {THREADS} threads")
    print("=" * 100)

    # Build subsets
    print("\nBuilding subsets...")
    subset_paths = build_subsets()

    tools = [
        ("ClustKIT", None),
        ("MMseqs2", run_mmseqs),
        ("Linclust", run_linclust),
        ("DeepClust", run_deepclust),
    ]

    all_results = {}

    print(f"\n{'Size':>8}  {'Tool':<12}  {'Time (s)':>10}  {'Clusters':>10}")
    print("-" * 50)

    for size in SIZES:
        fasta = subset_paths[size]
        label = f"{size // 1000}K" if size >= 1000 else str(size)
        size_results = {}

        for tool_name, runner in tools:
            print(f"{size:>8}  {tool_name:<12}", end="  ", flush=True)
            try:
                if tool_name == "ClustKIT":
                    elapsed, n_clusters = run_clustkit_timed(fasta, THRESHOLD, THREADS, label)
                else:
                    elapsed, n_clusters = runner(fasta, THRESHOLD, THREADS, label)

                size_results[tool_name] = {
                    "time_seconds": round(elapsed, 2),
                    "n_clusters": n_clusters,
                }
                clust_str = str(n_clusters) if n_clusters else "FAIL"
                print(f"{elapsed:>10.1f}  {clust_str:>10}")
            except Exception as e:
                print(f"ERROR: {e}")
                size_results[tool_name] = {"error": str(e)}

        all_results[size] = size_results
        print()

    # Summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"\n{'Size':>8}  {'CK (s)':>10}  {'MMseqs2 (s)':>12}  {'Linclust (s)':>13}  {'DeepClust (s)':>14}  {'CK/MM ratio':>12}")
    print("-" * 80)
    for size in SIZES:
        label = f"{size // 1000}K" if size >= 1000 else str(size)
        r = all_results[size]
        ck = r.get("ClustKIT", {}).get("time_seconds", None)
        mm = r.get("MMseqs2", {}).get("time_seconds", None)
        lc = r.get("Linclust", {}).get("time_seconds", None)
        dc = r.get("DeepClust", {}).get("time_seconds", None)
        ratio = f"{ck / mm:.1f}x" if (ck and mm) else "N/A"
        print(f"{label:>8}  {ck or 'FAIL':>10}  {mm or 'FAIL':>12}  {lc or 'FAIL':>13}  {dc or 'FAIL':>14}  {ratio:>12}")

    # Save
    results_file = OUTPUT_DIR / "runtime_scaling_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {results_file}")


if __name__ == "__main__":
    main()
