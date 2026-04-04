"""Large-scale clustering benchmark: ClustKIT vs MMseqs2 vs DeepClust vs Linclust.

Tests on TrEMBL subsamples (100K -> 5M) at t=0.3 and t=0.5.
Measures runtime, peak memory (via /usr/bin/time -v), and cluster count.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCALING_DIR = Path(__file__).resolve().parent / "data" / "scaling"
OUT_DIR = Path(__file__).resolve().parent / "data" / "scaling_results_large"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MMSEQS_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/mmseqs/bin/mmseqs"
DIAMOND_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/diamond-linux64/diamond"

THREADS = 8


def parse_time_v(stderr_text):
    """Parse /usr/bin/time -v output for peak memory and wall clock."""
    peak_kb = None
    wall_seconds = None
    for line in stderr_text.split("\n"):
        line = line.strip()
        if "Maximum resident set size" in line:
            peak_kb = int(line.split(":")[-1].strip())
        if "Elapsed (wall clock)" in line:
            parts = line.split(":")[-1].strip()
            time_parts = parts.split(":")
            if len(time_parts) == 3:
                wall_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + float(time_parts[2])
            elif len(time_parts) == 2:
                wall_seconds = int(time_parts[0]) * 60 + float(time_parts[1])
    return peak_kb, wall_seconds


def run_with_memory(cmd, timeout=86400):
    """Run command with /usr/bin/time -v to capture peak memory."""
    full_cmd = ["/usr/bin/time", "-v"] + cmd
    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
    peak_kb, wall_seconds = parse_time_v(result.stderr)
    return {
        "returncode": result.returncode,
        "peak_memory_kb": peak_kb,
        "peak_memory_mb": round(peak_kb / 1024, 1) if peak_kb else None,
        "peak_memory_gb": round(peak_kb / 1024 / 1024, 2) if peak_kb else None,
        "wall_seconds": wall_seconds,
    }


def run_clustkit(fasta, threshold, threads, timeout=86400):
    """Run ClustKIT via subprocess to capture memory."""
    out_dir = OUT_DIR / "clustkit_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    script = f"""
import sys; sys.path.insert(0, '.')
from clustkit.pipeline import run_pipeline
import numba, os
numba.set_num_threads({threads})
os.environ['OMP_NUM_THREADS'] = '{threads}'
config = dict(
    input='{fasta}', output='{out_dir}', threshold={threshold},
    mode='protein', alignment='align', sketch_size=128, kmer_size=5,
    sensitivity='high', cluster_method='leiden', representative='longest',
    device='cpu', threads={threads}, format='tsv',
    use_c_ext=True, band_width=100,
)
run_pipeline(config)
"""
    tmp_script = OUT_DIR / "run_clustkit_tmp.py"
    with open(tmp_script, "w") as f:
        f.write(script)
    result = run_with_memory(["python", str(tmp_script)], timeout=timeout)
    cluster_file = out_dir / "clusters.tsv"
    n_clusters = 0
    if cluster_file.exists():
        with open(cluster_file) as f:
            next(f)
            n_clusters = len(set(line.strip().split("\t")[1] for line in f))
    result["n_clusters"] = n_clusters
    return result


def run_mmseqs(fasta, threshold, threads, timeout=86400):
    tmp_dir = tempfile.mkdtemp(prefix="mmseqs_scale_")
    prefix = str(OUT_DIR / "mmseqs_tmp")
    result = run_with_memory([
        MMSEQS_BIN, "easy-cluster", str(fasta), prefix, tmp_dir,
        "--min-seq-id", str(threshold), "--threads", str(threads),
        "-c", "0.8", "--cov-mode", "0",
    ], timeout=timeout)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    n_clusters = 0
    tsv = prefix + "_cluster.tsv"
    if os.path.exists(tsv):
        with open(tsv) as f:
            n_clusters = len(set(line.strip().split("\t")[0] for line in f))
    result["n_clusters"] = n_clusters
    return result


def run_linclust(fasta, threshold, threads, timeout=86400):
    tmp_dir = tempfile.mkdtemp(prefix="linclust_scale_")
    prefix = str(OUT_DIR / "linclust_tmp")
    result = run_with_memory([
        MMSEQS_BIN, "easy-linclust", str(fasta), prefix, tmp_dir,
        "--min-seq-id", str(threshold), "--threads", str(threads),
        "-c", "0.8", "--cov-mode", "0",
    ], timeout=timeout)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    n_clusters = 0
    tsv = prefix + "_cluster.tsv"
    if os.path.exists(tsv):
        with open(tsv) as f:
            n_clusters = len(set(line.strip().split("\t")[0] for line in f))
    result["n_clusters"] = n_clusters
    return result


def run_deepclust(fasta, threshold, threads, timeout=86400):
    out_file = str(OUT_DIR / "deepclust_tmp.tsv")
    approx_id = int(threshold * 100)
    result = run_with_memory([
        DIAMOND_BIN, "deepclust", "-d", str(fasta), "-o", out_file,
        "--approx-id", str(approx_id), "--member-cover", "80",
        "--threads", str(threads), "-M", "64G",
    ], timeout=timeout)
    n_clusters = 0
    if os.path.exists(out_file):
        with open(out_file) as f:
            n_clusters = len(set(line.strip().split("\t")[0] for line in f))
    result["n_clusters"] = n_clusters
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[100000, 500000, 1000000])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tools", nargs="+", default=["clustkit", "mmseqs2", "linclust", "deepclust"])
    parser.add_argument("--timeout", type=int, default=86400)
    args = parser.parse_args()

    tools = {
        "clustkit": run_clustkit,
        "mmseqs2": run_mmseqs,
        "linclust": run_linclust,
        "deepclust": run_deepclust,
    }

    print("=" * 100)
    print(f"LARGE-SCALE CLUSTERING BENCHMARK ({args.threads} threads)")
    print("=" * 100)

    all_results = {}

    for size in args.sizes:
        size_k = size // 1000
        fasta = SCALING_DIR / f"trembl_{size_k}k.fasta"
        if not fasta.exists():
            print(f"\n  SKIP {size_k}K -- file not found: {fasta}")
            continue

        for threshold in args.thresholds:
            key = f"{size_k}k_t{threshold}"
            print(f"\n{'='*80}")
            print(f"  {size_k}K sequences, threshold={threshold}")
            print(f"{'='*80}")

            scenario = {}
            for tool_name in args.tools:
                if tool_name not in tools:
                    continue
                print(f"  {tool_name}...", end=" ", flush=True)
                try:
                    result = tools[tool_name](str(fasta), threshold, args.threads, args.timeout)
                    scenario[tool_name] = result
                    mem = result.get("peak_memory_gb", "?")
                    wall = result.get("wall_seconds", "?")
                    nc = result.get("n_clusters", "?")
                    rc = result.get("returncode", "?")
                    status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                    print(f"{status} clusters={nc}, time={wall}s, memory={mem}GB", flush=True)
                except subprocess.TimeoutExpired:
                    print(f"TIMEOUT ({args.timeout}s)", flush=True)
                    scenario[tool_name] = {"error": "timeout"}
                except Exception as e:
                    print(f"ERROR: {e}", flush=True)
                    scenario[tool_name] = {"error": str(e)}

            all_results[key] = scenario

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Config':<20} {'Tool':<12} {'Clusters':>10} {'Time(s)':>10} {'Mem(GB)':>10}")
    print("-" * 65)
    for key in sorted(all_results.keys()):
        for tool, r in sorted(all_results[key].items()):
            if "error" in r:
                print(f"{key:<20} {tool:<12} {'FAIL':>10} {'':>10} {'':>10}")
            else:
                nc = r.get("n_clusters", "?")
                wall = r.get("wall_seconds", "?")
                if isinstance(wall, float):
                    wall = f"{wall:.1f}"
                mem = r.get("peak_memory_gb", "?")
                print(f"{key:<20} {tool:<12} {str(nc):>10} {str(wall):>10} {str(mem):>10}")

    results_file = OUT_DIR / "scaling_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {results_file}")
