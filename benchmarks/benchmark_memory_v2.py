"""Memory profiling: peak RSS for ClustKIT vs competitors.

Uses /usr/bin/time -v to capture peak memory for each tool
on the 22K Pfam dataset at t=0.3 and t=0.5, plus 133K Pfam at t=0.5.
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

MMSEQS_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/mmseqs/bin/mmseqs"
DIAMOND_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/diamond-linux64/diamond"
CDHIT_SIF = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/cd-hit.sif"
VSEARCH_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/VSEARCH/vsearch-2.30.5-linux-x86_64/bin/vsearch"

OUT_DIR = Path("benchmarks/data/memory_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

THREADS = 8
DATASETS = {
    "22K_pfam": "benchmarks/data/pfam_mixed.fasta",
    "133K_pfam": "benchmarks/data/pfam_full/pfam_benchmark_large.fasta",
}


def parse_time_v(stderr):
    """Parse /usr/bin/time -v output."""
    peak_kb = wall = None
    for line in stderr.split("\n"):
        line = line.strip()
        if "Maximum resident set size" in line:
            peak_kb = int(line.split(":")[-1].strip())
        if "Elapsed (wall clock)" in line:
            parts = line.split(":")[-1].strip().split(":")
            if len(parts) == 3:
                wall = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                wall = int(parts[0]) * 60 + float(parts[1])
    return peak_kb, wall


def run_timed(cmd, timeout=14400):
    """Run with /usr/bin/time -v."""
    full_cmd = ["/usr/bin/time", "-v"] + cmd
    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
    peak_kb, wall = parse_time_v(result.stderr)
    return {
        "peak_mb": round(peak_kb / 1024, 1) if peak_kb else None,
        "wall_seconds": round(wall, 1) if wall else None,
        "returncode": result.returncode,
    }


def run_clustkit(fasta, threshold, mode="default"):
    modes = {
        "fast": {"cluster_method": "greedy", "band_width": 50, "sensitivity": "medium"},
        "default": {"cluster_method": "leiden", "band_width": 100, "sensitivity": "high"},
        "sensitive": {"cluster_method": "leiden", "band_width": 200, "sensitivity": "high"},
    }
    cfg = modes[mode]
    script = f"""
import sys; sys.path.insert(0, '.')
import numba, os
numba.set_num_threads({THREADS})
os.environ['OMP_NUM_THREADS'] = '{THREADS}'
from clustkit.pipeline import run_pipeline
run_pipeline(dict(
    input='{fasta}', output='/tmp/mem_ck_{mode}',
    threshold={threshold}, mode='protein', alignment='align',
    sketch_size=128, kmer_size=5, sensitivity='{cfg["sensitivity"]}',
    cluster_method='{cfg["cluster_method"]}', representative='longest',
    device='cpu', threads={THREADS}, format='tsv',
    use_c_ext=True, band_width={cfg["band_width"]},
    block='off', cascade='off',
))
"""
    tmp = OUT_DIR / f"run_ck_{mode}.py"
    with open(tmp, "w") as f:
        f.write(script)
    return run_timed(["python", str(tmp)])


def run_mmseqs(fasta, threshold):
    tmp_dir = tempfile.mkdtemp(prefix="mem_mmseqs_")
    prefix = str(OUT_DIR / "mem_mmseqs")
    r = run_timed([
        MMSEQS_BIN, "easy-cluster", str(fasta), prefix, tmp_dir,
        "--min-seq-id", str(threshold), "--threads", str(THREADS),
        "-c", "0.8", "--cov-mode", "0",
    ])
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return r


def run_linclust(fasta, threshold):
    tmp_dir = tempfile.mkdtemp(prefix="mem_linclust_")
    prefix = str(OUT_DIR / "mem_linclust")
    r = run_timed([
        MMSEQS_BIN, "easy-linclust", str(fasta), prefix, tmp_dir,
        "--min-seq-id", str(threshold), "--threads", str(THREADS),
        "-c", "0.8", "--cov-mode", "0",
    ])
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return r


def run_deepclust(fasta, threshold):
    out_file = str(OUT_DIR / "mem_deepclust.tsv")
    return run_timed([
        DIAMOND_BIN, "deepclust", "-d", str(fasta), "-o", out_file,
        "--approx-id", str(int(threshold * 100)), "--member-cover", "80",
        "--threads", str(THREADS), "-M", "64G",
    ])


def run_cdhit(fasta, threshold):
    prefix = str(OUT_DIR / "mem_cdhit")
    n = 5 if threshold >= 0.7 else (4 if threshold >= 0.6 else (3 if threshold >= 0.5 else 2))
    return run_timed([
        "singularity", "exec", CDHIT_SIF, "cd-hit",
        "-i", str(fasta), "-o", prefix,
        "-c", str(threshold), "-T", str(THREADS), "-M", "0", "-d", "0", "-n", str(n),
    ])


def run_vsearch(fasta, threshold):
    uc = str(OUT_DIR / "mem_vsearch.uc")
    return run_timed([
        VSEARCH_BIN, "--cluster_fast", str(fasta),
        "--id", str(threshold), "--uc", uc, "--threads", str(THREADS),
    ])


if __name__ == "__main__":
    tools = [
        ("CK (fast)", lambda f, t: run_clustkit(f, t, "fast")),
        ("CK (default)", lambda f, t: run_clustkit(f, t, "default")),
        ("CK (sensitive)", lambda f, t: run_clustkit(f, t, "sensitive")),
        ("MMseqs2", run_mmseqs),
        ("DeepClust", run_deepclust),
        ("Linclust", run_linclust),
        ("CD-HIT", run_cdhit),
        ("VSEARCH", run_vsearch),
    ]

    configs = [
        ("22K_pfam", 0.3),
        ("22K_pfam", 0.5),
        ("133K_pfam", 0.5),
    ]

    print("=" * 80)
    print(f"MEMORY PROFILING ({THREADS} threads)")
    print("=" * 80)

    all_results = {}

    for dataset_name, threshold in configs:
        fasta = DATASETS[dataset_name]
        key = f"{dataset_name}_t{threshold}"
        print(f"\n--- {dataset_name}, t={threshold} ---")
        print(f"{'Tool':<20} {'Peak MB':>10} {'Time(s)':>10}")
        print("-" * 42)

        scenario = {}
        for tool_name, runner in tools:
            print(f"  {tool_name}...", end=" ", flush=True)
            try:
                r = runner(fasta, threshold)
                scenario[tool_name] = r
                mb = r.get("peak_mb", "?")
                wall = r.get("wall_seconds", "?")
                status = "OK" if r["returncode"] == 0 else f"FAIL(rc={r['returncode']})"
                print(f"{status} {mb} MB, {wall}s", flush=True)
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                scenario[tool_name] = {"error": str(e)}

        all_results[key] = scenario

    out_file = OUT_DIR / "memory_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_file}")
