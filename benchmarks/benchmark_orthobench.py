"""Open_Orthobench benchmark: ClustKIT vs MMseqs2 vs DeepClust.

251K sequences from 12 species, evaluated against 70 curated orthogroups.
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

OB_DIR = Path(__file__).resolve().parent / "data" / "open_orthobench" / "BENCHMARKS"
OUT_DIR = Path(__file__).resolve().parent / "data" / "orthobench_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MMSEQS_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/mmseqs/bin/mmseqs"
DIAMOND_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/diamond-linux64/diamond"
THREADS = 8


def concatenate_proteomes():
    """Concatenate all input proteomes into one FASTA."""
    combined = OUT_DIR / "all_proteomes.fasta"
    if combined.exists():
        n = sum(1 for line in open(combined) if line.startswith(">"))
        print(f"  Using existing {combined.name} ({n} sequences)")
        return combined

    input_dir = OB_DIR / "Input"
    fastas = sorted(input_dir.glob("*.fa"))
    print(f"  Concatenating {len(fastas)} proteomes...", end=" ", flush=True)
    n = 0
    with open(combined, "w") as out:
        for fa in fastas:
            with open(fa) as f:
                for line in f:
                    if line.startswith(">"):
                        # Use just the gene ID (first word after >)
                        gene_id = line.strip().split()[0][1:]
                        out.write(f">{gene_id}\n")
                        n += 1
                    else:
                        out.write(line)
    print(f"{n} sequences")
    return combined


def clusters_to_orthobench_format(clusters, output_file):
    """Convert cluster dict {gene_id: cluster_id} to OrthoBench format.

    OrthoBench expects: one orthogroup per line, space-separated gene IDs.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for gene_id, cluster_id in clusters.items():
        # Strip sp|ACC|NAME format if present
        if "|" in gene_id:
            gene_id = gene_id.split("|")[1] if gene_id.startswith("sp|") else gene_id
        groups[cluster_id].append(gene_id)

    with open(output_file, "w") as f:
        for cluster_id in sorted(groups.keys()):
            members = sorted(groups[cluster_id])
            if len(members) >= 2:  # skip singletons
                f.write(" ".join(members) + "\n")

    return len(groups)


def run_benchmark_py(orthogroups_file):
    """Run the Open_Orthobench benchmark.py script."""
    benchmark_script = OB_DIR / "benchmark.py"
    result = subprocess.run(
        [sys.executable, str(benchmark_script), str(orthogroups_file)],
        capture_output=True, text=True, cwd=str(OB_DIR),
    )
    return result.stdout, result.stderr


def parse_mmseqs_clusters(tsv_path):
    clusters = {}
    rep_to_id = {}
    next_id = 0
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            if rep not in rep_to_id:
                rep_to_id[rep] = next_id
                next_id += 1
            clusters[member] = rep_to_id[rep]
    return clusters


CLUSTKIT_MODES = {
    "fast": {"cluster_method": "greedy", "band_width": 50, "sensitivity": "medium"},
    "default": {"cluster_method": "leiden", "band_width": 100, "sensitivity": "high"},
    "sensitive": {"cluster_method": "leiden", "band_width": 200, "sensitivity": "high"},
}


def run_clustkit_mode(fasta, threshold, mode_name="default"):
    """Run ClustKIT clustering in a specific mode."""
    from clustkit.pipeline import run_pipeline
    import numba
    numba.set_num_threads(THREADS)
    os.environ["OMP_NUM_THREADS"] = str(THREADS)

    mode_cfg = CLUSTKIT_MODES[mode_name]
    out_dir = OUT_DIR / f"clustkit_{mode_name}_t{threshold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "input": str(fasta),
        "output": str(out_dir),
        "threshold": threshold,
        "mode": "protein",
        "alignment": "align",
        "sketch_size": 128,
        "kmer_size": 5,
        "sensitivity": mode_cfg["sensitivity"],
        "cluster_method": mode_cfg["cluster_method"],
        "representative": "longest",
        "device": "cpu",
        "threads": THREADS,
        "format": "tsv",
        "use_c_ext": True,
        "band_width": mode_cfg["band_width"],
        "block": "off",
        "cascade": "off",
    }
    start = time.perf_counter()
    run_pipeline(config)
    elapsed = time.perf_counter() - start

    clusters = {}
    with open(out_dir / "clusters.tsv") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            clusters[parts[0]] = int(parts[1])
    return clusters, elapsed


def run_mmseqs(fasta, threshold):
    """Run MMseqs2 clustering."""
    tmp_dir = tempfile.mkdtemp(prefix="mmseqs_ob_")
    prefix = str(OUT_DIR / f"mmseqs_t{threshold}")
    cmd = [
        MMSEQS_BIN, "easy-cluster", str(fasta), prefix, tmp_dir,
        "--min-seq-id", str(threshold), "--threads", str(THREADS),
        "-c", "0.8", "--cov-mode", "0",
    ]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    elapsed = time.perf_counter() - start
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if result.returncode != 0:
        return None, elapsed
    return parse_mmseqs_clusters(prefix + "_cluster.tsv"), elapsed


def run_deepclust(fasta, threshold):
    """Run DIAMOND DeepClust."""
    out_file = str(OUT_DIR / f"deepclust_t{threshold}.tsv")
    approx_id = int(threshold * 100)
    cmd = [
        DIAMOND_BIN, "deepclust", "-d", str(fasta), "-o", out_file,
        "--approx-id", str(approx_id), "--member-cover", "80",
        "--threads", str(THREADS), "-M", "64G",
    ]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        return None, elapsed
    return parse_mmseqs_clusters(out_file), elapsed


def run_linclust(fasta, threshold):
    """Run MMseqs2 Linclust."""
    tmp_dir = tempfile.mkdtemp(prefix="linclust_ob_")
    prefix = str(OUT_DIR / f"linclust_t{threshold}")
    cmd = [
        MMSEQS_BIN, "easy-linclust", str(fasta), prefix, tmp_dir,
        "--min-seq-id", str(threshold), "--threads", str(THREADS),
        "-c", "0.8", "--cov-mode", "0",
    ]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    elapsed = time.perf_counter() - start
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if result.returncode != 0:
        return None, elapsed
    return parse_mmseqs_clusters(prefix + "_cluster.tsv"), elapsed


CDHIT_SIF = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/cd-hit.sif"
VSEARCH_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/VSEARCH/vsearch-2.30.5-linux-x86_64/bin/vsearch"


def run_cdhit(fasta, threshold):
    """Run CD-HIT clustering."""
    prefix = str(OUT_DIR / f"cdhit_t{threshold}")
    n = 5 if threshold >= 0.7 else (4 if threshold >= 0.6 else (3 if threshold >= 0.5 else 2))
    cmd = [
        "singularity", "exec", CDHIT_SIF, "cd-hit",
        "-i", str(fasta), "-o", prefix,
        "-c", str(threshold), "-T", str(THREADS), "-M", "0", "-d", "0", "-n", str(n),
    ]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        return None, elapsed
    clusters = {}
    current_cluster = -1
    with open(prefix + ".clstr") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                current_cluster = int(line.split()[1])
            elif line:
                seq_id = line.split(">")[1].split("...")[0]
                clusters[seq_id] = current_cluster
    return clusters, elapsed


def run_vsearch(fasta, threshold):
    """Run VSEARCH cluster_fast."""
    uc_path = str(OUT_DIR / f"vsearch_t{threshold}.uc")
    cmd = [
        VSEARCH_BIN, "--cluster_fast", str(fasta),
        "--id", str(threshold), "--uc", uc_path, "--threads", str(THREADS),
    ]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        return None, elapsed
    clusters = {}
    with open(uc_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            rec_type = parts[0]
            cluster_id = int(parts[1])
            seq_id = parts[8]
            if rec_type in ("S", "H"):
                clusters[seq_id] = cluster_id
    return clusters, elapsed


if __name__ == "__main__":
    print("=" * 90)
    print("OPEN_ORTHOBENCH BENCHMARK")
    print("=" * 90)

    # Step 1: Concatenate proteomes
    print("\nStep 1: Preparing input...", flush=True)
    fasta = concatenate_proteomes()

    tools = [
        ("ClustKIT (fast)", lambda f, t: run_clustkit_mode(f, t, "fast")),
        ("ClustKIT (default)", lambda f, t: run_clustkit_mode(f, t, "default")),
        ("ClustKIT (sensitive)", lambda f, t: run_clustkit_mode(f, t, "sensitive")),
        ("MMseqs2", run_mmseqs),
        ("DeepClust", run_deepclust),
        ("Linclust", run_linclust),
        ("CD-HIT", run_cdhit),
        ("VSEARCH", run_vsearch),
    ]

    all_results = {}

    for threshold in [0.5, 0.7]:
        print(f"\n{'='*80}")
        print(f"Threshold = {threshold}")
        print(f"{'='*80}")

        for tool_name, runner in tools:
            print(f"\n  {tool_name} (t={threshold})...", flush=True)
            try:
                clusters, elapsed = runner(fasta, threshold)
                if clusters is None:
                    print(f"    FAILED ({elapsed:.1f}s)")
                    all_results[f"{tool_name}_t{threshold}"] = {"error": "failed", "time": elapsed}
                    continue

                n_clusters = len(set(clusters.values()))
                print(f"    {n_clusters} clusters in {elapsed:.1f}s", flush=True)

                # Convert to OrthoBench format
                ob_file = OUT_DIR / f"{tool_name.lower()}_t{threshold}_orthogroups.txt"
                clusters_to_orthobench_format(clusters, ob_file)

                # Run benchmark.py
                print(f"    Running OrthoBench evaluation...", flush=True)
                stdout, stderr = run_benchmark_py(ob_file)
                print(f"    {stdout.strip()}" if stdout.strip() else f"    (no stdout)")
                if stderr.strip():
                    # Only print non-empty stderr lines
                    for line in stderr.strip().split("\n")[:5]:
                        print(f"    stderr: {line}")

                all_results[f"{tool_name}_t{threshold}"] = {
                    "n_clusters": n_clusters,
                    "time": round(elapsed, 2),
                    "benchmark_stdout": stdout,
                    "benchmark_stderr": stderr,
                }
            except Exception as e:
                print(f"    ERROR: {e}")
                all_results[f"{tool_name}_t{threshold}"] = {"error": str(e)}

    # Save results
    results_file = OUT_DIR / "orthobench_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")
