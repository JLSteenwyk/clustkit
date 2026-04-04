"""Thread scaling benchmark with current code (SW + Leiden + bw=100).

Tests ClustKIT and MMseqs2 at 1, 2, 4, 8, 16, 32 threads on 22K Pfam at t=0.5.
Also captures per-phase timing breakdown.
"""
import json
import os
import subprocess
import sys
import tempfile
import shutil
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MMSEQS_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/mmseqs/bin/mmseqs"
FASTA = "benchmarks/data/pfam_mixed.fasta"
OUT_DIR = Path("benchmarks/data/thread_scaling_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_clustkit(fasta, threshold, threads):
    """Run ClustKIT and capture per-phase timings from log."""
    import logging
    import numba
    import numpy as np
    numba.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)

    from clustkit.pipeline import run_pipeline

    out_dir = OUT_DIR / f"clustkit_t{threads}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Capture log output for phase timing
    log_capture = []
    handler = logging.Handler()
    handler.emit = lambda record: log_capture.append(record.getMessage())
    logger = logging.getLogger("clustkit")
    logger.addHandler(handler)

    config = {
        "input": fasta,
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
    total = time.perf_counter() - start

    logger.removeHandler(handler)

    # Parse phase timings from log
    phase_times = {}
    for msg in log_capture:
        if " done (" in msg:
            # e.g. "Phase 3: Pairwise similarity (SW local alignment (C/OpenMP)) done (123.45s)"
            try:
                phase_name = msg.split(" done (")[0].strip()
                time_str = msg.split(" done (")[1].rstrip(")")
                phase_times[phase_name] = float(time_str.rstrip("s"))
            except (IndexError, ValueError):
                pass

    return total, phase_times


def run_mmseqs(fasta, threshold, threads):
    """Run MMseqs2."""
    tmp_dir = tempfile.mkdtemp(prefix="mmseqs_ts_")
    prefix = str(OUT_DIR / f"mmseqs_t{threads}")
    cmd = [
        MMSEQS_BIN, "easy-cluster", str(fasta), prefix, tmp_dir,
        "--min-seq-id", str(threshold), "--threads", str(threads),
        "-c", "0.8", "--cov-mode", "0",
    ]
    start = time.perf_counter()
    subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.perf_counter() - start
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return elapsed


if __name__ == "__main__":
    threshold = 0.5
    thread_counts = [1, 2, 4, 8, 16, 32]

    print("=" * 90)
    print(f"THREAD SCALING BENCHMARK (22K Pfam, t={threshold}, SW+Leiden+bw=100)")
    print("=" * 90)

    results = {"clustkit": {}, "mmseqs2": {}}

    print(f"\n{'Threads':>8} {'CK Time':>10} {'MMseqs2':>10} {'CK Speedup':>12}")
    print("-" * 45)

    ck_base = None
    mm_base = None

    for threads in thread_counts:
        # ClustKIT
        ck_time, phase_times = run_clustkit(FASTA, threshold, threads)
        results["clustkit"][threads] = {
            "total": round(ck_time, 2),
            "phases": {k: round(v, 2) for k, v in phase_times.items()},
        }
        if ck_base is None:
            ck_base = ck_time

        # MMseqs2
        mm_time = run_mmseqs(FASTA, threshold, threads)
        results["mmseqs2"][threads] = {"total": round(mm_time, 2)}
        if mm_base is None:
            mm_base = mm_time

        ck_speedup = ck_base / ck_time
        print(f"{threads:>8} {ck_time:>9.1f}s {mm_time:>9.1f}s {ck_speedup:>11.1f}x", flush=True)

    # Phase breakdown at 8 threads
    print(f"\nPhase breakdown (ClustKIT, 8 threads):")
    phases_8 = results["clustkit"].get(8, {}).get("phases", {})
    for phase, t in sorted(phases_8.items()):
        print(f"  {phase}: {t:.2f}s")

    out_file = OUT_DIR / "thread_scaling_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")
