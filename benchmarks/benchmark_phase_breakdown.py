"""Runtime phase breakdown for ClustKIT.

Shows where time goes: Phase 0 (read), Phase 1 (sketch), Phase 2 (LSH),
Phase 3 (alignment), Phase 4 (graph), Phase 5 (clustering), Phase 6 (reps).

Tests on 22K Pfam at t=0.3 and t=0.5, and 133K Pfam at t=0.5.
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import numba

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.pipeline import run_pipeline

OUT_DIR = Path("benchmarks/data/phase_breakdown")
OUT_DIR.mkdir(parents=True, exist_ok=True)

THREADS = 8
DATASETS = {
    "22K_pfam": "benchmarks/data/pfam_mixed.fasta",
    "133K_pfam": "benchmarks/data/pfam_full/pfam_benchmark_large.fasta",
}


def run_with_phase_timing(fasta, threshold, label):
    """Run ClustKIT default and capture per-phase timings."""
    numba.set_num_threads(THREADS)
    os.environ["OMP_NUM_THREADS"] = str(THREADS)

    # Capture log messages for phase timing
    log_msgs = []
    handler = logging.Handler()
    handler.emit = lambda record: log_msgs.append(record.getMessage())
    logger = logging.getLogger("clustkit")
    old_level = logger.level
    logger.addHandler(handler)

    out_dir = OUT_DIR / f"run_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

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
        "threads": THREADS,
        "format": "tsv",
        "use_c_ext": True,
        "band_width": 100,
        "block": "off",
        "cascade": "off",
    }

    total_start = time.perf_counter()
    run_pipeline(config)
    total_time = time.perf_counter() - total_start

    logger.removeHandler(handler)

    # Parse phase timings
    phases = {}
    for msg in log_msgs:
        if " done (" in msg:
            try:
                name = msg.split(" done (")[0].strip()
                t = float(msg.split(" done (")[1].rstrip(")").rstrip("s"))
                phases[name] = t
            except (IndexError, ValueError):
                pass

    # Also extract key stats
    stats = {}
    for msg in log_msgs:
        if "candidate pairs" in msg.lower() and "Found" in msg:
            try:
                stats["candidate_pairs"] = int(msg.split("Found")[1].split("candidate")[0].strip())
            except (IndexError, ValueError):
                pass
        if "pairs above threshold" in msg:
            try:
                stats["filtered_pairs"] = int(msg.split()[0])
            except (IndexError, ValueError):
                pass
        if "clusters" in msg and "nodes" not in msg and "Done" not in msg:
            try:
                n = int(msg.split()[0])
                if n > 0:
                    stats["n_clusters"] = n
            except (IndexError, ValueError):
                pass

    return {
        "total": round(total_time, 2),
        "phases": {k: round(v, 2) for k, v in phases.items()},
        "stats": stats,
    }


if __name__ == "__main__":
    configs = [
        ("22K_pfam", 0.3, "22k_t03"),
        ("22K_pfam", 0.5, "22k_t05"),
        ("133K_pfam", 0.5, "133k_t05"),
    ]

    print("=" * 80)
    print(f"RUNTIME PHASE BREAKDOWN (ClustKIT default, {THREADS} threads)")
    print("=" * 80)

    all_results = {}

    for dataset_name, threshold, label in configs:
        fasta = DATASETS[dataset_name]
        print(f"\n--- {dataset_name}, t={threshold} ---", flush=True)

        result = run_with_phase_timing(fasta, threshold, label)
        all_results[label] = result

        total = result["total"]
        print(f"  Total: {total:.1f}s")
        print(f"  {'Phase':<55} {'Time':>8} {'%':>6}")
        print(f"  {'-'*72}")
        for phase, t in sorted(result["phases"].items(), key=lambda x: -x[1]):
            pct = 100 * t / total if total > 0 else 0
            print(f"  {phase:<55} {t:>7.1f}s {pct:>5.1f}%")
        if result["stats"]:
            print(f"  Stats: {result['stats']}")

    out_file = OUT_DIR / "phase_breakdown.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_file}")
