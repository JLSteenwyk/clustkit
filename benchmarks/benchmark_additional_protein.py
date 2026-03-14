"""Additional protein clustering benchmarks: SCOPe and UniRef.

C2 from the publication plan:
- SCOPe 2.08: Structural classification ground truth
- UniRef50/90 subset: UniRef cluster assignments as ground truth

Evaluates ClustKIT, CD-HIT, MMseqs2, Linclust, VSEARCH at multiple thresholds.
"""

import argparse
import json
import os
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.pipeline import run_pipeline

from benchmark_pfam_concordance import (
    evaluate_tool,
    parse_cdhit_clusters,
    parse_mmseqs_clusters,
    parse_vsearch_clusters,
    pairwise_precision_recall_f1,
    run_cdhit,
    run_mmseqs,
    run_mmseqs_linclust,
    run_vsearch,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "additional_protein_results"

SCOPE_THRESHOLDS = [0.3, 0.4, 0.5, 0.7, 0.9]
UNIREF_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]


# ---------------------------------------------------------------------------
# SCOPe data loading
# ---------------------------------------------------------------------------

def load_scope_data(fasta_path, classification_path):
    """Load SCOPe FASTA sequences and superfamily-level classification.

    Args:
        fasta_path: Path to ASTRAL SCOPe-95 FASTA file.
        classification_path: Path to dir.cla.scope.2.08-stable.txt.

    Returns:
        Tuple of (fasta_path_for_clustering, ground_truth) where ground_truth
        maps sequence ID to superfamily SCCS string.
    """
    # Parse classification file: extract SID -> superfamily mapping
    sid_to_superfamily = {}
    with open(classification_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            sid = parts[0]
            superfamily_sccs = parts[3]
            sid_to_superfamily[sid] = superfamily_sccs

    print(f"  SCOPe classification: {len(sid_to_superfamily)} SIDs loaded")

    # Parse FASTA to find which SIDs are present
    ground_truth = {}
    seq_ids_in_fasta = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                # ASTRAL headers: >d1dlwa_ or >e4lztI1 etc.
                sid = line.split()[0][1:]
                seq_ids_in_fasta.append(sid)
                if sid in sid_to_superfamily:
                    ground_truth[sid] = sid_to_superfamily[sid]

    print(f"  FASTA sequences: {len(seq_ids_in_fasta)}")
    print(f"  Sequences with classification: {len(ground_truth)}")

    if not ground_truth:
        raise ValueError(
            "No overlap between FASTA sequence IDs and classification SIDs. "
            "Check that the FASTA file uses SCOPe SIDs in headers."
        )

    # Write a filtered FASTA containing only sequences with classification
    filtered_fasta = Path(fasta_path).parent / "scope_filtered.fasta"
    n_written = 0
    with open(fasta_path) as fin, open(filtered_fasta, "w") as fout:
        write_this = False
        for line in fin:
            if line.startswith(">"):
                sid = line.split()[0][1:]
                write_this = sid in ground_truth
                if write_this:
                    fout.write(line)
                    n_written += 1
            elif write_this:
                fout.write(line)

    superfamily_counts = Counter(ground_truth.values())
    print(f"  Filtered FASTA: {n_written} sequences, "
          f"{len(superfamily_counts)} superfamilies")

    return filtered_fasta, ground_truth


# ---------------------------------------------------------------------------
# UniRef data loading
# ---------------------------------------------------------------------------

def load_uniref_data(fasta_path, uniref_type, max_sequences=100000):
    """Load UniRef FASTA and extract cluster assignments as ground truth.

    UniRef90 FASTA header format:
        >UniRef90_P12345 Cluster: UniRef90_P12345 n=5 ...
    UniRef50 FASTA header format:
        >UniRef50_P12345 Cluster: UniRef50_P12345 n=5 ...

    The cluster representative ID (e.g. UniRef90_P12345) is the ground truth
    cluster label.

    Args:
        fasta_path: Path to UniRef FASTA file.
        uniref_type: "50" or "90".
        max_sequences: Maximum number of sequences to subsample.

    Returns:
        Tuple of (subsampled_fasta_path, ground_truth) where ground_truth
        maps sequence ID to cluster representative ID.
    """
    prefix = f"UniRef{uniref_type}_"

    # First pass: collect all sequence IDs and their cluster assignments
    all_records = []  # list of (header_line, seq_lines, cluster_id)
    current_header = None
    current_seq_lines = []

    print(f"  Reading UniRef{uniref_type} FASTA (this may take a while)...")
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                if current_header is not None:
                    cluster_id = _parse_uniref_cluster(current_header, prefix)
                    if cluster_id is not None:
                        all_records.append(
                            (current_header, current_seq_lines, cluster_id)
                        )
                current_header = line.rstrip("\n\r")
                current_seq_lines = []
            else:
                current_seq_lines.append(line.rstrip("\n\r"))

        # Last record
        if current_header is not None:
            cluster_id = _parse_uniref_cluster(current_header, prefix)
            if cluster_id is not None:
                all_records.append(
                    (current_header, current_seq_lines, cluster_id)
                )

    print(f"  Total sequences parsed: {len(all_records)}")

    # Subsample if needed
    if len(all_records) > max_sequences:
        random.seed(42)
        all_records = random.sample(all_records, max_sequences)
        print(f"  Subsampled to {max_sequences} sequences")

    # Write subsampled FASTA and build ground truth
    subsampled_fasta = Path(fasta_path).parent / f"uniref{uniref_type}_subsample.fasta"
    ground_truth = {}

    with open(subsampled_fasta, "w") as fout:
        for header, seq_lines, cluster_id in all_records:
            seq_id = header.split()[0][1:]  # strip '>'
            ground_truth[seq_id] = cluster_id
            fout.write(header + "\n")
            fout.write("\n".join(seq_lines) + "\n")

    cluster_counts = Counter(ground_truth.values())
    n_singletons = sum(1 for c in cluster_counts.values() if c == 1)
    print(f"  Subsampled FASTA: {len(ground_truth)} sequences, "
          f"{len(cluster_counts)} clusters "
          f"({n_singletons} singletons)")

    return subsampled_fasta, ground_truth


def _parse_uniref_cluster(header, prefix):
    """Extract cluster representative ID from a UniRef FASTA header.

    Looks for 'Cluster: UniRefXX_ACCESSION' in the header. Falls back to
    the sequence ID itself (the first word after '>') if no Cluster field
    is found, since the representative sequence's own ID IS its cluster ID.

    Returns:
        Cluster representative ID string, or None if header is malformed.
    """
    # Try to find explicit "Cluster: ..." annotation
    idx = header.find("Cluster: ")
    if idx != -1:
        rest = header[idx + len("Cluster: "):]
        cluster_rep = rest.split()[0]
        return cluster_rep

    # Fallback: use the sequence ID itself as its cluster assignment
    parts = header.split()
    if parts:
        seq_id = parts[0].lstrip(">")
        if seq_id.startswith(prefix) or seq_id.startswith("UniRef"):
            return seq_id

    return None


# ---------------------------------------------------------------------------
# Run a single tool and evaluate
# ---------------------------------------------------------------------------

def run_clustkit(fasta_path, output_dir, threshold, threads=4):
    """Run ClustKIT pipeline and return (clusters_dict, elapsed_seconds)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "input": str(fasta_path),
        "output": str(out_dir),
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

    start = time.perf_counter()
    try:
        run_pipeline(config)
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"ClustKIT failed: {e}")
        return None, elapsed
    elapsed = time.perf_counter() - start

    clusters = {}
    tsv_path = out_dir / "clusters.tsv"
    if tsv_path.exists():
        with open(tsv_path) as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    clusters[parts[0]] = int(parts[1])

    return clusters, elapsed


def run_and_evaluate(tool_name, run_fn, fasta_path, output_prefix,
                     threshold, ground_truth, threads=4):
    """Run a tool and evaluate its output against ground truth.

    Args:
        tool_name: Display name for the tool.
        run_fn: Callable(fasta_path, output_prefix, threshold, threads) ->
                (clusters_dict_or_None, elapsed).
        fasta_path: Path to input FASTA.
        output_prefix: Output path prefix.
        threshold: Clustering identity threshold.
        ground_truth: Dict mapping seq_id -> ground truth label.
        threads: Thread count.

    Returns:
        Dict with evaluation metrics and runtime, or error dict.
    """
    print(f"  {tool_name} (t={threshold})...", end=" ", flush=True)

    clusters, elapsed = run_fn(fasta_path, output_prefix, threshold, threads)

    if clusters is None or len(clusters) == 0:
        print(f"FAILED ({elapsed:.2f}s)")
        return {"error": "failed", "runtime_seconds": round(elapsed, 2)}

    evaluation = evaluate_tool(ground_truth, clusters)
    evaluation["runtime_seconds"] = round(elapsed, 2)

    if "error" not in evaluation:
        print(
            f"{evaluation['n_predicted_clusters']} clusters, "
            f"ARI={evaluation['ARI']}, "
            f"P={evaluation['pairwise_precision']}, "
            f"R={evaluation['pairwise_recall']}, "
            f"F1={evaluation['pairwise_F1']}, "
            f"{elapsed:.2f}s"
        )
    else:
        print(f"EVAL ERROR: {evaluation['error']} ({elapsed:.2f}s)")

    return evaluation


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def benchmark_scope(args):
    """Run SCOPe superfamily benchmark across all tools and thresholds."""
    print("=" * 120)
    print("SCOPe 2.08 SUPERFAMILY BENCHMARK")
    print("=" * 120)
    print()

    fasta_path, ground_truth = load_scope_data(
        args.scope_fasta, args.scope_classification
    )
    print()

    thresholds = args.thresholds if args.thresholds else SCOPE_THRESHOLDS
    results_dir = OUTPUT_DIR / "scope"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for threshold in thresholds:
        print("-" * 120)
        print(f"Threshold = {threshold}")
        print("-" * 120)

        scenario = {}

        # ClustKIT
        clustkit_out = results_dir / f"clustkit_t{threshold}"
        evaluation = run_and_evaluate(
            "ClustKIT", run_clustkit, fasta_path,
            clustkit_out, threshold, ground_truth, args.threads,
        )
        scenario["clustkit"] = evaluation

        # CD-HIT
        cdhit_prefix = results_dir / f"cdhit_t{threshold}"
        evaluation = run_and_evaluate(
            "CD-HIT", run_cdhit, fasta_path,
            cdhit_prefix, threshold, ground_truth, args.threads,
        )
        scenario["cdhit"] = evaluation

        # MMseqs2
        mmseqs_prefix = results_dir / f"mmseqs_t{threshold}"
        evaluation = run_and_evaluate(
            "MMseqs2", run_mmseqs, fasta_path,
            mmseqs_prefix, threshold, ground_truth, args.threads,
        )
        scenario["mmseqs2"] = evaluation

        # Linclust
        linclust_prefix = results_dir / f"linclust_t{threshold}"
        evaluation = run_and_evaluate(
            "Linclust", run_mmseqs_linclust, fasta_path,
            linclust_prefix, threshold, ground_truth, args.threads,
        )
        scenario["linclust"] = evaluation

        # VSEARCH
        vsearch_prefix = results_dir / f"vsearch_t{threshold}"
        evaluation = run_and_evaluate(
            "VSEARCH", run_vsearch, fasta_path,
            vsearch_prefix, threshold, ground_truth, args.threads,
        )
        scenario["vsearch"] = evaluation

        all_results[str(threshold)] = scenario
        print()

    _print_summary("SCOPe Superfamily", thresholds, all_results)

    results_file = results_dir / "scope_benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    return all_results


def benchmark_uniref(args):
    """Run UniRef benchmark across all tools and thresholds."""
    uniref_type = args.uniref_type

    print("=" * 120)
    print(f"UniRef{uniref_type} BENCHMARK")
    print("=" * 120)
    print()

    fasta_path, ground_truth = load_uniref_data(
        args.uniref_fasta, uniref_type, args.max_sequences
    )
    print()

    # Default thresholds depend on UniRef type
    if args.thresholds:
        thresholds = args.thresholds
    elif uniref_type == "90":
        thresholds = [0.7, 0.9]
    elif uniref_type == "50":
        thresholds = [0.3, 0.5]
    else:
        thresholds = UNIREF_THRESHOLDS

    results_dir = OUTPUT_DIR / f"uniref{uniref_type}"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for threshold in thresholds:
        print("-" * 120)
        print(f"Threshold = {threshold}")
        print("-" * 120)

        scenario = {}

        # ClustKIT
        clustkit_out = results_dir / f"clustkit_t{threshold}"
        evaluation = run_and_evaluate(
            "ClustKIT", run_clustkit, fasta_path,
            clustkit_out, threshold, ground_truth, args.threads,
        )
        scenario["clustkit"] = evaluation

        # CD-HIT
        cdhit_prefix = results_dir / f"cdhit_t{threshold}"
        evaluation = run_and_evaluate(
            "CD-HIT", run_cdhit, fasta_path,
            cdhit_prefix, threshold, ground_truth, args.threads,
        )
        scenario["cdhit"] = evaluation

        # MMseqs2
        mmseqs_prefix = results_dir / f"mmseqs_t{threshold}"
        evaluation = run_and_evaluate(
            "MMseqs2", run_mmseqs, fasta_path,
            mmseqs_prefix, threshold, ground_truth, args.threads,
        )
        scenario["mmseqs2"] = evaluation

        # Linclust
        linclust_prefix = results_dir / f"linclust_t{threshold}"
        evaluation = run_and_evaluate(
            "Linclust", run_mmseqs_linclust, fasta_path,
            linclust_prefix, threshold, ground_truth, args.threads,
        )
        scenario["linclust"] = evaluation

        # VSEARCH
        vsearch_prefix = results_dir / f"vsearch_t{threshold}"
        evaluation = run_and_evaluate(
            "VSEARCH", run_vsearch, fasta_path,
            vsearch_prefix, threshold, ground_truth, args.threads,
        )
        scenario["vsearch"] = evaluation

        all_results[str(threshold)] = scenario
        print()

    _print_summary(f"UniRef{uniref_type}", thresholds, all_results)

    results_file = results_dir / f"uniref{uniref_type}_benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    return all_results


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _print_summary(benchmark_name, thresholds, all_results):
    """Print a formatted summary table."""
    print("=" * 120)
    print(f"SUMMARY: {benchmark_name}")
    print("=" * 120)
    print()
    print(
        f"{'Thresh':<8} {'Tool':<10} {'Clust':>6} {'ARI':>8} {'NMI':>8} "
        f"{'Homog':>8} {'Compl':>8} {'P(pw)':>8} {'R(pw)':>8} "
        f"{'F1(pw)':>8} {'Time':>10}"
    )
    print("-" * 120)

    tool_display = [
        ("ClustKIT", "clustkit"),
        ("CD-HIT", "cdhit"),
        ("MMseqs2", "mmseqs2"),
        ("Linclust", "linclust"),
        ("VSEARCH", "vsearch"),
    ]

    for t in thresholds:
        r = all_results.get(str(t), {})
        for display_name, key in tool_display:
            res = r.get(key, {})
            if "error" in res:
                rt = res.get("runtime_seconds", "?")
                print(
                    f"{t:<8} {display_name:<10} {'FAIL':>6} {'':>8} "
                    f"{'':>8} {'':>8} {'':>8} {'':>8} {'':>8} "
                    f"{'':>8} {rt:>9}s"
                )
            elif res:
                print(
                    f"{t:<8} {display_name:<10} "
                    f"{res['n_predicted_clusters']:>6} "
                    f"{res['ARI']:>8.4f} {res['NMI']:>8.4f} "
                    f"{res['homogeneity']:>8.4f} "
                    f"{res['completeness']:>8.4f} "
                    f"{res['pairwise_precision']:>8.4f} "
                    f"{res['pairwise_recall']:>8.4f} "
                    f"{res['pairwise_F1']:>8.4f} "
                    f"{res['runtime_seconds']:>9.2f}s"
                )
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Additional protein clustering benchmarks (SCOPe, UniRef).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run SCOPe benchmark
  python benchmark_additional_protein.py scope \\
      --scope-fasta data/scope/astral-scope-95.fa \\
      --scope-classification data/scope/dir.cla.scope.2.08-stable.txt

  # Run UniRef90 benchmark
  python benchmark_additional_protein.py uniref \\
      --uniref-fasta data/uniref/uniref90.fasta \\
      --uniref-type 90 --max-sequences 100000

  # Run both benchmarks
  python benchmark_additional_protein.py all \\
      --scope-fasta data/scope/astral-scope-95.fa \\
      --scope-classification data/scope/dir.cla.scope.2.08-stable.txt \\
      --uniref-fasta data/uniref/uniref90.fasta \\
      --uniref-type 90
""",
    )

    parser.add_argument(
        "benchmark",
        choices=["scope", "uniref", "all"],
        help="Which benchmark to run: scope, uniref, or all.",
    )

    # SCOPe arguments
    parser.add_argument(
        "--scope-fasta",
        type=str,
        default=None,
        help="Path to SCOPe ASTRAL FASTA file (e.g. astral-scope-95.fa).",
    )
    parser.add_argument(
        "--scope-classification",
        type=str,
        default=None,
        help="Path to SCOPe classification file "
             "(e.g. dir.cla.scope.2.08-stable.txt).",
    )

    # UniRef arguments
    parser.add_argument(
        "--uniref-fasta",
        type=str,
        default="/mnt/85740f55-8e9a-4214-9500-be446866627e/uniref50/uniref50.fasta",
        help="Path to UniRef FASTA file.",
    )
    parser.add_argument(
        "--uniref-type",
        type=str,
        choices=["50", "90"],
        default="50",
        help="UniRef type: 50 or 90 (default: 50).",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=100000,
        help="Max sequences to subsample for UniRef (default: 100000).",
    )

    # Shared arguments
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads for all tools (default: 4).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Identity thresholds to test. If not set, uses benchmark-specific "
             "defaults.",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.benchmark in ("scope", "all"):
        if not args.scope_fasta or not args.scope_classification:
            parser.error(
                "SCOPe benchmark requires --scope-fasta and "
                "--scope-classification."
            )
        if not Path(args.scope_fasta).exists():
            parser.error(f"SCOPe FASTA not found: {args.scope_fasta}")
        if not Path(args.scope_classification).exists():
            parser.error(
                f"SCOPe classification not found: {args.scope_classification}"
            )

    if args.benchmark in ("uniref", "all"):
        if not args.uniref_fasta:
            parser.error("UniRef benchmark requires --uniref-fasta.")
        if not Path(args.uniref_fasta).exists():
            parser.error(f"UniRef FASTA not found: {args.uniref_fasta}")

    # Run benchmarks
    if args.benchmark in ("scope", "all"):
        benchmark_scope(args)
        print()

    if args.benchmark in ("uniref", "all"):
        benchmark_uniref(args)


if __name__ == "__main__":
    main()
