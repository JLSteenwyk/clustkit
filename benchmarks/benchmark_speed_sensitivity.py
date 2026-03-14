#!/usr/bin/env python3
"""Speed-sensitivity trade-off benchmark.

Runs each tool at multiple sensitivity settings on the same SCOPe query/db
and measures ROC1, ROC5, MAP, Sensitivity@1%FDR, and search time.

Produces a scatter plot dataset like MMseqs2 paper Figure 2b.

Reuses the prepared SCOPe database from benchmark_scop_search.py.
"""

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.benchmark_scop_search import (
    classify_hit,
    compute_average_precision,
    compute_roc_n,
    compute_sensitivity_at_fdr,
    parse_scope_classification,
    read_fasta,
    write_fasta,
    RankedHit,
)

# ──────────────────────────────────────────────────────────────────────
# Tool paths
# ──────────────────────────────────────────────────────────────────────

MMSEQS_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/mmseqs/bin/mmseqs"
BLAST_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/ncbi-blast-2.17.0+/bin/blastp"
DIAMOND_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/diamond-linux64/diamond"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Tool configurations
# ──────────────────────────────────────────────────────────────────────

MMSEQS_SETTINGS = [
    {"label": "MMseqs2 (s=1)", "args": ["-s", "1"]},
    {"label": "MMseqs2 (s=4)", "args": ["-s", "4"]},
    {"label": "MMseqs2 (s=5.7)", "args": ["-s", "5.7"]},
    {"label": "MMseqs2 (s=7.5)", "args": ["-s", "7.5"]},
]

DIAMOND_SETTINGS = [
    {"label": "DIAMOND (faster)", "args": ["--faster"]},
    {"label": "DIAMOND (fast)", "args": ["--fast"]},
    {"label": "DIAMOND (default)", "args": []},
    {"label": "DIAMOND (sensitive)", "args": ["--sensitive"]},
    {"label": "DIAMOND (more-sensitive)", "args": ["--more-sensitive"]},
    {"label": "DIAMOND (very-sensitive)", "args": ["--very-sensitive"]},
    {"label": "DIAMOND (ultra-sensitive)", "args": ["--ultra-sensitive"]},
]

BLAST_SETTINGS = [
    {"label": "BLAST (fast)", "args": ["-task", "blastp-fast"]},
    {"label": "BLAST (default)", "args": []},
]

CLUSTKIT_SETTINGS = [
    # Vary sensitivity via phase_a_topk + max_cands_per_query
    # Higher phase_a_topk improves Phase B candidate quality
    # Lower max_cands cuts alignment cost with minimal sensitivity loss
    {"label": "ClustKIT (fast)", "threshold": 0.1,
     "phase_a_topk": 3000, "max_cands_per_query": 300},
    {"label": "ClustKIT (default)", "threshold": 0.1,
     "phase_a_topk": 10000, "max_cands_per_query": 800},
    {"label": "ClustKIT (sensitive)", "threshold": 0.1,
     "phase_a_topk": 20000, "max_cands_per_query": 1000},
    {"label": "ClustKIT (very-sensitive)", "threshold": 0.1,
     "phase_a_topk": 50000, "max_cands_per_query": 1500},
]


def _run(cmd, timeout=None):
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        log.error(f"Command failed: {' '.join(str(c) for c in cmd)}")
        log.error(f"stderr: {result.stderr[:500]}")
        raise RuntimeError(f"Command failed with code {result.returncode}")
    return result


# ──────────────────────────────────────────────────────────────────────
# Evaluation (reused from benchmark_scop_search)
# ──────────────────────────────────────────────────────────────────────

def evaluate_hits(hits, metadata):
    """Evaluate search hits → aggregate metrics dict."""
    domain_info = metadata["domain_info"]
    family_sizes = metadata["family_sizes_in_db"]
    query_sids = set(metadata["query_sids"])

    hits_by_query = defaultdict(list)
    for qid, tid, score in hits:
        hits_by_query[qid].append((tid, score))

    roc1_values = []
    roc5_values = []
    map_values = []
    sens_values = []

    for qid in query_sids:
        q_info = domain_info.get(qid)
        if q_info is None:
            continue
        fam_key = str(q_info["family"])
        total_tp = family_sizes.get(fam_key, 1) - 1
        if total_tp <= 0:
            continue

        query_hits = hits_by_query.get(qid, [])
        query_hits.sort(key=lambda x: -x[1])

        ranked = []
        for tid, score in query_hits:
            label = classify_hit(qid, tid, domain_info)
            if label != "IGNORE":
                ranked.append(RankedHit(target_id=tid, score=score, label=label))

        roc1_values.append(compute_roc_n(ranked, 1, total_tp))
        roc5_values.append(compute_roc_n(ranked, 5, total_tp))
        map_values.append(compute_average_precision(ranked, total_tp))
        sens_values.append(compute_sensitivity_at_fdr(ranked, total_tp, 0.01))

    n = len(roc1_values)
    return {
        "n_queries": n,
        "mean_roc1": float(np.mean(roc1_values)) if n else 0,
        "mean_roc5": float(np.mean(roc5_values)) if n else 0,
        "mean_map": float(np.mean(map_values)) if n else 0,
        "mean_sens_1pct_fdr": float(np.mean(sens_values)) if n else 0,
    }


# ──────────────────────────────────────────────────────────────────────
# Search functions
# ──────────────────────────────────────────────────────────────────────

def search_clustkit(query_fasta, db_index, threshold, top_k, threads,
                    phase_a_topk=10000, max_cands_per_query=2000,
                    min_ungapped_score=0):
    """Run ClustKIT search at a given threshold/sensitivity."""
    import numba
    numba.set_num_threads(threads)
    from clustkit.io import read_sequences
    from clustkit.kmer_index import search_kmer_index

    query_ds = read_sequences(query_fasta, "protein")

    t0 = time.perf_counter()
    results = search_kmer_index(
        db_index, query_ds, threshold=threshold, top_k=top_k,
        phase_a_topk=phase_a_topk,
        max_cands_per_query=max_cands_per_query,
        min_ungapped_score=min_ungapped_score,
    )
    elapsed = time.perf_counter() - t0

    hits = []
    for qhits in results.hits:
        for h in qhits:
            # Use alignment score for ranking (analogous to -evalue in BLAST/MMseqs2)
            hits.append((h.query_id, h.target_id, h.score if h.score != 0 else h.identity))

    return hits, elapsed


def search_mmseqs2(query_db, target_db, out_dir, extra_args, threads, max_hits):
    """Run MMseqs2 search with given sensitivity args."""
    result_db = str(out_dir / "result")
    tmp_dir = str(out_dir / "tmp")

    # Clean previous results
    for p in out_dir.glob("result*"):
        p.unlink()
    if (out_dir / "tmp").exists():
        import shutil
        shutil.rmtree(out_dir / "tmp")

    t0 = time.perf_counter()
    _run([
        MMSEQS_BIN, "search",
        query_db, target_db, result_db, tmp_dir,
        "--threads", str(threads),
        "--max-seqs", str(max_hits),
    ] + extra_args)

    result_tsv = str(out_dir / "result.tsv")
    _run([
        MMSEQS_BIN, "convertalis",
        query_db, target_db, result_db, result_tsv,
        "--format-output", "query,target,pident,evalue,bits",
    ])
    elapsed = time.perf_counter() - t0

    hits = []
    with open(result_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                hits.append((parts[0], parts[1], -float(parts[3])))
    return hits, elapsed


def search_diamond(query_fasta, db_path, out_dir, extra_args, threads, max_hits):
    """Run DIAMOND search with given sensitivity args."""
    result_tsv = str(out_dir / "result.tsv")

    t0 = time.perf_counter()
    _run([
        DIAMOND_BIN, "blastp",
        "-q", query_fasta,
        "-d", db_path,
        "-o", result_tsv,
        "--threads", str(threads),
        "--max-target-seqs", str(max_hits),
        "--outfmt", "6", "qseqid", "sseqid", "pident", "evalue", "bitscore",
    ] + extra_args)
    elapsed = time.perf_counter() - t0

    hits = []
    with open(result_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                hits.append((parts[0], parts[1], -float(parts[3])))
    return hits, elapsed


def search_blast(query_fasta, db_path, out_dir, extra_args, threads, max_hits):
    """Run BLAST search with given task args."""
    result_tsv = str(out_dir / "result.tsv")

    t0 = time.perf_counter()
    _run([
        BLAST_BIN,
        "-query", query_fasta,
        "-db", db_path,
        "-out", result_tsv,
        "-outfmt", "6 qseqid sseqid pident evalue bitscore",
        "-num_threads", str(threads),
        "-max_target_seqs", str(max_hits),
        "-evalue", "10",
    ] + extra_args, timeout=86400)
    elapsed = time.perf_counter() - t0

    hits = []
    with open(result_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                hits.append((parts[0], parts[1], -float(parts[3])))
    return hits, elapsed


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Speed-sensitivity trade-off benchmark"
    )
    parser.add_argument(
        "--scop-dir",
        default="benchmarks/data/scop_search_results",
        help="Directory with prepared SCOPe benchmark data (from benchmark_scop_search.py)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/data/speed_sensitivity_results",
        help="Output directory",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--num-queries", type=int, default=2000)
    parser.add_argument("--max-hits", type=int, default=500)
    parser.add_argument(
        "--tools", default="clustkit,mmseqs2,diamond",
        help="Comma-separated tools to test",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scop_dir = Path(args.scop_dir)
    tools = args.tools.split(",")

    # ── Load metadata ──
    with open(scop_dir / "metadata.json") as f:
        full_metadata = json.load(f)

    # ── Subsample queries ──
    all_query_sids = full_metadata["query_sids"]
    random.seed(42)
    if len(all_query_sids) > args.num_queries:
        query_sids = sorted(random.sample(all_query_sids, args.num_queries))
    else:
        query_sids = all_query_sids
    log.info(f"Using {len(query_sids)} queries (subsampled from {len(all_query_sids)})")

    # Write subsampled query FASTA
    all_seqs = dict(read_fasta(scop_dir / "queries.fasta"))
    sub_seqs = [(sid, all_seqs[sid]) for sid in query_sids if sid in all_seqs]
    query_fasta = str(out / "queries_subset.fasta")
    write_fasta(sub_seqs, query_fasta)
    log.info(f"Wrote {len(sub_seqs)} queries to {query_fasta}")

    # Update metadata for evaluation
    metadata = dict(full_metadata)
    metadata["query_sids"] = query_sids
    metadata["query_count"] = len(query_sids)

    db_fasta = full_metadata["db_fasta"]

    # ── Build indices ──
    # MMseqs2 query db for subsampled queries
    mmseqs_dir = out / "mmseqs_db"
    mmseqs_dir.mkdir(parents=True, exist_ok=True)
    if "mmseqs2" in tools:
        log.info("Building MMseqs2 query db for subset...")
        _run([MMSEQS_BIN, "createdb", query_fasta, str(mmseqs_dir / "query")])
        # Reuse target db from main benchmark if available, else build
        mmseqs_target = scop_dir / "mmseqs_db" / "db"
        if not mmseqs_target.with_suffix(".dbtype").exists():
            log.info("Building MMseqs2 target db...")
            _run([MMSEQS_BIN, "createdb", db_fasta, str(mmseqs_target)])

    # DIAMOND db
    diamond_dir = out / "diamond_db"
    diamond_dir.mkdir(parents=True, exist_ok=True)
    diamond_db = scop_dir / "diamond_db" / "db"
    if "diamond" in tools and not diamond_db.with_suffix(".dmnd").exists():
        log.info("Building DIAMOND db...")
        _run([DIAMOND_BIN, "makedb", "--in", db_fasta, "-d", str(diamond_db)])

    # BLAST db
    blast_db = scop_dir / "blast_db" / "db"
    if "blast" in tools and not Path(str(blast_db) + ".pdb").exists():
        log.info("Building BLAST db...")
        from benchmarks.benchmark_scop_search import MAKEBLASTDB_BIN
        _run([MAKEBLASTDB_BIN, "-in", db_fasta, "-dbtype", "prot", "-out", str(blast_db)])

    # ClustKIT db — build once at t=0.1 (most sensitive)
    clustkit_db_dir = out / "clustkit_db"
    db_index = None
    if "clustkit" in tools:
        if (clustkit_db_dir / "params.json").exists():
            log.info("Loading existing ClustKIT index...")
            from clustkit.database import load_database
            db_index = load_database(clustkit_db_dir)
        else:
            log.info("Building ClustKIT index (threshold=0.1, max sensitivity)...")
            import numba
            numba.set_num_threads(args.threads)
            from clustkit.database import build_database, save_database
            t0 = time.perf_counter()
            db_index = build_database(db_fasta, mode="protein", threshold=0.1, sensitivity="high")
            save_database(db_index, clustkit_db_dir)
            log.info(f"ClustKIT index built in {time.perf_counter() - t0:.1f}s")

    # ── Run all configurations ──
    all_results = []

    # ClustKIT
    if "clustkit" in tools:
        import numba
        numba.set_num_threads(args.threads)
        for cfg in CLUSTKIT_SETTINGS:
            label = cfg["label"]
            log.info(f"Running {label}...")
            hits, elapsed = search_clustkit(
                query_fasta, db_index, cfg["threshold"], args.max_hits, args.threads,
                phase_a_topk=cfg.get("phase_a_topk", 10000),
                max_cands_per_query=cfg.get("max_cands_per_query", 2000),
                min_ungapped_score=cfg.get("min_ungapped_score", 0),
            )
            metrics = evaluate_hits(hits, metadata)
            result = {"label": label, "tool": "clustkit", "time": elapsed, **metrics}
            all_results.append(result)
            log.info(
                f"  {label}: ROC1={metrics['mean_roc1']:.4f} "
                f"MAP={metrics['mean_map']:.4f} "
                f"Time={elapsed:.1f}s"
            )
            _save_and_print(all_results, out)

    # MMseqs2
    if "mmseqs2" in tools:
        mmseqs_query = str(mmseqs_dir / "query")
        mmseqs_target_str = str(scop_dir / "mmseqs_db" / "db")
        for cfg in MMSEQS_SETTINGS:
            label = cfg["label"]
            log.info(f"Running {label}...")
            try:
                hits, elapsed = search_mmseqs2(
                    mmseqs_query, mmseqs_target_str,
                    mmseqs_dir, cfg["args"], args.threads, args.max_hits,
                )
                metrics = evaluate_hits(hits, metadata)
                result = {"label": label, "tool": "mmseqs2", "time": elapsed, **metrics}
                all_results.append(result)
                log.info(
                    f"  {label}: ROC1={metrics['mean_roc1']:.4f} "
                    f"MAP={metrics['mean_map']:.4f} "
                    f"Time={elapsed:.1f}s"
                )
            except Exception as e:
                log.error(f"  {label} failed: {e}")
            _save_and_print(all_results, out)

    # DIAMOND
    if "diamond" in tools:
        for cfg in DIAMOND_SETTINGS:
            label = cfg["label"]
            log.info(f"Running {label}...")
            try:
                hits, elapsed = search_diamond(
                    query_fasta, str(diamond_db),
                    out, cfg["args"], args.threads, args.max_hits,
                )
                metrics = evaluate_hits(hits, metadata)
                result = {"label": label, "tool": "diamond", "time": elapsed, **metrics}
                all_results.append(result)
                log.info(
                    f"  {label}: ROC1={metrics['mean_roc1']:.4f} "
                    f"MAP={metrics['mean_map']:.4f} "
                    f"Time={elapsed:.1f}s"
                )
            except Exception as e:
                log.error(f"  {label} failed: {e}")
            _save_and_print(all_results, out)

    # BLAST
    if "blast" in tools:
        for cfg in BLAST_SETTINGS:
            label = cfg["label"]
            log.info(f"Running {label}...")
            try:
                hits, elapsed = search_blast(
                    query_fasta, str(blast_db),
                    out, cfg["args"], args.threads, args.max_hits,
                )
                metrics = evaluate_hits(hits, metadata)
                result = {"label": label, "tool": "blast", "time": elapsed, **metrics}
                all_results.append(result)
                log.info(
                    f"  {label}: ROC1={metrics['mean_roc1']:.4f} "
                    f"MAP={metrics['mean_map']:.4f} "
                    f"Time={elapsed:.1f}s"
                )
            except Exception as e:
                log.error(f"  {label} failed: {e}")
            _save_and_print(all_results, out)

    # ── Final summary ──
    _save_and_print(all_results, out, final=True)


def _save_and_print(results, out, final=False):
    """Save results and print current table."""
    with open(out / "speed_sensitivity_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if final:
        print("\n" + "=" * 100)
        print("SPEED-SENSITIVITY TRADE-OFF RESULTS")
        print("=" * 100)
    print(
        f"\n{'Label':35s} {'ROC1':>7s} {'ROC5':>7s} {'MAP':>7s} "
        f"{'S@1%FDR':>8s} {'Time(s)':>8s} {'Queries':>8s}"
    )
    print("-" * 100)
    for r in results:
        print(
            f"{r['label']:35s} "
            f"{r['mean_roc1']:7.4f} "
            f"{r['mean_roc5']:7.4f} "
            f"{r['mean_map']:7.4f} "
            f"{r['mean_sens_1pct_fdr']:8.4f} "
            f"{r['time']:8.1f} "
            f"{r['n_queries']:8d}"
        )
    print()


if __name__ == "__main__":
    main()
