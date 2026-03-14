#!/usr/bin/env python3
"""SCOPe-based protein sequence search benchmark.

Methodology based on Steinegger & Söding, Nature Biotechnology 2017 (MMseqs2).

Setup:
  - Queries: SCOPe 2.08 domain sequences (filtered subset)
  - Database: SCOPe domains + reversed decoys + optional UniProt background
  - TP:  same SCOP family (excluding self-hit)
  - FP:  different SCOP fold OR reversed/decoy sequence
  - IGN: same superfamily but different family, or non-SCOP background

Metrics:
  - ROC1:  mean fraction of TPs ranked above the 1st FP  (as in MMseqs2 paper)
  - ROC5:  mean fraction of TPs ranked above the 5th FP
  - MAP:   Mean Average Precision
  - Sensitivity at 1% FDR

Tools: ClustKIT (pre-indexed), MMseqs2, DIAMOND, BLAST

Usage:
  python benchmark_scop_search.py \\
      --scope-fasta /path/to/astral-scopedom-seqres-gd-all-2.08-stable.fa \\
      --scope-cla   /path/to/dir.cla.scope.2.08-stable.txt \\
      --threads 4 --threshold 0.3 \\
      [--swissprot-fasta /path/to/uniprot_sprot.fasta]
"""

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ──────────────────────────────────────────────────────────────────────
# Tool paths
# ──────────────────────────────────────────────────────────────────────

MMSEQS_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/mmseqs/bin/mmseqs"
BLAST_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/ncbi-blast-2.17.0+/bin/blastp"
MAKEBLASTDB_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/ncbi-blast-2.17.0+/bin/makeblastdb"
DIAMOND_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/diamond-linux64/diamond"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SCOPeDomain:
    sid: str
    classification: str  # e.g. a.1.1.1
    family_id: int
    superfamily_id: int
    fold_id: int
    class_id: int


@dataclass
class RankedHit:
    """A single hit with classification label."""
    target_id: str
    score: float          # identity (higher=better) or -evalue (higher=better)
    label: str            # "TP", "FP", or "IGNORE"


# ──────────────────────────────────────────────────────────────────────
# SCOPe parsing
# ──────────────────────────────────────────────────────────────────────

def parse_scope_classification(cla_path: str | Path) -> dict[str, SCOPeDomain]:
    """Parse SCOPe dir.cla file → {sid: SCOPeDomain}."""
    domains = {}
    with open(cla_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            sid = parts[0]
            classification = parts[3]
            ids = {}
            for item in parts[5].split(","):
                k, v = item.split("=")
                ids[k] = int(v)
            domains[sid] = SCOPeDomain(
                sid=sid,
                classification=classification,
                family_id=ids.get("fa", -1),
                superfamily_id=ids.get("sf", -1),
                fold_id=ids.get("cf", -1),
                class_id=ids.get("cl", -1),
            )
    return domains


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Read FASTA → [(id, sequence), ...]."""
    sequences = []
    header = None
    seq_parts = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if header is not None:
                    sequences.append((header, "".join(seq_parts)))
                header = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line.strip())
    if header is not None:
        sequences.append((header, "".join(seq_parts)))
    return sequences


def write_fasta(sequences: list[tuple[str, str]], path: str | Path):
    """Write [(id, sequence), ...] to FASTA."""
    with open(path, "w") as f:
        for seq_id, seq in sequences:
            f.write(f">{seq_id}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")


# ──────────────────────────────────────────────────────────────────────
# Phase 1: PREPARE
# ──────────────────────────────────────────────────────────────────────

def prepare(args) -> dict:
    """Parse SCOPe, filter, create database FASTA with decoys.

    Returns metadata dict saved to output_dir/metadata.json.
    """
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("Phase 1: PREPARE")

    # ── Parse SCOPe classification ──
    scope_domains = parse_scope_classification(args.scope_cla)
    log.info(f"  Parsed {len(scope_domains)} SCOPe domain classifications")

    # ── Read SCOPe sequences ──
    scope_seqs = read_fasta(args.scope_fasta)
    log.info(f"  Read {len(scope_seqs)} SCOPe sequences from FASTA")

    # ── Filter: keep only domains with classification + valid length ──
    min_len, max_len = 30, 5000
    filtered = []
    for sid, seq in scope_seqs:
        if sid not in scope_domains:
            continue
        if len(seq) < min_len or len(seq) > max_len:
            continue
        # Skip sequences with non-standard characters
        if any(c not in "ACDEFGHIKLMNPQRSTVWY" for c in seq.upper()):
            continue
        filtered.append((sid, seq))
    log.info(
        f"  After filtering (len {min_len}-{max_len}, valid AA, classified): "
        f"{len(filtered)} domains"
    )

    # ── Count families and select queries ──
    family_members = defaultdict(list)
    for sid, seq in filtered:
        fam = scope_domains[sid].family_id
        family_members[fam].append(sid)

    # Only use families with >= 2 members (need at least 1 query + 1 target)
    valid_families = {f: m for f, m in family_members.items() if len(m) >= 2}
    log.info(
        f"  {len(valid_families)} families with >=2 members "
        f"(of {len(family_members)} total)"
    )

    # Select queries: up to max_queries_per_family per family
    max_per_fam = args.max_queries_per_family
    random.seed(42)
    query_sids = set()
    for fam, members in valid_families.items():
        chosen = random.sample(members, min(max_per_fam, len(members)))
        query_sids.update(chosen)

    # Cap total queries
    if args.max_queries and len(query_sids) > args.max_queries:
        query_sids = set(random.sample(sorted(query_sids), args.max_queries))

    log.info(f"  Selected {len(query_sids)} query domains")

    # ── Build database and query FASTAs ──
    sid_to_seq = dict(filtered)
    query_seqs = [(sid, sid_to_seq[sid]) for sid in sorted(query_sids)]

    # Database = all SCOPe domains (targets for same-family matches)
    db_seqs = list(filtered)
    scope_db_count = len(db_seqs)
    log.info(f"  Database: {scope_db_count} SCOPe domains")

    # Add reversed SCOPe sequences as decoys
    reversed_seqs = [(f"REV_{sid}", seq[::-1]) for sid, seq in filtered]
    db_seqs.extend(reversed_seqs)
    log.info(f"  + {len(reversed_seqs)} reversed SCOPe decoys")

    # Optionally add SwissProt background
    swissprot_count = 0
    if args.swissprot_fasta:
        log.info(f"  Reading SwissProt: {args.swissprot_fasta}")
        sp_seqs = read_fasta(args.swissprot_fasta)
        # Filter to valid protein sequences
        sp_filtered = []
        for sid, seq in sp_seqs:
            if min_len <= len(seq) <= max_len:
                # Prefix to distinguish from SCOPe IDs
                sp_filtered.append((f"SP_{sid}", seq))
        swissprot_count = len(sp_filtered)
        db_seqs.extend(sp_filtered)
        # Also add reversed SwissProt as decoys
        sp_reversed = [(f"REV_SP_{sid}", seq[::-1]) for sid, seq in sp_filtered]
        db_seqs.extend(sp_reversed)
        log.info(
            f"  + {swissprot_count} SwissProt sequences "
            f"+ {len(sp_reversed)} reversed SwissProt decoys"
        )

    log.info(f"  Total database: {len(db_seqs)} sequences")

    # Write FASTAs
    query_fasta = out / "queries.fasta"
    db_fasta = out / "database.fasta"
    write_fasta(query_seqs, query_fasta)
    write_fasta(db_seqs, db_fasta)
    log.info(f"  Written: {query_fasta} ({len(query_seqs)} seqs)")
    log.info(f"  Written: {db_fasta} ({len(db_seqs)} seqs)")

    # ── Build family size map (for computing total TPs per query) ──
    # For each query, count how many same-family domains are in the database
    # (excluding self)
    family_sizes_in_db = defaultdict(int)
    for sid, _ in filtered:  # all SCOPe domains are in the database
        fam = scope_domains[sid].family_id
        family_sizes_in_db[fam] += 1

    # ── Save metadata ──
    # Domain classification for all SCOPe domains in the database
    domain_info = {}
    for sid, _ in filtered:
        d = scope_domains[sid]
        domain_info[sid] = {
            "family": d.family_id,
            "superfamily": d.superfamily_id,
            "fold": d.fold_id,
            "class": d.class_id,
            "classification": d.classification,
        }

    metadata = {
        "query_count": len(query_seqs),
        "db_total": len(db_seqs),
        "db_scope_count": scope_db_count,
        "db_reversed_count": len(reversed_seqs),
        "db_swissprot_count": swissprot_count,
        "family_count": len(valid_families),
        "family_sizes_in_db": {str(k): v for k, v in family_sizes_in_db.items()},
        "query_sids": sorted(query_sids),
        "domain_info": domain_info,
        "query_fasta": str(query_fasta),
        "db_fasta": str(db_fasta),
    }

    meta_path = out / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"  Metadata saved to {meta_path}")

    # ── Length statistics ──
    q_lens = [len(seq) for _, seq in query_seqs]
    d_lens = [len(seq) for _, seq in db_seqs]
    log.info(
        f"  Query lengths: min={min(q_lens)}, max={max(q_lens)}, "
        f"mean={np.mean(q_lens):.0f}, median={np.median(q_lens):.0f}"
    )
    log.info(
        f"  Database lengths: min={min(d_lens)}, max={max(d_lens)}, "
        f"mean={np.mean(d_lens):.0f}, median={np.median(d_lens):.0f}"
    )

    return metadata


# ──────────────────────────────────────────────────────────────────────
# Phase 2: INDEX
# ──────────────────────────────────────────────────────────────────────

def build_indices(args, metadata: dict):
    """Build database indices for all tools."""
    out = Path(args.output_dir)
    db_fasta = metadata["db_fasta"]

    log.info("Phase 2: INDEX")

    tools = args.tools.split(",")

    # ── ClustKIT ──
    if "clustkit" in tools:
        log.info("  Building ClustKIT database index...")
        t0 = time.perf_counter()
        import numba
        numba.set_num_threads(args.threads)
        from clustkit.database import build_database, save_database

        idx_dir = out / "clustkit_db"
        db_index = build_database(
            db_fasta,
            mode="protein",
            threshold=args.threshold,
            sensitivity="high",
        )
        save_database(db_index, idx_dir)
        elapsed = time.perf_counter() - t0
        log.info(f"  ClustKIT index built in {elapsed:.1f}s → {idx_dir}")

    # ── MMseqs2 ──
    if "mmseqs2" in tools:
        log.info("  Building MMseqs2 database...")
        t0 = time.perf_counter()
        mmseqs_db = out / "mmseqs_db" / "db"
        mmseqs_db.parent.mkdir(parents=True, exist_ok=True)
        _run([MMSEQS_BIN, "createdb", db_fasta, str(mmseqs_db)])
        elapsed = time.perf_counter() - t0
        log.info(f"  MMseqs2 createdb in {elapsed:.1f}s")

        # Also create query db
        query_db = out / "mmseqs_db" / "query"
        _run([MMSEQS_BIN, "createdb", metadata["query_fasta"], str(query_db)])

    # ── DIAMOND ──
    if "diamond" in tools:
        log.info("  Building DIAMOND database...")
        t0 = time.perf_counter()
        diamond_db = out / "diamond_db" / "db"
        diamond_db.parent.mkdir(parents=True, exist_ok=True)
        _run([DIAMOND_BIN, "makedb", "--in", db_fasta, "-d", str(diamond_db)])
        elapsed = time.perf_counter() - t0
        log.info(f"  DIAMOND makedb in {elapsed:.1f}s")

    # ── BLAST ──
    if "blast" in tools:
        log.info("  Building BLAST database...")
        t0 = time.perf_counter()
        blast_db = out / "blast_db" / "db"
        blast_db.parent.mkdir(parents=True, exist_ok=True)
        _run([
            MAKEBLASTDB_BIN, "-in", db_fasta,
            "-dbtype", "prot", "-out", str(blast_db),
        ])
        elapsed = time.perf_counter() - t0
        log.info(f"  BLAST makeblastdb in {elapsed:.1f}s")


def _run(cmd, timeout=None):
    """Run a subprocess, raise on failure."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        log.error(f"Command failed: {' '.join(str(c) for c in cmd)}")
        log.error(f"stderr: {result.stderr[:1000]}")
        raise RuntimeError(f"Command failed with code {result.returncode}")
    return result


# ──────────────────────────────────────────────────────────────────────
# Phase 3: SEARCH
# ──────────────────────────────────────────────────────────────────────

def run_searches(args, metadata: dict) -> dict[str, list[tuple[str, str, float]]]:
    """Run searches with all tools.

    Returns: {tool_name: [(query_id, target_id, score), ...]}
    Score is identity for ClustKIT, -evalue for others (higher = more confident).
    """
    out = Path(args.output_dir)
    results = {}
    tools = args.tools.split(",")

    log.info("Phase 3: SEARCH")

    search_funcs = {
        "clustkit": _search_clustkit,
        "mmseqs2": _search_mmseqs2,
        "diamond": _search_diamond,
        "blast": _search_blast,
    }

    for tool_name in ["clustkit", "mmseqs2", "diamond", "blast"]:
        if tool_name not in tools:
            continue
        # Skip if raw hits already saved from a previous run
        hits_path = out / f"{tool_name}_raw_hits.json"
        if hits_path.exists():
            log.info(f"  Loading cached {tool_name} results from {hits_path}")
            with open(hits_path) as f:
                results[tool_name] = json.load(f)
            continue
        try:
            hits = search_funcs[tool_name](args, metadata, out)
            results[tool_name] = hits
            # Save immediately so we don't lose results on later failures
            with open(hits_path, "w") as f:
                json.dump(hits, f)
            log.info(f"  Saved {tool_name} raw hits to {hits_path}")
        except Exception as e:
            log.error(f"  {tool_name} search failed: {e}")
            log.error("  Continuing with remaining tools...")

    return results


def _search_clustkit(args, metadata, out):
    """ClustKIT search using pre-built index."""
    log.info("  Running ClustKIT search (pre-indexed)...")
    import numba
    numba.set_num_threads(args.threads)
    from clustkit.database import load_database
    from clustkit.io import read_sequences
    from clustkit.search import search_with_index

    t0 = time.perf_counter()

    # Load pre-built index
    db_index = load_database(out / "clustkit_db")

    # Read queries
    query_dataset = read_sequences(metadata["query_fasta"], "protein")

    # Search
    search_results = search_with_index(
        db_index,
        query_dataset,
        threshold=args.threshold,
        top_k=args.max_hits,
    )

    elapsed = time.perf_counter() - t0
    log.info(f"  ClustKIT search completed in {elapsed:.1f}s")

    # Convert to common format: (query_id, target_id, score)
    hits = []
    for query_hits in search_results.hits:
        for h in query_hits:
            hits.append((h.query_id, h.target_id, h.identity))

    log.info(f"  ClustKIT: {len(hits)} total hits")

    # Save timing
    timing_path = out / "clustkit_timing.json"
    with open(timing_path, "w") as f:
        json.dump({
            "search_time": elapsed,
            "num_candidates": search_results.num_candidates,
            "num_aligned": search_results.num_aligned,
        }, f, indent=2)

    return hits


def _search_mmseqs2(args, metadata, out):
    """MMseqs2 search."""
    log.info("  Running MMseqs2 search...")
    t0 = time.perf_counter()

    query_db = str(out / "mmseqs_db" / "query")
    target_db = str(out / "mmseqs_db" / "db")
    result_db = str(out / "mmseqs_db" / "result")
    tmp_dir = str(out / "mmseqs_db" / "tmp")

    _run([
        MMSEQS_BIN, "search",
        query_db, target_db, result_db, tmp_dir,
        "--threads", str(args.threads),
        "-s", "7.5",  # high sensitivity
        "--max-seqs", str(args.max_hits),
    ])

    # Convert to tab-separated
    result_tsv = str(out / "mmseqs2_results.tsv")
    _run([
        MMSEQS_BIN, "convertalis",
        query_db, target_db, result_db, result_tsv,
        "--format-output", "query,target,pident,evalue,bits",
    ])

    elapsed = time.perf_counter() - t0
    log.info(f"  MMseqs2 search completed in {elapsed:.1f}s")

    # Parse results: rank by evalue (lower = better → use -evalue as score)
    hits = []
    with open(result_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            query_id = parts[0]
            target_id = parts[1]
            evalue = float(parts[3])
            # Use negative evalue so higher = more confident
            hits.append((query_id, target_id, -evalue))

    log.info(f"  MMseqs2: {len(hits)} total hits")

    with open(out / "mmseqs2_timing.json", "w") as f:
        json.dump({"search_time": elapsed}, f, indent=2)

    return hits


def _search_diamond(args, metadata, out):
    """DIAMOND search."""
    log.info("  Running DIAMOND search...")
    t0 = time.perf_counter()

    result_tsv = str(out / "diamond_results.tsv")
    _run([
        DIAMOND_BIN, "blastp",
        "-q", metadata["query_fasta"],
        "-d", str(out / "diamond_db" / "db"),
        "-o", result_tsv,
        "--threads", str(args.threads),
        "--max-target-seqs", str(args.max_hits),
        "--outfmt", "6", "qseqid", "sseqid", "pident", "evalue", "bitscore",
        "--sensitive",
    ])

    elapsed = time.perf_counter() - t0
    log.info(f"  DIAMOND search completed in {elapsed:.1f}s")

    hits = []
    with open(result_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            query_id = parts[0]
            target_id = parts[1]
            evalue = float(parts[3])
            hits.append((query_id, target_id, -evalue))

    log.info(f"  DIAMOND: {len(hits)} total hits")

    with open(out / "diamond_timing.json", "w") as f:
        json.dump({"search_time": elapsed}, f, indent=2)

    return hits


def _search_blast(args, metadata, out):
    """BLAST search."""
    log.info("  Running BLAST search...")
    t0 = time.perf_counter()

    result_tsv = str(out / "blast_results.tsv")
    _run([
        BLAST_BIN,
        "-query", metadata["query_fasta"],
        "-db", str(out / "blast_db" / "db"),
        "-out", result_tsv,
        "-outfmt", "6 qseqid sseqid pident evalue bitscore",
        "-num_threads", str(args.threads),
        "-max_target_seqs", str(args.max_hits),
        "-evalue", "10",
    ], timeout=86400)  # 24h timeout for BLAST

    elapsed = time.perf_counter() - t0
    log.info(f"  BLAST search completed in {elapsed:.1f}s")

    hits = []
    with open(result_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            query_id = parts[0]
            target_id = parts[1]
            evalue = float(parts[3])
            hits.append((query_id, target_id, -evalue))

    log.info(f"  BLAST: {len(hits)} total hits")

    with open(out / "blast_timing.json", "w") as f:
        json.dump({"search_time": elapsed}, f, indent=2)

    return hits


# ──────────────────────────────────────────────────────────────────────
# Phase 4: EVALUATE
# ──────────────────────────────────────────────────────────────────────

def classify_hit(
    query_id: str,
    target_id: str,
    domain_info: dict,
) -> str:
    """Classify a hit as TP, FP, or IGNORE.

    TP:     same SCOP family (excluding self-hit)
    FP:     different SCOP fold, OR reversed/decoy sequence
    IGNORE: same superfamily different family, non-SCOP background, or self-hit
    """
    # Self-hit
    if target_id == query_id:
        return "IGNORE"

    # Reversed decoy → always FP
    if target_id.startswith("REV_"):
        return "FP"

    # SwissProt background (non-SCOP) → IGNORE
    if target_id.startswith("SP_"):
        return "IGNORE"

    # Both are SCOPe domains
    q_info = domain_info.get(query_id)
    t_info = domain_info.get(target_id)

    if q_info is None or t_info is None:
        return "IGNORE"

    if q_info["family"] == t_info["family"]:
        return "TP"
    elif q_info["fold"] != t_info["fold"]:
        return "FP"
    else:
        # Same fold, different family (could be same or different superfamily)
        # Per MMseqs2 paper: "Other cases are ignored"
        return "IGNORE"


def compute_roc_n(ranked_hits: list[RankedHit], n: int, total_tp: int) -> float:
    """Compute ROCn: fraction of TPs ranked above the nth FP.

    Args:
        ranked_hits: Hits sorted by score descending (IGNORE already removed).
        n: Stop at the nth FP.
        total_tp: Total number of TPs in the database for this query.

    Returns:
        Fraction of TPs found before the nth FP (0.0 to 1.0).
    """
    if total_tp == 0:
        return 0.0

    tp_found = 0
    fp_found = 0
    for hit in ranked_hits:
        if hit.label == "FP":
            fp_found += 1
            if fp_found >= n:
                break
        elif hit.label == "TP":
            tp_found += 1

    return tp_found / total_tp


def compute_average_precision(
    ranked_hits: list[RankedHit], total_tp: int,
) -> float:
    """Compute Average Precision for a single query.

    AP = (1/total_relevant) * sum_{k: hit_k is TP} precision(k)
    where precision(k) = (TPs in top k) / k
    """
    if total_tp == 0:
        return 0.0

    tp_count = 0
    ap_sum = 0.0
    for rank, hit in enumerate(ranked_hits, 1):
        if hit.label == "TP":
            tp_count += 1
            ap_sum += tp_count / rank
        # FPs increase rank but not tp_count

    return ap_sum / total_tp


def compute_sensitivity_at_fdr(
    ranked_hits: list[RankedHit], total_tp: int, fdr_threshold: float = 0.01,
) -> float:
    """Compute sensitivity at a given FDR threshold.

    Walks through ranked hits, tracking TP and FP counts.
    Reports the deepest sensitivity where FDR <= fdr_threshold.
    FDR = FP / (TP + FP).
    """
    if total_tp == 0:
        return 0.0

    tp_count = 0
    fp_count = 0
    best_sensitivity = 0.0

    for hit in ranked_hits:
        if hit.label == "TP":
            tp_count += 1
        elif hit.label == "FP":
            fp_count += 1

        total = tp_count + fp_count
        if total > 0:
            fdr = fp_count / total
            if fdr <= fdr_threshold:
                best_sensitivity = tp_count / total_tp

    return best_sensitivity


def evaluate_tool(
    tool_name: str,
    raw_hits: list[tuple[str, str, float]],
    metadata: dict,
) -> dict:
    """Evaluate a tool's search results.

    Args:
        tool_name: Name of the tool.
        raw_hits: [(query_id, target_id, score), ...] Higher score = more confident.
        metadata: Benchmark metadata from Phase 1.

    Returns:
        Dict with per-query and aggregate metrics.
    """
    domain_info = metadata["domain_info"]
    family_sizes = metadata["family_sizes_in_db"]
    query_sids = set(metadata["query_sids"])

    # Group hits by query, sort by score descending
    hits_by_query = defaultdict(list)
    for qid, tid, score in raw_hits:
        hits_by_query[qid].append((tid, score))

    per_query = {}
    roc1_values = []
    roc5_values = []
    map_values = []
    sens_1pct_values = []

    for qid in sorted(query_sids):
        # Total TPs for this query = same-family members in DB minus self
        q_info = domain_info.get(qid)
        if q_info is None:
            continue
        fam_key = str(q_info["family"])
        total_tp = family_sizes.get(fam_key, 1) - 1  # subtract self
        if total_tp <= 0:
            continue

        # Get ranked hits, classify, remove IGNORE
        query_hits = hits_by_query.get(qid, [])
        query_hits.sort(key=lambda x: -x[1])  # sort by score descending

        ranked = []
        for tid, score in query_hits:
            label = classify_hit(qid, tid, domain_info)
            if label != "IGNORE":
                ranked.append(RankedHit(target_id=tid, score=score, label=label))

        # Compute metrics
        roc1 = compute_roc_n(ranked, 1, total_tp)
        roc5 = compute_roc_n(ranked, 5, total_tp)
        ap = compute_average_precision(ranked, total_tp)
        sens = compute_sensitivity_at_fdr(ranked, total_tp, 0.01)

        roc1_values.append(roc1)
        roc5_values.append(roc5)
        map_values.append(ap)
        sens_1pct_values.append(sens)

        per_query[qid] = {
            "total_tp": total_tp,
            "hits_reported": len(query_hits),
            "hits_evaluated": len(ranked),
            "tp_found": sum(1 for h in ranked if h.label == "TP"),
            "fp_found": sum(1 for h in ranked if h.label == "FP"),
            "roc1": roc1,
            "roc5": roc5,
            "ap": ap,
            "sensitivity_at_1pct_fdr": sens,
        }

    # Aggregate
    n = len(roc1_values)
    aggregate = {
        "tool": tool_name,
        "n_queries_evaluated": n,
        "mean_roc1": float(np.mean(roc1_values)) if n > 0 else 0.0,
        "std_roc1": float(np.std(roc1_values)) if n > 0 else 0.0,
        "median_roc1": float(np.median(roc1_values)) if n > 0 else 0.0,
        "mean_roc5": float(np.mean(roc5_values)) if n > 0 else 0.0,
        "std_roc5": float(np.std(roc5_values)) if n > 0 else 0.0,
        "mean_map": float(np.mean(map_values)) if n > 0 else 0.0,
        "std_map": float(np.std(map_values)) if n > 0 else 0.0,
        "mean_sensitivity_at_1pct_fdr": (
            float(np.mean(sens_1pct_values)) if n > 0 else 0.0
        ),
        "std_sensitivity_at_1pct_fdr": (
            float(np.std(sens_1pct_values)) if n > 0 else 0.0
        ),
        "total_hits": sum(
            pq["hits_reported"] for pq in per_query.values()
        ),
        "total_tp_found": sum(
            pq["tp_found"] for pq in per_query.values()
        ),
        "total_fp_found": sum(
            pq["fp_found"] for pq in per_query.values()
        ),
        # ROC1 cumulative distribution (for plotting like MMseqs2 Fig 2a)
        "roc1_distribution": {
            "p10": float(np.percentile(roc1_values, 10)) if n > 0 else 0.0,
            "p25": float(np.percentile(roc1_values, 25)) if n > 0 else 0.0,
            "p50": float(np.percentile(roc1_values, 50)) if n > 0 else 0.0,
            "p75": float(np.percentile(roc1_values, 75)) if n > 0 else 0.0,
            "p90": float(np.percentile(roc1_values, 90)) if n > 0 else 0.0,
        },
    }

    return {"aggregate": aggregate, "per_query": per_query}


def evaluate_all(args, all_hits: dict[str, list], metadata: dict):
    """Evaluate all tools and save results."""
    out = Path(args.output_dir)

    log.info("Phase 4: EVALUATE")

    # ── Load search timing for each tool ──
    timings = {}
    for tool_name in ["clustkit", "diamond", "mmseqs2", "blast"]:
        timing_path = out / f"{tool_name}_timing.json"
        if timing_path.exists():
            with open(timing_path) as f:
                timings[tool_name] = json.load(f)

    all_results = {}
    for tool_name, hits in all_hits.items():
        log.info(f"  Evaluating {tool_name} ({len(hits)} hits)...")
        result = evaluate_tool(tool_name, hits, metadata)

        # Merge search time into aggregate metrics
        if tool_name in timings:
            result["aggregate"]["search_time_seconds"] = timings[tool_name]["search_time"]
            # Include extra ClustKIT details if available
            if "num_candidates" in timings[tool_name]:
                result["aggregate"]["num_candidates"] = timings[tool_name]["num_candidates"]
            if "num_aligned" in timings[tool_name]:
                result["aggregate"]["num_aligned"] = timings[tool_name]["num_aligned"]

        all_results[tool_name] = result

        agg = result["aggregate"]
        search_t = agg.get("search_time_seconds", 0)
        log.info(
            f"    {tool_name}: ROC1={agg['mean_roc1']:.4f} "
            f"ROC5={agg['mean_roc5']:.4f} "
            f"MAP={agg['mean_map']:.4f} "
            f"Sens@1%FDR={agg['mean_sensitivity_at_1pct_fdr']:.4f} "
            f"Time={search_t:.1f}s "
            f"({agg['n_queries_evaluated']} queries, "
            f"{agg['total_tp_found']} TP, {agg['total_fp_found']} FP)"
        )

    # Save results
    results_path = out / "scop_search_results.json"
    summary = {
        tool: result["aggregate"] for tool, result in all_results.items()
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"  Summary saved to {results_path}")

    # Save full results (with per-query) separately
    full_path = out / "scop_search_results_full.json"
    with open(full_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"  Full results saved to {full_path}")

    # ── Print comparison table ──
    print("\n" + "=" * 110)
    print("SCOPe SEARCH BENCHMARK RESULTS (search time excludes database indexing)")
    print("=" * 110)
    print(
        f"{'Tool':12s} {'ROC1':>8s} {'ROC5':>8s} {'MAP':>8s} "
        f"{'S@1%FDR':>8s} {'TP':>8s} {'FP':>8s} {'Queries':>8s} "
        f"{'Search(s)':>10s} {'Seqs/s':>10s}"
    )
    print("-" * 110)
    nq = metadata["query_count"]
    for tool_name in ["clustkit", "diamond", "mmseqs2", "blast"]:
        if tool_name not in all_results:
            continue
        agg = all_results[tool_name]["aggregate"]
        search_t = agg.get("search_time_seconds", 0)
        throughput = nq / search_t if search_t > 0 else 0
        print(
            f"{tool_name:12s} "
            f"{agg['mean_roc1']:8.4f} "
            f"{agg['mean_roc5']:8.4f} "
            f"{agg['mean_map']:8.4f} "
            f"{agg['mean_sensitivity_at_1pct_fdr']:8.4f} "
            f"{agg['total_tp_found']:8d} "
            f"{agg['total_fp_found']:8d} "
            f"{agg['n_queries_evaluated']:8d} "
            f"{search_t:10.1f} "
            f"{throughput:10.1f}"
        )
    print("=" * 110)
    print()

    return all_results


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SCOPe-based protein sequence search benchmark"
    )
    parser.add_argument(
        "--scope-fasta", required=True,
        help="Path to SCOPe ASTRAL domain sequences FASTA",
    )
    parser.add_argument(
        "--scope-cla", required=True,
        help="Path to SCOPe dir.cla classification file",
    )
    parser.add_argument(
        "--swissprot-fasta", default=None,
        help="Optional SwissProt FASTA for background sequences",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/data/scop_search_results",
        help="Output directory",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--max-hits", type=int, default=500)
    parser.add_argument(
        "--max-queries-per-family", type=int, default=5,
        help="Max query domains per SCOP family",
    )
    parser.add_argument(
        "--max-queries", type=int, default=10000,
        help="Max total query domains",
    )
    parser.add_argument(
        "--tools", default="clustkit,mmseqs2,diamond,blast",
        help="Comma-separated list of tools to benchmark",
    )
    parser.add_argument(
        "--phases", default="prepare,index,search,evaluate",
        help="Comma-separated phases to run (prepare,index,search,evaluate)",
    )

    args = parser.parse_args()
    phases = args.phases.split(",")
    out = Path(args.output_dir)

    # ── Phase 1: Prepare ──
    if "prepare" in phases:
        metadata = prepare(args)
    else:
        with open(out / "metadata.json") as f:
            metadata = json.load(f)
        log.info(f"Loaded existing metadata ({metadata['query_count']} queries)")

    # ── Phase 2: Index ──
    if "index" in phases:
        build_indices(args, metadata)

    # ── Phase 3: Search ──
    if "search" in phases:
        all_hits = run_searches(args, metadata)

        # Save raw hits for re-evaluation without re-searching
        for tool_name, hits in all_hits.items():
            hits_path = out / f"{tool_name}_raw_hits.json"
            with open(hits_path, "w") as f:
                json.dump(hits, f)
    else:
        # Load saved raw hits
        all_hits = {}
        tools = args.tools.split(",")
        for tool_name in tools:
            hits_path = out / f"{tool_name}_raw_hits.json"
            if hits_path.exists():
                with open(hits_path) as f:
                    all_hits[tool_name] = json.load(f)
                log.info(
                    f"Loaded {len(all_hits[tool_name])} saved hits for {tool_name}"
                )

    # ── Phase 4: Evaluate ──
    if "evaluate" in phases:
        evaluate_all(args, all_hits, metadata)


if __name__ == "__main__":
    main()
