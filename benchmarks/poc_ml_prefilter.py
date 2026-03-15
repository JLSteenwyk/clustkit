#!/usr/bin/env python3
"""POC: ML prefilter for SW alignment acceleration.

Trains a model to predict SW alignment scores from cheap features,
then evaluates how many candidate pairs can be skipped before alignment
with minimal ROC1 loss.

Features (all much cheaper than SW alignment):
  - query_length, target_length, length_ratio, length_diff
  - shared_k3_count: exact k=3 k-mers shared (standard alphabet)
  - composition_cosine: AA frequency cosine similarity

Pipeline:
  1. Run candidate generation (Phase A + reduced alphabet)
  2. Compute features for all candidate pairs (~seconds)
  3. Run SW alignment to get ground-truth scores (~minutes)
  4. Train sklearn HistGradientBoostingRegressor on (features → SW score)
  5. Evaluate: at various prefilter thresholds, measure filtering rate vs ROC1
"""

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from numba import njit, prange, int32, int64, float32

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.benchmark_scop_search import (
    classify_hit, compute_roc_n, read_fasta, write_fasta, RankedHit,
)

import numba
numba.set_num_threads(8)

from clustkit.database import load_database
from clustkit.io import read_sequences
from clustkit.kmer_index import (
    search_kmer_index, build_kmer_index, compute_freq_threshold,
    _batch_score_queries, REDUCED_ALPHA, REDUCED_ALPHA_SIZE,
    _remap_flat, _batch_score_queries_spaced, build_kmer_index_spaced,
)
from clustkit.pairwise import _batch_sw_compact_scored, BLOSUM62
from clustkit.search import _merge_sequences_for_alignment, _remap_pairs_to_merged


# ──────────────────────────────────────────────────────────────────────
# Feature computation (Numba-accelerated)
# ──────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _build_query_kmer_sets(q_flat, q_offsets, q_lengths, k, alpha_size):
    """Pre-compute k-mer presence sets for all queries."""
    nq = len(q_lengths)
    num_possible = int64(1)
    for _ in range(k):
        num_possible *= int64(alpha_size)
    kmer_sets = np.zeros((nq, num_possible), dtype=np.bool_)
    for qi in range(nq):
        start = int64(q_offsets[qi])
        length = int32(q_lengths[qi])
        for pos in range(length - k + 1):
            kmer_val = int64(0)
            valid = True
            for j in range(k):
                r = q_flat[start + pos + j]
                if r >= alpha_size:
                    valid = False
                    break
                kmer_val = kmer_val * int64(alpha_size) + int64(r)
            if valid:
                kmer_sets[qi, kmer_val] = True
    return kmer_sets


@njit(parallel=True, cache=True)
def _compute_features(
    pairs, q_kmer_sets, q_flat, q_offsets, q_lengths,
    t_flat, t_offsets, t_lengths, k, alpha_size,
):
    """Compute features for all candidate pairs.

    Returns: (n_pairs, 6) float32 array with columns:
      [query_len, target_len, length_ratio, length_diff,
       shared_k3_count, composition_cosine]
    """
    m = len(pairs)
    features = np.empty((m, 6), dtype=np.float32)

    for idx in prange(m):
        qi = pairs[idx, 0]
        ti = pairs[idx, 1]

        q_len = float32(q_lengths[qi])
        t_len = float32(t_lengths[ti])
        shorter = min(q_len, t_len)
        longer = max(q_len, t_len)

        features[idx, 0] = q_len
        features[idx, 1] = t_len
        features[idx, 2] = shorter / longer if longer > 0 else float32(0)
        features[idx, 3] = abs(q_len - t_len)

        # Shared k-mer count
        t_start = int64(t_offsets[ti])
        t_length = int32(t_lengths[ti])
        shared = int32(0)
        for pos in range(t_length - k + 1):
            kmer_val = int64(0)
            valid = True
            for j in range(k):
                r = t_flat[t_start + pos + j]
                if r >= alpha_size:
                    valid = False
                    break
                kmer_val = kmer_val * int64(alpha_size) + int64(r)
            if valid and q_kmer_sets[qi, kmer_val]:
                shared += int32(1)
        features[idx, 4] = float32(shared)

        # Composition cosine similarity
        q_freq = np.zeros(20, dtype=np.float32)
        q_start = int64(q_offsets[qi])
        for pos in range(int32(q_lengths[qi])):
            aa = q_flat[q_start + pos]
            if aa < 20:
                q_freq[aa] += float32(1.0)

        t_freq = np.zeros(20, dtype=np.float32)
        for pos in range(t_length):
            aa = t_flat[t_start + pos]
            if aa < 20:
                t_freq[aa] += float32(1.0)

        dot = float32(0)
        q_norm = float32(0)
        t_norm = float32(0)
        for a in range(20):
            dot += q_freq[a] * t_freq[a]
            q_norm += q_freq[a] * q_freq[a]
            t_norm += t_freq[a] * t_freq[a]
        if q_norm > 0 and t_norm > 0:
            features[idx, 5] = dot / (q_norm ** float32(0.5) * t_norm ** float32(0.5))
        else:
            features[idx, 5] = float32(0)

    return features


# ──────────────────────────────────────────────────────────────────────
# ROC1 evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_roc1(hits_by_query, metadata):
    domain_info = metadata["domain_info"]
    family_sizes = metadata["family_sizes_in_db"]
    query_sids = set(metadata["query_sids"])
    roc1_values = []
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
    return float(np.mean(roc1_values)) if roc1_values else 0.0


def collect_hits(candidate_pairs, sw_scores, query_ds, db_ds, nq, top_k=500):
    """Collect top-k hits per query from candidate pairs + scores."""
    hits_by_query = defaultdict(list)
    for idx in range(len(candidate_pairs)):
        qi = int(candidate_pairs[idx, 0])
        ti = int(candidate_pairs[idx, 1])
        score = float(sw_scores[idx])
        qid = query_ds.ids[qi]
        tid = db_ds.ids[ti]
        hits_by_query[qid].append((tid, score))
    # Keep top-k per query
    for qid in hits_by_query:
        hits_by_query[qid] = sorted(hits_by_query[qid], key=lambda x: -x[1])[:top_k]
    return hits_by_query


# ──────────────────────────────────────────────────────────────────────
# Main POC
# ──────────────────────────────────────────────────────────────────────

def main():
    scop_dir = Path("benchmarks/data/scop_search_results")
    out_dir = Path("benchmarks/data/speed_sensitivity_results")

    with open(scop_dir / "metadata.json") as f:
        full_metadata = json.load(f)

    all_query_sids = full_metadata["query_sids"]
    random.seed(42)
    query_sids = sorted(random.sample(all_query_sids, min(2000, len(all_query_sids))))
    all_seqs = dict(read_fasta(scop_dir / "queries.fasta"))
    sub_seqs = [(sid, all_seqs[sid]) for sid in query_sids if sid in all_seqs]
    query_fasta = str(out_dir / "queries_subset.fasta")
    write_fasta(sub_seqs, query_fasta)
    metadata = dict(full_metadata)
    metadata["query_sids"] = query_sids

    print("Loading database...", flush=True)
    db_index = load_database(out_dir / "clustkit_db")
    query_ds = read_sequences(query_fasta, "protein")
    db_ds = db_index.dataset
    nq = query_ds.num_sequences
    nd = db_ds.num_sequences
    print(f"Loaded {nq} queries, {nd} database sequences\n", flush=True)

    # ─── Step 1: Generate candidates (dual k=5 + 2 spaced seeds) ────
    print("=" * 100)
    print("Step 1: Candidate generation")
    print("=" * 100, flush=True)

    mc = 8000
    topk = 200000
    freq_pctl = 99.5

    q_flat = query_ds.flat_sequences
    q_off = query_ds.offsets.astype(np.int64)
    q_lens = query_ds.lengths.astype(np.int32)

    # Standard k=3 Phase A
    k = int32(db_index.params.get("kmer_index_k", db_index.params["kmer_size"]))
    freq_thresh = compute_freq_threshold(db_index.kmer_freqs, nd, freq_pctl)

    out_targets = np.empty((nq, mc), dtype=np.int32)
    out_counts = np.zeros(nq, dtype=np.int32)

    t0 = time.perf_counter()
    _batch_score_queries(
        q_flat, q_off, q_lens,
        k, int32(20),
        db_index.kmer_offsets, db_index.kmer_entries, db_index.kmer_freqs,
        freq_thresh, int32(nd), int32(2), int32(2), int32(10),
        int32(mc), int32(topk),
        out_targets, out_counts,
    )
    print(f"  Standard k=3: {int(out_counts.sum())} candidates ({time.perf_counter()-t0:.1f}s)", flush=True)

    # Flatten
    total_std = int(out_counts.sum())
    candidate_pairs = np.empty((total_std, 2), dtype=np.int32)
    pos = 0
    for qi in range(nq):
        nc = int(out_counts[qi])
        if nc > 0:
            candidate_pairs[pos:pos+nc, 0] = qi
            candidate_pairs[pos:pos+nc, 1] = out_targets[qi, :nc]
            pos += nc

    # Reduced k=5
    t0 = time.perf_counter()
    red_q_flat = _remap_flat(q_flat, REDUCED_ALPHA, len(q_flat))
    db_flat = db_ds.flat_sequences
    red_db_flat = _remap_flat(db_flat, REDUCED_ALPHA, len(db_flat))

    red_off, red_ent, red_freq = build_kmer_index(
        red_db_flat, db_ds.offsets, db_ds.lengths, 5, "protein",
        alpha_size=REDUCED_ALPHA_SIZE,
    )
    red_freq_thresh = compute_freq_threshold(red_freq, nd, freq_pctl)
    red_out_t = np.empty((nq, mc), dtype=np.int32)
    red_out_c = np.zeros(nq, dtype=np.int32)
    _batch_score_queries(
        red_q_flat, q_off, q_lens,
        int32(5), int32(REDUCED_ALPHA_SIZE),
        red_off, red_ent, red_freq, red_freq_thresh,
        int32(nd), int32(2), int32(2), int32(10),
        int32(mc), int32(topk),
        red_out_t, red_out_c,
    )
    red_total = int(red_out_c.sum())
    print(f"  Reduced k=5: {red_total} candidates ({time.perf_counter()-t0:.1f}s)", flush=True)

    # Spaced seed 11011
    t0 = time.perf_counter()
    sp_off, sp_ent, sp_freq, sp_seed_off, sp_w, sp_span = build_kmer_index_spaced(
        red_db_flat, db_ds.offsets, db_ds.lengths, "11011", "protein",
        alpha_size=REDUCED_ALPHA_SIZE,
    )
    sp_freq_thresh = compute_freq_threshold(sp_freq, nd, freq_pctl)
    sp_out_t = np.empty((nq, mc), dtype=np.int32)
    sp_out_c = np.zeros(nq, dtype=np.int32)
    _batch_score_queries_spaced(
        red_q_flat, q_off, q_lens,
        sp_seed_off, int32(sp_w), int32(sp_span), int32(REDUCED_ALPHA_SIZE),
        sp_off, sp_ent, sp_freq, sp_freq_thresh,
        int32(nd), int32(2), int32(2), int32(10),
        int32(mc), int32(topk),
        sp_out_t, sp_out_c,
    )
    sp_total = int(sp_out_c.sum())
    print(f"  Spaced 11011: {sp_total} candidates ({time.perf_counter()-t0:.1f}s)", flush=True)

    # Union all candidates
    all_packed = [candidate_pairs[:, 0].astype(np.int64) * nd + candidate_pairs[:, 1].astype(np.int64)]
    for out_t, out_c in [(red_out_t, red_out_c), (sp_out_t, sp_out_c)]:
        tot = int(out_c.sum())
        if tot > 0:
            pairs = np.empty((tot, 2), dtype=np.int32)
            p = 0
            for qi in range(nq):
                nc = int(out_c[qi])
                if nc > 0:
                    pairs[p:p+nc, 0] = qi
                    pairs[p:p+nc, 1] = out_t[qi, :nc]
                    p += nc
            all_packed.append(pairs[:, 0].astype(np.int64) * nd + pairs[:, 1].astype(np.int64))

    union = np.unique(np.concatenate(all_packed))
    candidate_pairs = np.empty((len(union), 2), dtype=np.int32)
    candidate_pairs[:, 0] = (union // nd).astype(np.int32)
    candidate_pairs[:, 1] = (union % nd).astype(np.int32)
    print(f"\n  Total after union: {len(candidate_pairs)} candidate pairs", flush=True)

    # ─── Step 2: Compute features ───────────────────────────────────
    print("\n" + "=" * 100)
    print("Step 2: Feature computation")
    print("=" * 100, flush=True)

    t0 = time.perf_counter()
    q_kmer_sets = _build_query_kmer_sets(q_flat, q_off, q_lens, int32(3), int32(20))
    print(f"  Query k-mer sets built ({time.perf_counter()-t0:.1f}s)", flush=True)

    t0 = time.perf_counter()
    features = _compute_features(
        candidate_pairs, q_kmer_sets,
        q_flat, q_off, q_lens,
        db_flat, db_ds.offsets.astype(np.int64), db_ds.lengths.astype(np.int32),
        int32(3), int32(20),
    )
    feat_time = time.perf_counter() - t0
    print(f"  Features computed: {features.shape} ({feat_time:.1f}s)", flush=True)
    print(f"  Feature names: [query_len, target_len, length_ratio, length_diff, "
          f"shared_k3, comp_cosine]", flush=True)

    # ─── Step 3: Run SW alignment (ground truth) ────────────────────
    print("\n" + "=" * 100)
    print("Step 3: SW alignment (ground truth labels)")
    print("=" * 100, flush=True)

    merged = _merge_sequences_for_alignment(query_ds, db_ds)
    merged_pairs = _remap_pairs_to_merged(candidate_pairs, merged["nq"])

    sort_key = np.minimum(merged_pairs[:, 0], merged_pairs[:, 1])
    sort_order = np.argsort(sort_key, kind="mergesort")
    merged_pairs = merged_pairs[sort_order]
    candidate_pairs = candidate_pairs[sort_order]
    features = features[sort_order]

    all_lengths = np.concatenate([query_ds.lengths, db_ds.lengths])
    band_width = max(20, int(np.percentile(all_lengths, 95) * 0.3))

    t0 = time.perf_counter()
    sims, raw_scores, aln_mask = _batch_sw_compact_scored(
        merged_pairs,
        merged["flat_sequences"],
        merged["offsets"],
        merged["lengths"],
        np.float32(0.1),
        int32(band_width),
        BLOSUM62,
    )
    aln_time = time.perf_counter() - t0
    print(f"  SW alignment: {len(candidate_pairs)} pairs in {aln_time:.1f}s", flush=True)
    print(f"  Score range: [{raw_scores.min()}, {raw_scores.max()}], "
          f"median={np.median(raw_scores):.0f}", flush=True)

    # ─── Step 4: Train ML model ─────────────────────────────────────
    print("\n" + "=" * 100)
    print("Step 4: Train ML models")
    print("=" * 100, flush=True)

    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    X = features
    y_score = raw_scores.astype(np.float32)
    y_binary = (raw_scores > np.median(raw_scores)).astype(np.int32)

    # Train/test split (80/20 by query to avoid leakage)
    query_ids = candidate_pairs[:, 0]
    unique_queries = np.unique(query_ids)
    np.random.seed(42)
    np.random.shuffle(unique_queries)
    split = int(len(unique_queries) * 0.8)
    train_queries = set(unique_queries[:split].tolist())
    train_mask = np.array([int(qi) in train_queries for qi in query_ids])
    test_mask = ~train_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y_score[train_mask], y_score[test_mask]
    y_bin_train, y_bin_test = y_binary[train_mask], y_binary[test_mask]

    print(f"  Train: {len(X_train)} pairs, Test: {len(X_test)} pairs", flush=True)

    # Regression model
    t0 = time.perf_counter()
    reg = HistGradientBoostingRegressor(
        max_iter=200, max_depth=6, learning_rate=0.1, random_state=42,
    )
    reg.fit(X_train, y_train)
    reg_time = time.perf_counter() - t0
    y_pred = reg.predict(X_test)
    corr = np.corrcoef(y_test, y_pred)[0, 1]
    print(f"\n  Regression model (HistGBR):", flush=True)
    print(f"    Train time: {reg_time:.1f}s", flush=True)
    print(f"    Pearson r: {corr:.4f}", flush=True)
    print(f"    RMSE: {np.sqrt(np.mean((y_test - y_pred)**2)):.1f}", flush=True)

    # Binary classifier
    t0 = time.perf_counter()
    clf = HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.1, random_state=42,
    )
    clf.fit(X_train, y_bin_train)
    clf_time = time.perf_counter() - t0
    y_bin_pred = clf.predict(X_test)
    acc = np.mean(y_bin_pred == y_bin_test)
    print(f"\n  Binary classifier (above-median score):", flush=True)
    print(f"    Train time: {clf_time:.1f}s", flush=True)
    print(f"    Accuracy: {acc:.4f}", flush=True)

    # Feature importance
    feat_names = ["query_len", "target_len", "len_ratio", "len_diff",
                  "shared_k3", "comp_cosine"]
    try:
        importances = reg.feature_importances_
    except AttributeError:
        # Fallback: use permutation importance on a sample
        from sklearn.inspection import permutation_importance
        sample_n = min(50000, len(X_test))
        perm_result = permutation_importance(
            reg, X_test[:sample_n], y_test[:sample_n],
            n_repeats=3, random_state=42,
        )
        importances = perm_result.importances_mean
    print(f"\n  Feature importance (regression):", flush=True)
    for name, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
        print(f"    {name:15s}: {imp:.4f}", flush=True)

    # ─── Step 5: Evaluate prefilter ─────────────────────────────────
    print("\n" + "=" * 100)
    print("Step 5: Prefilter evaluation (simulated)")
    print("=" * 100, flush=True)

    # Predict scores for test set
    y_pred_all = reg.predict(X)  # predict for ALL pairs

    # Compute baseline ROC1 (no filtering)
    passing_mask = raw_scores > 0
    hits_all = collect_hits(
        candidate_pairs[passing_mask], raw_scores[passing_mask],
        query_ds, db_ds, nq,
    )
    baseline_roc1 = evaluate_roc1(hits_all, metadata)
    print(f"\n  Baseline ROC1 (no prefilter): {baseline_roc1:.4f}", flush=True)
    print(f"  Baseline aligned: {len(candidate_pairs)} pairs in {aln_time:.1f}s", flush=True)

    # Test prefilter at various thresholds
    print(f"\n  {'Threshold':>12s} {'Kept':>10s} {'Filtered':>10s} {'% Kept':>8s} "
          f"{'ROC1':>7s} {'ROC1 loss':>10s} {'Est time':>10s}", flush=True)
    print("  " + "-" * 80, flush=True)

    percentiles = [10, 20, 30, 40, 50, 60, 70, 80]
    for pct in percentiles:
        thresh = np.percentile(y_pred_all, pct)
        keep_mask = y_pred_all >= thresh
        n_kept = int(keep_mask.sum())
        n_filtered = len(candidate_pairs) - n_kept

        # Run ROC1 on kept pairs
        kept_pairs = candidate_pairs[keep_mask]
        kept_scores = raw_scores[keep_mask]
        kept_passing = kept_scores > 0
        if kept_passing.sum() > 0:
            hits_filtered = collect_hits(
                kept_pairs[kept_passing], kept_scores[kept_passing],
                query_ds, db_ds, nq,
            )
            filtered_roc1 = evaluate_roc1(hits_filtered, metadata)
        else:
            filtered_roc1 = 0.0

        pct_kept = 100 * n_kept / len(candidate_pairs)
        roc1_loss = baseline_roc1 - filtered_roc1
        est_aln_time = aln_time * (n_kept / len(candidate_pairs))

        print(f"  {thresh:12.1f} {n_kept:10d} {n_filtered:10d} {pct_kept:7.1f}% "
              f"{filtered_roc1:7.4f} {roc1_loss:+10.4f} {est_aln_time:9.1f}s", flush=True)

    # Inference speed
    t0 = time.perf_counter()
    for _ in range(3):
        _ = reg.predict(X)
    inf_time = (time.perf_counter() - t0) / 3
    print(f"\n  ML inference time: {inf_time:.2f}s for {len(X)} pairs", flush=True)
    print(f"  Feature computation time: {feat_time:.1f}s", flush=True)
    print(f"  SW alignment time: {aln_time:.1f}s", flush=True)
    print(f"\n  Overhead of prefilter: {feat_time + inf_time:.1f}s "
          f"(vs {aln_time:.1f}s alignment)", flush=True)

    print(f"\n  Summary: feature computation ({feat_time:.1f}s) + ML inference ({inf_time:.1f}s) "
          f"= {feat_time + inf_time:.1f}s overhead", flush=True)
    print(f"  If prefilter removes 50% of pairs: save ~{aln_time*0.5:.0f}s alignment, "
          f"net saving ~{aln_time*0.5 - feat_time - inf_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
