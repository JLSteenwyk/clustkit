#!/usr/bin/env python3
"""Improved ML two-tier: Phase A/B scores + better models (XGBoost, LightGBM).

Key improvement: expose actual Phase A/B scores per-pair per-index as features.
Previously we only had n_indices (binary count). Now we have the actual scores
that the pipeline computed, which directly predict alignment quality.

Models tested:
  - RF (100 trees, d=12) — previous baseline
  - RF (300 trees, d=20) — deeper architecture
  - XGBoost
  - LightGBM
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
    build_kmer_index, compute_freq_threshold,
    _batch_score_queries_with_scores, _batch_score_queries_spaced_with_scores,
    build_kmer_index_spaced,
    REDUCED_ALPHA, REDUCED_ALPHA_SIZE,
    _remap_flat, _build_query_kmer_sets,
)
from clustkit.pairwise import _batch_sw_compact_scored, BLOSUM62
from clustkit.search import (
    _merge_sequences_for_alignment, _remap_pairs_to_merged, _collect_top_k_hits,
)


@njit(parallel=True, cache=True)
def _compute_composition_cosine(pairs, q_flat, q_offsets, q_lengths,
                                t_flat, t_offsets, t_lengths):
    """Compute AA composition cosine similarity for each pair."""
    m = len(pairs)
    result = np.empty(m, dtype=np.float32)
    for idx in prange(m):
        qi = pairs[idx, 0]
        ti = pairs[idx, 1]
        q_freq = np.zeros(20, dtype=np.float32)
        q_start = int64(q_offsets[qi])
        for pos in range(int32(q_lengths[qi])):
            aa = q_flat[q_start + pos]
            if aa < 20:
                q_freq[aa] += float32(1.0)
        t_freq = np.zeros(20, dtype=np.float32)
        t_start = int64(t_offsets[ti])
        for pos in range(int32(t_lengths[ti])):
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
            result[idx] = dot / (q_norm ** float32(0.5) * t_norm ** float32(0.5))
        else:
            result[idx] = float32(0)
    return result


def evaluate_roc1(results_hits, metadata):
    domain_info = metadata["domain_info"]
    family_sizes = metadata["family_sizes_in_db"]
    query_sids = set(metadata["query_sids"])
    hits_by_query = defaultdict(list)
    for qhits in results_hits:
        for h in qhits:
            hits_by_query[h.query_id].append((h.target_id, h.score if h.score != 0 else h.identity))
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

    q_flat = query_ds.flat_sequences
    q_off = query_ds.offsets.astype(np.int64)
    q_lens = query_ds.lengths.astype(np.int32)
    db_flat = db_ds.flat_sequences
    t_off = db_ds.offsets.astype(np.int64)
    t_lens = db_ds.lengths.astype(np.int32)
    mc = 8000
    topk = 200000

    print(f"Loaded {nq} queries, {nd} database sequences\n", flush=True)

    # ── Step 1: Generate candidates WITH scores from each index ──────
    print("=" * 100)
    print("Step 1: Candidate generation with per-index Phase A/B scores")
    print("=" * 100, flush=True)

    freq_thresh = compute_freq_threshold(db_index.kmer_freqs, nd, 99.5)
    k = int32(db_index.params.get("kmer_index_k", db_index.params["kmer_size"]))

    def run_index_with_scores(label, is_spaced=False, **kwargs):
        t0 = time.perf_counter()
        out_t = np.empty((nq, mc), dtype=np.int32)
        out_c = np.zeros(nq, dtype=np.int32)
        out_s = np.zeros((nq, mc), dtype=np.int32)
        if is_spaced:
            _batch_score_queries_spaced_with_scores(**kwargs, out_targets=out_t,
                                                     out_counts=out_c, out_scores=out_s)
        else:
            _batch_score_queries_with_scores(**kwargs, out_targets=out_t,
                                             out_counts=out_c, out_scores=out_s)
        total = int(out_c.sum())
        # Flatten into pairs + scores
        pairs = np.empty((total, 2), dtype=np.int32)
        scores = np.empty(total, dtype=np.int32)
        p = 0
        for qi in range(nq):
            nc = int(out_c[qi])
            if nc > 0:
                pairs[p:p+nc, 0] = qi
                pairs[p:p+nc, 1] = out_t[qi, :nc]
                scores[p:p+nc] = out_s[qi, :nc]
                p += nc
        packed = pairs[:, 0].astype(np.int64) * nd + pairs[:, 1].astype(np.int64)
        print(f"  {label}: {total} candidates ({time.perf_counter()-t0:.0f}s)", flush=True)
        return packed, pairs, scores

    # Standard k=3
    p_std, pairs_std, scores_std = run_index_with_scores("Standard k=3",
        q_flat=q_flat, q_offsets=q_off, q_lengths=q_lens,
        k=k, alpha_size=int32(20),
        kmer_offsets=db_index.kmer_offsets, kmer_entries=db_index.kmer_entries,
        kmer_freqs=db_index.kmer_freqs, freq_thresh=freq_thresh,
        num_db=int32(nd), min_total_hits=int32(2), min_diag_hits=int32(2),
        diag_bin_width=int32(10), max_cands=int32(mc), phase_a_topk=int32(topk))

    # Reduced k=4
    red_q_flat = _remap_flat(q_flat, REDUCED_ALPHA, len(q_flat))
    red_db_flat = _remap_flat(db_flat, REDUCED_ALPHA, len(db_flat))

    red4_off, red4_ent, red4_freq = build_kmer_index(
        red_db_flat, db_ds.offsets, db_ds.lengths, 4, "protein", alpha_size=REDUCED_ALPHA_SIZE)
    red4_ft = compute_freq_threshold(red4_freq, nd, 99.5)
    p_red4, pairs_red4, scores_red4 = run_index_with_scores("Reduced k=4",
        q_flat=red_q_flat, q_offsets=q_off, q_lengths=q_lens,
        k=int32(4), alpha_size=int32(REDUCED_ALPHA_SIZE),
        kmer_offsets=red4_off, kmer_entries=red4_ent, kmer_freqs=red4_freq,
        freq_thresh=red4_ft, num_db=int32(nd), min_total_hits=int32(2),
        min_diag_hits=int32(2), diag_bin_width=int32(10),
        max_cands=int32(mc), phase_a_topk=int32(topk))

    # Reduced k=5
    red5_off, red5_ent, red5_freq = build_kmer_index(
        red_db_flat, db_ds.offsets, db_ds.lengths, 5, "protein", alpha_size=REDUCED_ALPHA_SIZE)
    red5_ft = compute_freq_threshold(red5_freq, nd, 99.5)
    p_red5, pairs_red5, scores_red5 = run_index_with_scores("Reduced k=5",
        q_flat=red_q_flat, q_offsets=q_off, q_lengths=q_lens,
        k=int32(5), alpha_size=int32(REDUCED_ALPHA_SIZE),
        kmer_offsets=red5_off, kmer_entries=red5_ent, kmer_freqs=red5_freq,
        freq_thresh=red5_ft, num_db=int32(nd), min_total_hits=int32(2),
        min_diag_hits=int32(2), diag_bin_width=int32(10),
        max_cands=int32(mc), phase_a_topk=int32(topk))

    # Spaced seeds
    sp1_data = build_kmer_index_spaced(
        red_db_flat, db_ds.offsets, db_ds.lengths, "11011", "protein", alpha_size=REDUCED_ALPHA_SIZE)
    sp1_off, sp1_ent, sp1_freq, sp1_so, sp1_w, sp1_span = sp1_data
    sp1_ft = compute_freq_threshold(sp1_freq, nd, 99.5)
    p_sp1, pairs_sp1, scores_sp1 = run_index_with_scores("Spaced 11011", is_spaced=True,
        q_flat=red_q_flat, q_offsets=q_off, q_lengths=q_lens,
        seed_offsets=sp1_so, weight=int32(sp1_w), span=int32(sp1_span),
        alpha_size=int32(REDUCED_ALPHA_SIZE),
        kmer_offsets=sp1_off, kmer_entries=sp1_ent, kmer_freqs=sp1_freq,
        freq_thresh=sp1_ft, num_db=int32(nd), min_total_hits=int32(2),
        min_diag_hits=int32(2), diag_bin_width=int32(10),
        max_cands=int32(mc), phase_a_topk=int32(topk))

    sp2_data = build_kmer_index_spaced(
        red_db_flat, db_ds.offsets, db_ds.lengths, "110011", "protein", alpha_size=REDUCED_ALPHA_SIZE)
    sp2_off, sp2_ent, sp2_freq, sp2_so, sp2_w, sp2_span = sp2_data
    sp2_ft = compute_freq_threshold(sp2_freq, nd, 99.5)
    p_sp2, pairs_sp2, scores_sp2 = run_index_with_scores("Spaced 110011", is_spaced=True,
        q_flat=red_q_flat, q_offsets=q_off, q_lengths=q_lens,
        seed_offsets=sp2_so, weight=int32(sp2_w), span=int32(sp2_span),
        alpha_size=int32(REDUCED_ALPHA_SIZE),
        kmer_offsets=sp2_off, kmer_entries=sp2_ent, kmer_freqs=sp2_freq,
        freq_thresh=sp2_ft, num_db=int32(nd), min_total_hits=int32(2),
        min_diag_hits=int32(2), diag_bin_width=int32(10),
        max_cands=int32(mc), phase_a_topk=int32(topk))

    # ── Build per-pair score lookup for each index ───────────────────
    print("\n  Building per-pair score maps...", flush=True)
    index_data = [
        ("std_k3", p_std, pairs_std, scores_std),
        ("red_k4", p_red4, pairs_red4, scores_red4),
        ("red_k5", p_red5, pairs_red5, scores_red5),
        ("sp_11011", p_sp1, pairs_sp1, scores_sp1),
        ("sp_110011", p_sp2, pairs_sp2, scores_sp2),
    ]

    # Union all (d[1] is already packed int64)
    all_packed = np.concatenate([d[1] for d in index_data])
    unique_packed, inverse = np.unique(all_packed, return_inverse=True)

    candidate_pairs = np.empty((len(unique_packed), 2), dtype=np.int32)
    candidate_pairs[:, 0] = (unique_packed // nd).astype(np.int32)
    candidate_pairs[:, 1] = (unique_packed % nd).astype(np.int32)
    n_total = len(candidate_pairs)
    print(f"  Total unique: {n_total}", flush=True)

    # For each unique pair, look up its score from each index (0 if not found)
    # Build hash maps: packed_id -> score for each index
    per_index_scores = np.zeros((n_total, 5), dtype=np.float32)
    offset = 0
    for idx_i, (name, packed, pairs, scores) in enumerate(index_data):
        packed_vals = pairs[:, 0].astype(np.int64) * nd + pairs[:, 1].astype(np.int64)
        # Map packed -> position in unique_packed via searchsorted
        positions = np.searchsorted(unique_packed, packed_vals)
        per_index_scores[positions, idx_i] = scores.astype(np.float32)
        print(f"    {name}: mapped {len(scores)} scores", flush=True)

    # Count indices per pair
    n_indices = (per_index_scores > 0).sum(axis=1).astype(np.float32)

    # ── Step 2: Compute additional features ──────────────────────────
    print("\n" + "=" * 100)
    print("Step 2: Feature computation")
    print("=" * 100, flush=True)

    t0 = time.perf_counter()
    q_lens_f = q_lens[candidate_pairs[:, 0]].astype(np.float32)
    t_lens_f = t_lens[candidate_pairs[:, 1]].astype(np.float32)
    shorter = np.minimum(q_lens_f, t_lens_f)
    longer = np.maximum(q_lens_f, t_lens_f)
    len_ratio = np.where(longer > 0, shorter / longer, 0).astype(np.float32)
    len_diff = np.abs(q_lens_f - t_lens_f)

    # Composition cosine
    comp_cos = _compute_composition_cosine(
        candidate_pairs, q_flat, q_off, q_lens,
        db_flat, t_off, t_lens,
    )

    # Max and sum of per-index scores
    max_score = per_index_scores.max(axis=1)
    sum_score = per_index_scores.sum(axis=1)

    # Build feature matrix: 13 features
    features = np.column_stack([
        per_index_scores,       # 5: score from each index
        n_indices,              # 1: number of indices that found this pair
        max_score,              # 1: max score across indices
        sum_score,              # 1: sum of scores across indices
        len_ratio,              # 1: shorter/longer
        len_diff,               # 1: abs length difference
        q_lens_f,               # 1: query length
        t_lens_f,               # 1: target length
        comp_cos,               # 1: composition cosine
    ])
    feat_time = time.perf_counter() - t0
    print(f"  Features: {features.shape} ({feat_time:.1f}s)", flush=True)

    feat_names = ["score_std_k3", "score_red_k4", "score_red_k5",
                  "score_sp_11011", "score_sp_110011",
                  "n_indices", "max_score", "sum_score",
                  "len_ratio", "len_diff", "query_len", "target_len",
                  "comp_cosine"]

    # ── Step 3: SW alignment ─────────────────────────────────────────
    print("\n" + "=" * 100)
    print("Step 3: SW alignment (ground truth)")
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
        merged_pairs, merged["flat_sequences"], merged["offsets"],
        merged["lengths"], np.float32(0.1), int32(band_width), BLOSUM62)
    aln_time = time.perf_counter() - t0
    print(f"  Aligned {n_total} pairs in {aln_time:.1f}s", flush=True)

    # ── Step 4: Train models ─────────────────────────────────────────
    print("\n" + "=" * 100)
    print("Step 4: Train and compare models (13 features)")
    print("=" * 100, flush=True)

    query_ids = candidate_pairs[:, 0]
    unique_queries = np.unique(query_ids)
    np.random.seed(42)
    np.random.shuffle(unique_queries)
    split = int(len(unique_queries) * 0.8)
    train_queries = set(unique_queries[:split].tolist())
    train_mask = np.array([int(qi) in train_queries for qi in query_ids])

    X_train, X_test = features[train_mask], features[~train_mask]
    y_train, y_test = raw_scores[train_mask], raw_scores[~train_mask]
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}\n", flush=True)

    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    import lightgbm as lgb

    models = [
        ("RF (100, d=12)", RandomForestRegressor(
            n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)),
        ("RF (300, d=20)", RandomForestRegressor(
            n_estimators=300, max_depth=20, n_jobs=-1, random_state=42)),
        ("XGBoost (200, d=8)", xgb.XGBRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            n_jobs=-1, random_state=42, tree_method="hist")),
        ("LightGBM (200, d=8)", lgb.LGBMRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            n_jobs=-1, random_state=42, verbose=-1)),
        ("LightGBM (500, d=12)", lgb.LGBMRegressor(
            n_estimators=500, max_depth=12, learning_rate=0.05,
            n_jobs=-1, random_state=42, verbose=-1)),
    ]

    print(f"  {'Model':30s} {'Train(s)':>8s} {'Infer(s)':>8s} {'r':>8s} "
          f"{'MAE':>8s} {'RMSE':>8s}", flush=True)
    print("  " + "-" * 75, flush=True)

    model_results = []
    for name, model in models:
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        inf_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            y_pred = model.predict(X_test)
            inf_times.append(time.perf_counter() - t0)
        inf_time = np.median(inf_times) * (n_total / len(X_test))

        corr = np.corrcoef(y_test, y_pred)[0, 1]
        mae = float(np.mean(np.abs(y_test - y_pred)))
        rmse = float(np.sqrt(np.mean((y_test - y_pred)**2)))

        print(f"  {name:30s} {train_time:8.1f} {inf_time:8.1f} {corr:8.4f} "
              f"{mae:8.2f} {rmse:8.2f}", flush=True)
        model_results.append({
            "name": name, "model": model, "train_time": train_time,
            "inf_time": inf_time, "corr": corr, "mae": mae, "rmse": rmse,
        })

    # ── Step 5: Two-tier with best model ─────────────────────────────
    print("\n" + "=" * 100)
    print("Step 5: Two-tier results (best model)")
    print("=" * 100, flush=True)

    # Baseline ROC1
    passing = raw_scores > 0
    hits_base = _collect_top_k_hits(
        candidate_pairs[passing], sims[passing], nq, 500,
        query_ds, db_ds, passing_scores=raw_scores[passing].astype(np.float32))
    baseline_roc1 = evaluate_roc1(hits_base, metadata)

    best = max(model_results, key=lambda r: r["corr"])
    print(f"  Best model: {best['name']} (r={best['corr']:.4f})", flush=True)
    print(f"  Baseline: ROC1={baseline_roc1:.4f}, align_time={aln_time:.1f}s\n", flush=True)

    predicted = best["model"].predict(features)

    print(f"  {'N':>8s} {'Pairs':>12s} {'Align(s)':>10s} {'Speedup':>8s} "
          f"{'ROC1':>7s} {'vs MMseqs2':>10s}", flush=True)
    print("  " + "-" * 65, flush=True)

    for N in [500, 1000, 2000, 3000, 5000, 8000, 12000]:
        qi_col = candidate_pairs[:, 0]
        keep = np.zeros(n_total, dtype=np.bool_)
        for qi in range(nq):
            mask = qi_col == qi
            if mask.sum() == 0:
                continue
            idx = np.where(mask)[0]
            if len(idx) <= N:
                keep[idx] = True
            else:
                top = np.argsort(-predicted[idx])[:N]
                keep[idx[top]] = True

        sel = candidate_pairs[keep]
        sel_m = _remap_pairs_to_merged(sel, merged["nq"])
        sk = np.minimum(sel_m[:, 0], sel_m[:, 1])
        so = np.argsort(sk, kind="mergesort")
        sel_m = sel_m[so]
        sel = sel[so]

        t0 = time.perf_counter()
        s_sims, s_scores, s_mask = _batch_sw_compact_scored(
            sel_m, merged["flat_sequences"], merged["offsets"],
            merged["lengths"], np.float32(0.1), int32(band_width), BLOSUM62)
        at = time.perf_counter() - t0

        p = s_scores > 0
        hits = _collect_top_k_hits(
            sel[p], s_sims[p], nq, 500, query_ds, db_ds,
            passing_scores=s_scores[p].astype(np.float32))
        roc1 = evaluate_roc1(hits, metadata)

        speedup = aln_time / at
        vs = roc1 - 0.7942
        print(f"  {N:8d} {len(sel):12d} {at:9.1f}s {speedup:7.2f}x "
              f"{roc1:7.4f} {vs:+10.4f}", flush=True)

    # Compare all models at N=5000
    print(f"\n  All models at N=5000:", flush=True)
    print(f"  {'Model':30s} {'ROC1':>7s} {'vs baseline':>12s}", flush=True)
    print("  " + "-" * 55, flush=True)
    for r in model_results:
        pred = r["model"].predict(features)
        qi_col = candidate_pairs[:, 0]
        keep = np.zeros(n_total, dtype=np.bool_)
        for qi in range(nq):
            mask = qi_col == qi
            if mask.sum() == 0:
                continue
            idx = np.where(mask)[0]
            if len(idx) <= 5000:
                keep[idx] = True
            else:
                top = np.argsort(-pred[idx])[:5000]
                keep[idx[top]] = True
        sel = candidate_pairs[keep]
        sel_m = _remap_pairs_to_merged(sel, merged["nq"])
        sk = np.minimum(sel_m[:, 0], sel_m[:, 1])
        so = np.argsort(sk, kind="mergesort")
        sel_m = sel_m[so]; sel = sel[so]
        s_sims, s_scores, _ = _batch_sw_compact_scored(
            sel_m, merged["flat_sequences"], merged["offsets"],
            merged["lengths"], np.float32(0.1), int32(band_width), BLOSUM62)
        p = s_scores > 0
        hits = _collect_top_k_hits(
            sel[p], s_sims[p], nq, 500, query_ds, db_ds,
            passing_scores=s_scores[p].astype(np.float32))
        roc1 = evaluate_roc1(hits, metadata)
        print(f"  {r['name']:30s} {roc1:7.4f} {baseline_roc1 - roc1:+12.4f}", flush=True)

    print(f"\n  Reference: MMseqs2=0.7942  DIAMOND=0.7963")
    print(f"  Previous (8 features, RF): r=0.978")


if __name__ == "__main__":
    main()
