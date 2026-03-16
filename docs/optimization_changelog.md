# ClustKIT Optimization Changelog

Complete record of every perturbation tested for sequence search sensitivity and speed, with version-tagged results.

---

## Baseline (v0.0) — Pre-optimization

**Search (SCOPe, 10K queries, 591K db, 8 threads):**
| Tool | ROC1 | ROC5 | MAP | Time |
|------|------|------|-----|------|
| BLAST | 0.8395 | 0.8419 | 0.8429 | 963s |
| MMseqs2 (s=7.5) | 0.7902 | 0.7907 | 0.7907 | 75s |
| DIAMOND | 0.7742 | 0.7746 | 0.7746 | 63s |
| ClustKIT (LSH) | 0.6473 | 0.6529 | 0.6535 | 5374s |

**Clustering (Pfam, 22K seqs, 56 families):**
| Tool | ARI (t=0.4) | F1 (t=0.4) | Clusters | Time |
|------|-------------|------------|----------|------|
| ClustKIT (192 threads) | 0.3148 | 0.3183 | 5039 | 375s |
| ClustKIT (4 threads) | 0.3148 | 0.3183 | 5039 | 5706s |

**Thread scaling:** 42x speedup at 192 threads (outperforms MMseqs2 above 64 threads).

---

## v1.0 — K-mer Inverted Index

**Change:** Replaced LSH candidate generation with CSR k-mer inverted index + Phase A counting + Phase B diagonal scoring.

**Impact:** 130x faster search (13s vs 1700s for 2K queries). ROC1=0.609-0.622 (still below MMseqs2).

**Files:** `clustkit/kmer_index.py`, `clustkit/database.py`

---

## v1.1 — Parameter Tuning

**Changes tested (2K query subset, 591K db):**

| Parameter | Values tested | Best | ROC1 impact |
|-----------|--------------|------|-------------|
| phase_a_topk | 5K, 10K, 50K, 100K, 200K, 500K | 200K | Saturates at 200K (0.752) |
| freq_percentile | 90, 95, 97, 99, 99.5, 99.9 | 99.0-99.5 | +0.007 (0.752→0.759) |
| max_cands_per_query | 1K, 2K, 3K, 5K, 8K | 5K+ | Each step helps |
| min_diag_hits | 1, 2, 3 | 2 | Phase B critical (without: -0.08) |

**Pilot files:** `pilot_ranking.py`, `pilot_ranking2.py`, `pilot_topk_ceiling.py`, `pilot_freq_idf.py`, `pilot_best_combo.py`

**Key finding:** Phase A top-K is the dominant parameter. Phase B diagonal scoring is critical. Ranking method doesn't matter (all 9 methods tested gave identical ROC1).

---

## v2.0 — Smith-Waterman Local Alignment

**Change:** Replaced Needleman-Wunsch (global) alignment with Smith-Waterman (local) for the search path. NW penalizes terminal gaps, depressing scores for divergent homologs with different lengths.

**ROC1:** 0.767 → 0.775 (+0.008)

**Details:**
- Score-based filtering (`mask = score > 0`) required — identity-based threshold was too conservative for local alignment
- Full band_width used (no adaptive reduction) since SW doesn't need to span both full sequences
- No length-ratio pre-filter needed

| Config | ROC1 | Aligned | Time |
|--------|------|---------|------|
| NW baseline (mc=5K) | 0.7672 | 10M | 258s |
| SW (mc=3K) | 0.7669 | 6M | 272s |
| SW (mc=5K) | 0.7707 | 10M | 364s |
| SW (mc=8K) | 0.7748 | 16M | 547s |
| SW (mc=15K) | 0.7817 | 30M | 953s |
| SW (mc=20K) | 0.7845 | 40M | 1247s |

**Files:** `clustkit/pairwise.py` (`_sw_identity_and_score`, `_batch_sw_compact_scored`), `clustkit/kmer_index.py` (`local_alignment` parameter)

**Pilot files:** `pilot_sw.py`, `pilot_sw2.py`, `pilot_sw3.py`

---

## v2.1 — E-value Normalization

**Change:** Normalize SW scores by sequence lengths: `rank_score = lambda*S - log(m*n)`

**ROC1:** +0.0002 (negligible)

**Conclusion:** Ranking is not the bottleneck. All ranking methods give identical ROC1.

---

## v3.0 — Reduced Alphabet Dual-Index

**Change:** Map 20 amino acids to 9 groups (Murphy-10-like), build k=4 or k=5 index on reduced alphabet, union candidates with standard k=3 index.

**ROC1:** 0.775 → 0.795 (+0.020) — single biggest improvement

**Alphabet mapping (ClustKIT encoding ACDEFGHIKLMNPQRSTVWY):**
- {A,G}=0 {C}=1 {D,E,N,Q}=2 {F,W,Y}=3 {H}=4 {I,L,M,V}=5 {K,R}=6 {P}=7 {S,T}=8

| Config | ROC1 | Aligned | Time |
|--------|------|---------|------|
| Standard k=3 only (mc=8K) | 0.7748 | 16M | 563s |
| Dual: std k=3 + reduced k=4 (mc=8K) | 0.7949 | 31M | 1169s |
| Dual: std k=3 + reduced k=3 (mc=8K) | 0.7845 | 31M | 2113s |
| Dual: std k=3 + reduced k=5 (mc=8K) | 0.7948 | 31M | 980s |
| Dual: std k=3 + reduced k=4 (mc=15K) | 0.8014 | 57M | 1900s |

**Key finding:** k=4 and k=5 give identical ROC1 but k=5 is 6x faster in Phase A (59K vs 6.5K possible k-mers → shorter posting lists).

**Files:** `clustkit/kmer_index.py` (REDUCED_ALPHA, `_remap_flat`, `reduced_alphabet` parameter)

**Pilot files:** `pilot_reduced_alpha.py`, `pilot_reduced_k5.py`, `pilot_reduced_k4_mc15k.py`

---

## v3.1 — Triple Index (k=4 + k=5)

**Change:** Run BOTH reduced k=4 and k=5 indices, union all candidates.

**ROC1:** 0.795 → 0.800 (+0.005)

| Config | ROC1 | Aligned |
|--------|------|---------|
| Dual k=5 only | 0.7948 | 31M |
| Triple k=[4,5] | 0.7998 | 41M |
| Triple k=[4,5] + IDF | 0.8000 | 41M |

**File:** `clustkit/kmer_index.py` (`reduced_k` accepts list)

---

## v3.2 — IDF-Weighted Phase A

**Change:** Accumulate `log2(N/freq)` instead of raw count per k-mer hit in Phase A. Rare k-mers get higher weight.

**ROC1:** +0.002 (marginal)

**Files:** `clustkit/kmer_index.py` (`_score_query_two_stage_idf`, `_batch_score_queries_idf`, `use_idf` parameter)

---

## v4.0 — Spaced Seeds

**Change:** Non-contiguous k-mer patterns (e.g., `11011`) on reduced alphabet. Tolerates substitutions at "don't care" positions.

**ROC1:** 0.800 → 0.812 (+0.012)

| Config | ROC1 | Aligned | Time |
|--------|------|---------|------|
| Triple k=[4,5] only | 0.8000 | 41M | 1497s |
| + spaced 11011 | 0.8062 | 52M | 2000s |
| + spaced 110011 | 0.8099 | 55M | 2049s |
| + spaced 10101 | 0.8028 | 55M | 2969s |
| + spaced [11011,110011] (mc=8K) | 0.8123 | 65M | 2510s |
| + spaced [11011,110011] (mc=15K) | 0.8178 | 115M | 3953s |

**Files:** `clustkit/kmer_index.py` (`build_kmer_index_spaced`, `_batch_score_queries_spaced`, `spaced_seeds` parameter)

**Pilot files:** `pilot_spaced_seeds.py`

---

## v4.1 — Additional Spaced Seeds

**Change:** Tested 4 more patterns: `101011`, `1010101`, `1001011`, `11000011`

| Config | ROC1 |
|--------|------|
| + 1010101 (w=4, span=7) | 0.8089 |
| + 101011 (w=4, span=6) | 0.8081 |
| + 1001011 (w=4, span=7) | 0.8077 |
| + 11000011 (w=4, span=8) | 0.8068 |
| All 4 seeds combined | **0.8184** |

**Conclusion:** Diminishing returns. Best 4-seed combo (0.8184) barely exceeds 2-seed (0.8178).

**Pilot file:** `pilot_seeds2.py`

---

## v4.2 — Multiple Reduced Alphabets

**Change:** Tested Dayhoff-6 and Hydrophobicity-8 alphabets alongside Murphy-10.

| Config | ROC1 | Aligned | Time |
|--------|------|---------|------|
| Murphy + Dayhoff-6 k=[4,5] | 0.8064 | 63M | 5541s |
| Murphy + Hydro-8 k=[4,5] | 0.8040 | 65M | 4615s |
| Murphy + Dayhoff + Hydro | 0.8091 | 85M | 7254s |
| All alphabets + spaced seeds | 0.8173 | 105M | 8960s |

**Conclusion:** Extra alphabets overlap with spaced seeds. Not worth the compute.

**Pilot file:** `pilot_alphabets.py`

---

## v5.0 — C/OpenMP Extension

**Change:** Rewrote Phase A+B k-mer scoring in C with GCC -O3 -march=native and OpenMP parallelism.

| Index | Numba | C | Speedup |
|-------|-------|---|---------|
| Standard k=3 | 64.5s | 7.6s | **8.4x** |
| Reduced k=5 | 24.2s | 2.6s | **9.3x** |
| Est. all 5 indices | ~184s | ~26s | **~7x** |

**Files:** `clustkit/csrc/kmer_score.c`

**Build:** `gcc -O3 -march=native -fopenmp -shared -fPIC -o kmer_score.so kmer_score.c`

**Pilot file:** `pilot_c_vs_numba.py`

---

## v6.0 — ML Two-Tier Alignment

**Change:** Train ML model to predict SW scores from cheap features. Select top-N per query by predicted score, run actual SW only on the reduced set.

### v6.0a — Initial POC (4 features)
- Features: shared_k3, length_ratio, length_diff, comp_proxy
- Model: HistGradientBoostingRegressor, r=0.965
- Feature cost: 12.6s for 43M pairs

### v6.0b — Model Comparison
| Model | r | MAE | Inference |
|-------|---|-----|-----------|
| RF (100, d=12) | 0.966 | 4.86 | 4.8s |
| MLP (64-32) | 0.967 | 4.87 | 3.9s |
| HistGBR | 0.893 | 5.33 | 11.4s |
| Ridge | 0.653 | 18.66 | <0.1s |

### v6.0c — MAE-Margin Prefilter
**Result:** Not effective at mc=8K. Phase A/B already produces high-quality candidates — almost all pairs have positive predicted scores.

### v6.1 — Rich Features (8 features + n_indices)
- Added: n_indices_found, shared_red_k5, proper composition cosine
- Model: RF, r=0.978 (up from 0.966)
- n_indices is dominant feature (importance 0.846)

### v6.2 — Phase A/B Score Features (13 features)
- Added: actual Phase A/B scores from each of the 5 indices
- Model: LightGBM (500, d=12), **r=0.9837**, MAE=4.22

| Model | r | MAE | Inference |
|-------|---|-----|-----------|
| LightGBM (500, d=12) | **0.9837** | 4.22 | 11.8s |
| RF (300, d=20) | 0.9828 | 4.23 | 21.7s |
| XGBoost (200, d=8) | 0.9646 | 4.31 | 3.1s |

### Two-Tier Results (Best: LightGBM 13 features, full sensitivity config)

| Buffer N | Pairs | Align Speedup | ROC1 | vs MMseqs2 |
|----------|-------|---------------|------|------------|
| 500 | 1M | 61x | 0.794 | -0.001 |
| 1,000 | 2M | 30x | 0.799 | +0.005 |
| 2,000 | 4M | 15x | 0.802 | +0.008 |
| 3,000 | 6M | 10x | 0.804 | +0.010 |
| 5,000 | 10M | 5.9x | 0.807 | +0.013 |
| 8,000 | 16M | 3.7x | 0.809 | +0.015 |
| 12,000 | 24M | 2.5x | 0.811 | +0.017 |

**Files:** `clustkit/kmer_index.py` (`_batch_score_queries_with_scores`, `ml_prefilter_candidates`, `_build_query_kmer_sets`, `_compute_prefilter_features`)

**Pilot files:** `poc_ml_prefilter.py`, `poc_ml_prefilter_models.py`, `pilot_rf_prefilter.py`, `pilot_ml_twotier.py`, `pilot_ml_twotier2.py`, `pilot_ml_twotier3.py`, `pilot_ml_fast_features.py`

---

## Things That Did NOT Help

| Approach | Result | Why |
|----------|--------|-----|
| E-value normalization | +0.0002 ROC1 | Ranking not the bottleneck |
| Ungapped prefilter (3 diags) | -0.029 ROC1 | Checks wrong diagonals, removes true homologs |
| All ranking methods (9 tested) | Identical ROC1 | Candidate generation is the bottleneck |
| BLOSUM62 similar k-mer (k=5) | Below k=3 baseline | Too slow, score-weighted matching didn't help |
| ML MAE-margin prefilter | No filtering at mc=8K | Candidates already high-quality from Phase A/B |
| Reduced k=3 on 9-group alphabet | ROC1=0.7845 | Only 729 k-mers, too noisy, massive posting lists |
| Extra alphabets (Dayhoff, Hydro) | +0.004-0.009 | Overlap with spaced seeds, not worth compute |

---

## v6.3 — Fast Feature Union + Ensemble Scoring

**Change (Speed):** Replaced 5× `np.searchsorted` with single sort-merge pass for building per-pair feature matrix during candidate union.

**Change (Sensitivity):** Tested self-calibrating ensemble scoring — LightGBM trained on (SW_score + index_features) for reranking after alignment. Also tested classifier and regressor for no-alignment ranking.

**Affects:** Search only.

**Ensemble scoring results (full alignment, 65M pairs):**

| Ranking Method | ROC1 | vs SW Baseline |
|----------------|------|----------------|
| SW score (baseline) | 0.8117 | — |
| **Ensemble rerank (SW + index features)** | **0.8117** | **-0.0001** |
| Classifier P(high SW) | 0.7489 | -0.063 |
| Regressor predicted SW (no alignment) | 0.7604 | -0.051 |

**Small-sample calibration:** 100 queries → r=0.971 (vs r=0.983 full training). Viable for database-agnostic deployment.

**Verdict:** Ensemble reranking gives **zero ROC1 improvement**. SW scores are already optimal for ranking. Index features add no new signal. No-alignment approaches (classifier/regressor) are 5-6% worse. **SW alignment is irreplaceable for ranking quality.**

**Pilot file:** `pilot_fast_ensemble.py`

---

## v7.0 — C/OpenMP SW Alignment

**Change:** Rewrote banded Smith-Waterman in C with GCC -O3 -march=native and OpenMP. Pre-allocated thread-local DP workspace eliminates per-pair malloc overhead.

**Affects:** Search only (clustering uses NW path).

| Config | Time (31M pairs) | Speedup vs Numba | Score Correlation |
|--------|-------------------|-------------------|-------------------|
| Numba SW (bw=126) | 863s | 1.0x | baseline |
| **C SW (bw=126)** | **89s** | **9.7x** | **1.000 (exact)** |
| **C SW (bw=50)** | **38s** | **22.5x** | 0.995 |

**Key finding:** 100% exact score match at bw=126 confirms the C implementation is bit-identical to Numba. Band reduction to 50 gives additional 2.3x with r=0.995 score correlation.

**Files:** `clustkit/csrc/sw_align.c`

**Build:** `gcc -O3 -march=native -fopenmp -shared -fPIC -o sw_align.so sw_align.c`

**Pilot file:** `pilot_c_sw.py`

---

## Current Best Pipeline

```
Standard k=3 Phase A+B (C extension)     →  8K candidates/query
Reduced k=4 Phase A+B (Numba)            →  8K candidates/query
Reduced k=5 Phase A+B (Numba)            →  8K candidates/query
Spaced seed 11011 Phase A+B (Numba)      →  8K candidates/query
Spaced seed 110011 Phase A+B (Numba)     →  8K candidates/query
                                             ↓
                     Union + deduplicate → ~65M unique pairs
                                             ↓
              ML predict (LightGBM) → select top-N per query
                                             ↓
                   SW local alignment → actual scores
                                             ↓
                  Rank by SW score → top-500 hits/query
```

**Estimated end-to-end timing (all C extensions + ML two-tier N=2000):**
- Candidate gen (C extension): ~26s
- ML features + inference: ~35s
- SW alignment (C, 4M pairs, bw=126): ~11s
- SW alignment (C, 4M pairs, bw=50): ~5s
- **Total (bw=126): ~72s** (5x vs MMseqs2)
- **Total (bw=50): ~66s** (4.7x vs MMseqs2)
- vs MMseqs2: 14s
- ROC1: 0.802 (beats MMseqs2 0.794)

**Without ML two-tier (full alignment, max sensitivity):**
- Candidate gen (C): ~26s
- SW alignment (C, 31M pairs, bw=126): ~89s
- **Total: ~115s** (8x vs MMseqs2)
- ROC1: 0.818
