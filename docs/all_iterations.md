# ClustKIT: Complete Iteration Results

Every configuration tested across sensitivity, speed, and ML experiments.
SCOPe benchmark: 591K database, 2000 query subset, 8 threads.
Reference: MMseqs2 (s=7.5) ROC1=0.7942, 14s. DIAMOND ROC1=0.7963, 13s.

---

## 1. Sensitivity Iterations (Candidate Generation)

### 1.1 Alignment Method

| Version | Change | ROC1 | Delta | Time |
|---------|--------|------|-------|------|
| v0.0 | ClustKIT LSH baseline | 0.647 | — | 5374s |
| v1.0 | K-mer inverted index (k=3) | 0.622 | -0.025 | 13s |
| v1.1 | + tuned params (topk=200K, pctl=99.5) | 0.764 | +0.142 | ~160s |
| **v2.0** | **NW → SW local alignment (mc=8K)** | **0.775** | **+0.011** | 547s |

### 1.2 SW Alignment Variants

| Config | ROC1 | Time | Notes |
|--------|------|------|-------|
| NW baseline (mc=5K) | 0.767 | 258s | Global alignment |
| SW mc=3K | 0.767 | 272s | Too few candidates |
| SW mc=5K | 0.771 | 364s | |
| SW mc=8K | 0.775 | 547s | |
| SW mc=15K | 0.782 | 953s | |
| SW mc=20K | 0.785 | 1247s | Diminishing returns |

### 1.3 SW Score Filtering

| Method | ROC1 | Notes |
|--------|------|-------|
| Identity-based (matches/shorter ≥ 0.1) | 0.770 | Too conservative for SW |
| **Score-based (score > 0)** | **0.775** | Passes all positive SW scores |

### 1.4 Reduced Alphabet

| Config | ROC1 | Delta | Time |
|--------|------|-------|------|
| Standard k=3 only (mc=8K) | 0.775 | — | 563s |
| **+ Reduced k=4 (Murphy-10, mc=8K)** | **0.795** | **+0.020** | 1169s |
| + Reduced k=3 (Murphy-10, mc=8K) | 0.785 | +0.010 | 2113s |
| + Reduced k=5 (Murphy-10, mc=8K) | 0.795 | +0.020 | 980s |
| + Reduced k=4 (mc=15K) | 0.801 | +0.026 | 1900s |

### 1.5 Triple Index

| Config | ROC1 | Delta vs dual | Time |
|--------|------|---------------|------|
| Dual k=5 only (mc=8K) | 0.795 | — | 981s |
| Dual k=5 + IDF (mc=8K) | 0.794 | -0.001 | 993s |
| **Triple k=[4,5] (mc=8K)** | **0.800** | **+0.005** | 1452s |
| Triple k=[4,5] + IDF (mc=8K) | 0.800 | +0.005 | 1482s |
| Triple k=[4,5] + IDF (mc=15K) | 0.808 | +0.013 | 2453s |

### 1.6 IDF-Weighted Phase A

| Config | ROC1 | Delta | Notes |
|--------|------|-------|-------|
| No IDF (triple, mc=8K) | 0.800 | — | |
| **+ IDF scoring** | **0.800** | **+0.000** | Negligible effect |
| IDF on dual k=5 | 0.794 | -0.001 | Slightly worse |

### 1.7 Spaced Seeds

| Pattern | Weight | Span | ROC1 (added to triple) | Time |
|---------|--------|------|------------------------|------|
| None (triple only) | — | — | 0.800 | 1497s |
| **11011** | **4** | **5** | **0.806** | 2000s |
| **110011** | **4** | **6** | **0.810** | 2049s |
| 10101 | 3 | 5 | 0.803 | 2969s |
| [11011,110011] mc=8K | — | — | 0.812 | 2510s |
| **[11011,110011] mc=15K** | — | — | **0.818** | 3953s |
| 101011 | 4 | 6 | 0.808 | 3873s |
| 1010101 | 4 | 7 | 0.809 | 3902s |
| 1001011 | 4 | 7 | 0.808 | 3888s |
| 11000011 | 4 | 8 | 0.807 | 3914s |
| [11011,110011,1010101,11000011] | — | — | 0.818 | 6740s |

### 1.8 Multiple Alphabets

| Config | ROC1 | Time | Notes |
|--------|------|------|-------|
| Murphy-10 only (triple+IDF) | 0.800 | 1497s | Baseline |
| + Dayhoff-6 k=[4,5] | 0.806 | 5541s | +0.006 |
| + Hydro-8 k=[4,5] | 0.804 | 4615s | +0.004 |
| Murphy + Dayhoff + Hydro | 0.809 | 7254s | Overlaps with seeds |
| All alphabets + spaced seeds | 0.817 | 8960s | Diminishing returns |

### 1.9 E-value Normalization

| Config | ROC1 | Notes |
|--------|------|-------|
| Raw SW scores | 0.775 | |
| **+ E-value normalization** | **0.775** | **+0.0002 — no effect** |

### 1.10 Ranking Methods (all identical)

| Method | ROC1 |
|--------|------|
| Raw score | 0.697 |
| Identity | 0.697 |
| E-value | 0.697 |
| Bit score | 0.697 |
| Per-residue score | 0.697 |
| Normalized score (sqrt) | 0.697 |

All 9 methods: identical ROC1. Candidate generation is the bottleneck, not ranking.

### 1.11 Parameter Sweeps

**phase_a_topk:**

| topk | ROC1 | Notes |
|------|------|-------|
| 5K | 0.697 | |
| 10K | 0.720 | |
| 50K | 0.736 | |
| 100K | 0.751 | |
| **200K** | **0.752** | **Saturates here** |
| 500K | 0.752 | Same as 200K |

**freq_percentile:**

| Percentile | ROC1 |
|------------|------|
| 90.0 | 0.745 |
| 95.0 | 0.752 |
| 97.0 | 0.755 |
| **99.0** | **0.759** |
| **99.5** | **0.764** |
| 99.9 | 0.758 |

**max_cands_per_query (with SW):**

| mc | ROC1 | Time |
|----|------|------|
| 3K | 0.767 | 272s |
| 5K | 0.771 | 364s |
| 8K | 0.775 | 547s |
| 15K | 0.782 | 953s |
| 20K | 0.785 | 1247s |

---

## 2. Speed Iterations

### 2.1 C/OpenMP Extensions

| Component | Numba | C | Speedup |
|-----------|-------|---|---------|
| Standard k=3 scoring | 64.5s | 7.6s | 8.4x |
| Reduced k=5 scoring | 24.2s | 2.6s | 9.3x |
| Spaced 110011 scoring | 157s | 22s | 7x |
| SW alignment (31M, bw=126) | 863s | 89s | 9.7x |
| SW alignment (31M, bw=50) | — | 38s | 22.5x vs Numba |

### 2.2 Diagonal-Hint SW Banding

| Config | Time (16M pairs) | ROC1 | vs bw=126 |
|--------|-------------------|------|-----------|
| No hint, bw=126 | 48.5s | 0.7710 | baseline |
| No hint, bw=50 | 20.6s | 0.7685 | -0.003 |
| No hint, bw=30 | 11.7s | 0.7629 | -0.008 |
| No hint, bw=20 | 7.5s | 0.7547 | -0.016 |
| **Diag hint, bw=50** | **19.0s** | **0.7702** | **-0.001** |
| Diag hint, bw=30 | 10.5s | 0.7692 | -0.002 |
| Diag hint, bw=20 | 6.7s | 0.7686 | -0.002 |
| Diag hint, bw=15 | 5.0s | 0.7684 | -0.003 |

### 2.3 Ungapped Prefilter (Negative Result)

| Threshold | ROC1 | Aligned | Speedup |
|-----------|------|---------|---------|
| None | 0.795 | 31M | 1.0x |
| ≥ 11 | 0.766 | 26M | 1.1x |
| ≥ 15 | 0.707 | 14M | 1.8x |
| ≥ 20 | 0.655 | 4M | 4.0x |

Harmful — checks wrong diagonals, removes true homologs.

### 2.4 Fused Multi-Index Scoring (Negative Result)

| Config | Scoring | Total | ROC1 |
|--------|---------|-------|------|
| Fused (no Phase B) | 4s | 19s | 0.665 |
| Separate (with Phase B) | 77s | 129s | 0.802 |

Phase B cannot be skipped — ROC1 collapses by -0.14.

### 2.5 Early Termination

All 2000 queries had fast-index score ≥ 20. Slow indices contributed only -0.001 ROC1.

| Config | Time | ROC1 |
|--------|------|------|
| All 5 indices | 152s | 0.769 |
| 2 indices only | 53s | 0.768 |

### 2.6 Index Count Comparison

| Indices | Time | ROC1 | Notes |
|---------|------|------|-------|
| 2 (std+r5) | 53-73s | 0.768-0.784 | Below MMseqs2 |
| **3 (std+r5+sp)** | **97-134s** | **0.798-0.810** | **Beats MMseqs2** |
| 4 (std+r4+r5+sp) | 155s | 0.799 | Marginal gain |
| 5 (std+r4+r5+sp1+sp2) | 129s | 0.802 | From v7.4 |

### 2.7 Stable vs Unstable C Tie-Breaking

| Config | ROC1 | Notes |
|--------|------|-------|
| Unstable qsort (bw=20) | 0.798 | Arbitrary tie-breaking |
| **Stable sort (bw=50)** | **0.810** | **Ascending ID for ties (+0.012)** |

### 2.8 Pre-Built Index Infrastructure

| Database | Numba ROC1 | C ROC1 | Notes |
|----------|-----------|--------|-------|
| Old DB (no pre-built) | 0.808 | — | On-the-fly index build |
| **V3 DB (pre-built)** | **0.808** | **0.810** | **No regression** |

### 2.9 Compact Phase A Index

| Config | Time | ROC1 | Notes |
|--------|------|------|-------|
| Full entries (int64) | ~100s | 0.798 | 8 bytes/entry |
| Compact entries (int32) | ~98s | 0.798 | 4 bytes/entry, marginal speedup |

---

## 3. ML Experiments

### 3.1 ML Prefilter POC (4 features)

| Model | Pearson r | MAE | Inference (43M) |
|-------|-----------|-----|-----------------|
| HistGBR (d=6, n=200) | 0.893 | 5.33 | 11.4s |
| **RF (n=100)** | **0.966** | **4.86** | **4.8s** |
| MLP (64-32) | 0.967 | 4.87 | 3.9s |
| Ridge | 0.653 | 18.66 | <0.1s |

### 3.2 ML Model Comparison (4 features → 8 → 13)

| Features | Best Model | Pearson r | MAE |
|----------|-----------|-----------|-----|
| 4 (shared_k3, lengths) | RF (100) | 0.966 | 4.86 |
| 8 (+ n_indices, red_k5, comp) | RF (100) | 0.978 | 4.86 |
| **13 (+ per-index Phase A/B scores)** | **LightGBM (500)** | **0.984** | **4.22** |

### 3.3 ML Two-Tier (4 features, dual k=5)

| Buffer N | Align Speedup | ROC1 |
|----------|---------------|------|
| 500 | 6.7x | 0.750 |
| 1000 | 5.5x | 0.760 |
| 2000 | 4.1x | 0.772 |
| 5000 | 2.3x | 0.784 |
| 8000 | 1.6x | 0.790 |

### 3.4 ML Two-Tier (13 features, full sensitivity)

| Buffer N | Align Speedup | ROC1 | vs MMseqs2 |
|----------|---------------|------|------------|
| 500 | 61x | 0.794 | -0.001 |
| 1000 | 30x | 0.799 | +0.005 |
| 2000 | 15x | 0.802 | +0.008 |
| 5000 | 5.9x | 0.807 | +0.013 |
| 8000 | 3.7x | 0.809 | +0.015 |
| 12000 | 2.5x | 0.811 | +0.017 |

### 3.5 ML MAE-Margin Prefilter (Negative Result)

| Margin | % Kept | ROC1 |
|--------|--------|------|
| 2.0×MAE | 100% | 0.795 |
| 1.5×MAE | 100% | 0.795 |
| 0.5×MAE | 94% | 0.794 |

Not effective — Phase A/B already produces high-quality candidates.

### 3.6 Ensemble Reranking (Negative Result)

| Method | ROC1 | vs SW baseline |
|--------|------|----------------|
| SW score (baseline) | 0.812 | — |
| Ensemble (SW + features) | 0.812 | -0.000 |
| Classifier P(high SW) | 0.749 | -0.063 |
| Regressor (no alignment) | 0.760 | -0.051 |

SW scores already optimal for ranking — no ML model improves on them.

### 3.7 Self-Calibrating LightGBM

| Model Size | Calibration | ROC1 at N=3000 |
|------------|------------|----------------|
| LGB-500-d12 | 500 queries | 0.793 |
| LGB-200-d8 | 500 queries | 0.794 |
| LGB-100-d6 | 500 queries | 0.795 |
| **LGB-50-d4** | **500 queries** | **0.796** |
| LGB-50-d4 | 1000 queries | 0.800 |

Smallest model generalizes best — larger models overfit calibration sample.

### 3.8 Heuristic vs LightGBM Selection

| Method | ROC1 (N=5000) | Notes |
|--------|---------------|-------|
| Heuristic (n_idx*10+max) | 0.786 | Simple formula |
| **LightGBM-50** | **0.796+** | **Better ranking** |

---

## 4. Pipeline Evolution Summary

| Version | Time | ROC1 | vs MMseqs2 | Key Change |
|---------|------|------|------------|------------|
| v0.0 | 5374s | 0.647 | -0.147 | LSH baseline |
| v1.0 | 13s | 0.622 | -0.172 | K-mer inverted index |
| v1.1 | ~160s | 0.764 | -0.030 | Parameter tuning |
| v2.0 | 547s | 0.775 | -0.019 | SW local alignment |
| v3.0 | 1169s | 0.795 | +0.001 | Reduced alphabet dual-index |
| v3.1 | 1452s | 0.800 | +0.006 | Triple index k=[4,5] |
| v4.0 | 2510s | 0.812 | +0.018 | Spaced seeds [11011,110011] |
| v4.1 | 6740s | 0.818 | +0.024 | 4 spaced seeds (peak sensitivity) |
| v5.0 | — | — | — | C/OpenMP scoring (8-9x speedup) |
| v6.0 | — | 0.802 | +0.008 | ML two-tier (LightGBM) |
| v7.0 | — | — | — | C/OpenMP SW alignment (10x speedup) |
| v7.3 | 65s | 0.796 | +0.002 | All-C + LGB-50 (3 indices) |
| v7.4 | 129s | 0.802 | +0.008 | 5 indices + LGB + N=5000 |
| v7.6 | — | — | — | Diagonal-hint alignment banding |
| v8.0 | 98s | 0.798 | +0.003 | Integrated C pipeline + diag hints |
| **v8.1** | **134s** | **0.810** | **+0.016** | **Stable tie-breaking + bw=50** |

---

## 5. Negative Results (Important Lessons)

| Approach | Result | Why It Failed |
|----------|--------|---------------|
| E-value normalization | +0.0002 | Ranking not the bottleneck |
| Ungapped prefilter (3 diags) | -0.029 | Checks wrong diagonals |
| 9 ranking methods | All identical | Candidate generation is bottleneck |
| BLOSUM62 similar k-mer (k=5) | Below baseline | Score-weighted matching unhelpful |
| ML MAE-margin prefilter | No filtering | Candidates already high-quality |
| Reduced k=3 (9 groups) | 0.785 | Only 729 k-mers, too noisy |
| Extra alphabets (Dayhoff, Hydro) | +0.004-0.009 | Overlap with spaced seeds |
| More seed patterns (4 total) | +0.006 | Diminishing returns |
| Fused scoring (no Phase B) | 0.665 | Phase B essential for quality |
| Heuristic selection | 0.786 | LightGBM ranks better |
| Ensemble reranking | +0.000 | SW scores already optimal |
