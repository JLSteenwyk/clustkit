# ClustKIT Search Benchmark Results

## SCOPe Search Benchmark

**Dataset:** 591,288 database sequences from SCOPe 2.08
**Ground truth:** Same family = TP, different fold = FP, same superfamily/different family = IGNORE
**Metric:** ROC1 = mean fraction of TPs ranked above the 1st FP per query

### Pre-optimization Baseline (10K queries)

| Tool | ROC1 | ROC5 | MAP | S@1%FDR | TP | FP | Time |
|------|------|------|-----|---------|----|----|------|
| BLAST | 0.8395 | 0.8419 | 0.8429 | 0.8396 | 384,763 | 100,419 | 963s |
| MMseqs2 (s=7.5) | 0.7902 | 0.7907 | 0.7907 | 0.7902 | 349,116 | 947 | 75s |
| DIAMOND | 0.7742 | 0.7746 | 0.7746 | 0.7742 | 340,194 | 953 | 63s |
| ClustKIT (LSH) | 0.6473 | 0.6529 | 0.6535 | 0.6474 | 261,389 | 151,679 | 5374s |

### Post-optimization Results (2K query subset)

**Reference values (2K subset):** MMseqs2=0.7942, DIAMOND=0.7963

#### Sensitivity Evolution

| Version | Config | ROC1 | vs MMseqs2 |
|---------|--------|------|------------|
| v1.0 | K-mer index, k=3 | 0.622 | -0.172 |
| v1.1 | + tuned params (topk=200K, pctl=99.5) | 0.764 | -0.030 |
| v2.0 | + SW alignment (mc=8K) | 0.775 | -0.019 |
| v2.0 | + SW (mc=20K) | 0.785 | -0.010 |
| v3.0 | + reduced alphabet k=4 (mc=8K) | 0.795 | +0.001 |
| v3.0 | + reduced alphabet k=4 (mc=15K) | 0.801 | +0.007 |
| v3.1 | + triple index k=[4,5] | 0.800 | +0.006 |
| v3.2 | + IDF-weighted Phase A | 0.808 | +0.014 |
| v4.0 | + spaced seeds [11011,110011] (mc=8K) | 0.812 | +0.018 |
| v4.0 | + spaced seeds [11011,110011] (mc=15K) | 0.818 | +0.024 |
| **v4.1** | **+ 4 seeds combined** | **0.818** | **+0.024** |

#### Speed Evolution

| Version | Config | Est. Total | vs MMseqs2 |
|---------|--------|-----------|------------|
| v1.0 | K-mer index | ~13s | 0.9x |
| v2.0 | + SW (mc=8K) | ~980s | 70x |
| v3.0 | + reduced alphabet | ~980s | 70x |
| v4.0 | + spaced seeds | ~2500s | 179x |
| v5.0 | + C extension (scoring) | ~26s scoring | — |
| v6.2 | + ML two-tier N=2K | ~187s | 13x |
| **v7.0** | **+ C SW alignment (bw=126)** | **~72s** | **5x** |
| **v7.0** | **+ C SW alignment (bw=50)** | **~66s** | **4.7x** |
| v7.0 | C scoring + C SW (no ML, full) | ~115s | 8x |
| v7.3 | All-C + LGB-50 N=3000 bw=50 | ~65s | 4.6x |
| **v7.4** | **5idx All-C + LGB-50 N=5000 bw=50** | **~129s** | **9.2x** |

---

## Detailed Speed-Sensitivity Trade-off

### MC Scaling (SW alignment, no reduced alphabet)

| mc | ROC1 | Aligned | Time |
|----|------|---------|------|
| 3,000 | 0.767 | 6M | 272s |
| 5,000 | 0.771 | 10M | 364s |
| 8,000 | 0.775 | 16M | 547s |
| 15,000 | 0.782 | 30M | 953s |
| 20,000 | 0.785 | 40M | 1247s |

### MC Scaling (with reduced alphabet dual-index)

| mc | ROC1 | Aligned | Time |
|----|------|---------|------|
| 8,000 (k=4) | 0.795 | 31M | 1169s |
| 8,000 (k=5) | 0.795 | 31M | 980s |
| 15,000 (k=4) | 0.801 | 57M | 1900s |

### Spaced Seed Patterns

| Pattern | Weight | Span | ROC1 (added to triple) | Time |
|---------|--------|------|------------------------|------|
| 11011 | 4 | 5 | 0.806 | 2000s |
| 110011 | 4 | 6 | 0.810 | 2049s |
| 10101 | 3 | 5 | 0.803 | 2969s |
| 101011 | 4 | 6 | 0.808 | 3873s |
| 1010101 | 4 | 7 | 0.809 | 3902s |
| 1001011 | 4 | 7 | 0.808 | 3888s |
| 11000011 | 4 | 8 | 0.807 | 3914s |
| [11011,110011] | — | — | 0.812 | 2510s |
| [11011,110011,1010101,11000011] | — | — | 0.818 | 6740s |

### ML Two-Tier (LightGBM, 13 features, full sensitivity config)

| Buffer N | Pairs Aligned | Align Speedup | ROC1 | vs MMseqs2 |
|----------|--------------|---------------|------|------------|
| 500 | 1M | 61x | 0.794 | -0.001 |
| 1,000 | 2M | 30x | 0.799 | +0.005 |
| 2,000 | 4M | 15x | 0.802 | +0.008 |
| 3,000 | 6M | 10x | 0.804 | +0.010 |
| 5,000 | 10M | 5.9x | 0.807 | +0.013 |
| 8,000 | 16M | 3.7x | 0.809 | +0.015 |
| 12,000 | 24M | 2.5x | 0.811 | +0.017 |

---

## Negative Results

### Ungapped Prefilter
| Threshold | ROC1 | Aligned | Speedup |
|-----------|------|---------|---------|
| None | 0.795 | 31M | 1.0x |
| >= 11 | 0.766 | 26M | 1.1x |
| >= 15 | 0.707 | 14M | 1.8x |
| >= 20 | 0.655 | 4M | 4.0x |

**Reason:** Only checks 3 fixed diagonals, misses true homologs on other diagonals.

### ML MAE-Margin Prefilter
| Margin | % Kept | ROC1 | Speedup |
|--------|--------|------|---------|
| 2.0×MAE | 100% | 0.795 | 1.00x |
| 1.5×MAE | 100% | 0.795 | 1.00x |
| 1.0×MAE | 99.6% | 0.795 | 1.01x |
| 0.5×MAE | 94.1% | 0.794 | 1.02x |

**Reason:** At mc=8K, Phase A/B already produces high-quality candidates — almost all pairs score well.

### Ranking Method Comparison
All 9 methods tested (raw_score, identity, evalue, bit_score, per_residue, etc.) gave identical ROC1. Bottleneck is candidate generation, not ranking.

---

## File Inventory

### Result JSONs
- `benchmarks/data/scop_search_results/scop_search_results.json` — Full SCOPe baseline
- `benchmarks/data/speed_sensitivity_results/speed_sensitivity_results.json` — Speed-sensitivity curves
- `benchmarks/data/speed_sensitivity_results/sw_pilot.json` — SW alignment results
- `benchmarks/data/speed_sensitivity_results/best_combo_pilot.json` — Parameter tuning
- `benchmarks/data/speed_sensitivity_results/reduced_alpha_pilot.json` — Reduced alphabet
- `benchmarks/data/speed_sensitivity_results/idf_triple_pilot.json` — IDF + triple index
- `benchmarks/data/speed_sensitivity_results/spaced_seeds_pilot.json` — Spaced seeds
- `benchmarks/data/speed_sensitivity_results/seeds2_pilot.json` — Additional seeds
- `benchmarks/data/speed_sensitivity_results/alphabets_pilot.json` — Multiple alphabets
- `benchmarks/data/speed_sensitivity_results/speed_pilot.json` — Ungapped prefilter
- `benchmarks/data/speed_sensitivity_results/rf_prefilter_results.json` — RF prefilter
- `benchmarks/data/speed_sensitivity_results/ml_twotier_results.json` — ML two-tier v1
- `benchmarks/data/speed_sensitivity_results/ml_twotier2_results.json` — ML two-tier v2
- `benchmarks/data/speed_sensitivity_results/ml_model_comparison.json` — Model comparison

### Log Files
All pilot logs in `benchmarks/pilot_*.log` and `benchmarks/poc_*.log`.
