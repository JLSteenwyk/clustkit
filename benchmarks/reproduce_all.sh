#!/usr/bin/env bash
#
# Reproduce all ClustKIT benchmarks for the manuscript.
#
# Usage:
#   bash benchmarks/reproduce_all.sh [THREADS]
#
# Prerequisites:
#   - ClustKIT installed: pip install -e ".[dev,benchmarks]"
#   - External tools: CD-HIT, MMseqs2, VSEARCH (paths configured in scripts)
#   - Pfam data downloaded: python benchmarks/download_pfam_families.py
#
set -euo pipefail

THREADS=${1:-4}
RESULTS_DIR="benchmarks/data/manuscript_results"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "ClustKIT Benchmark Reproduction Suite"
echo "Threads: $THREADS"
echo "Results: $RESULTS_DIR"
echo "============================================================"
echo

# Step 0: Download Pfam benchmark data (if not present)
if [ ! -d "benchmarks/data/pfam_families" ]; then
    echo ">>> Step 0: Downloading Pfam families..."
    python benchmarks/download_pfam_families.py
else
    echo ">>> Step 0: Pfam data already present, skipping download."
fi
echo

# Step 1: Pfam concordance benchmark (Table 1, Fig 2)
echo ">>> Step 1: Pfam concordance benchmark..."
python benchmarks/benchmark_pfam_concordance.py \
    --threads "$THREADS" \
    --thresholds 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
    2>&1 | tee "$RESULTS_DIR/pfam_concordance.log"
echo

# Step 2: Ablation study (Fig 6)
echo ">>> Step 2: Ablation study at t=0.4..."
python benchmarks/benchmark_ablation.py \
    --threads "$THREADS" \
    --threshold 0.4 \
    2>&1 | tee "$RESULTS_DIR/ablation.log"
echo

# Step 3: Thread scaling (Fig S3)
echo ">>> Step 3: Thread scaling..."
python benchmarks/benchmark_thread_scaling.py \
    --max-threads "$THREADS" \
    --threshold 0.5 \
    2>&1 | tee "$RESULTS_DIR/thread_scaling.log"
echo

# Step 4: Correctness validation (supplementary)
echo ">>> Step 4: Correctness validation..."
if [ -f "benchmarks/data/pfam_mixed.fasta" ]; then
    python benchmarks/benchmark_correctness.py \
        -i benchmarks/data/pfam_mixed.fasta \
        --threads "$THREADS" \
        2>&1 | tee "$RESULTS_DIR/correctness.log"
else
    echo "  (skipped: pfam_mixed.fasta not found — run step 1 first)"
fi
echo

# Step 5: LSH recall measurement (Fig S6)
echo ">>> Step 5: LSH recall measurement..."
if [ -f "benchmarks/data/pfam_mixed.fasta" ]; then
    python benchmarks/benchmark_lsh_recall.py \
        -i benchmarks/data/pfam_mixed.fasta \
        -t 0.5 \
        --threads "$THREADS" \
        2>&1 | tee "$RESULTS_DIR/lsh_recall.log"
else
    echo "  (skipped: pfam_mixed.fasta not found — run step 1 first)"
fi
echo

# Step 6: Phase 3 profiling (Fig S4, S5)
echo ">>> Step 6: Phase 3 profiling..."
if [ -f "benchmarks/data/pfam_mixed.fasta" ]; then
    python benchmarks/profile_phase3.py \
        -i benchmarks/data/pfam_mixed.fasta \
        -t 0.4 \
        --max-threads "$THREADS" \
        --skip-timing \
        2>&1 | tee "$RESULTS_DIR/profile_phase3.log"
else
    echo "  (skipped: pfam_mixed.fasta not found — run step 1 first)"
fi
echo

# Step 7: Search benchmarks (Fig 4, requires Pfam data)
echo ">>> Step 7: Search benchmarks (CS4)..."
python benchmarks/benchmark_search.py cs4 \
    --threads "$THREADS" \
    --skip-blast --skip-diamond \
    2>&1 | tee "$RESULTS_DIR/search_cs4.log"
echo

echo "============================================================"
echo "All benchmarks complete. Results saved to $RESULTS_DIR/"
echo "============================================================"
echo
echo "Optional benchmarks (require additional data downloads):"
echo "  # Large-scale scaling (CP1) — requires SwissProt FASTA:"
echo "  python benchmarks/benchmark_large_scale.py -i swissprot.fasta --threads $THREADS"
echo
echo "  # Nucleotide benchmark (C3) — requires 16S rRNA data:"
echo "  python benchmarks/benchmark_nucleotide.py -i 16s.fasta --taxonomy taxonomy.tsv --threads $THREADS"
echo
echo "  # Annotation consistency (CP2-CP4) — requires UniProt-GOA:"
echo "  python benchmarks/benchmark_annotation_consistency.py cp2 cp3 cp4 \\"
echo "      --fasta swissprot.fasta --goa-file goa.gaf --pfam-file interpro.tsv --threads $THREADS"
echo
echo "  # SCOP sensitivity (CP5) — requires SCOP data:"
echo "  python benchmarks/benchmark_search.py cp5 --threads $THREADS"
