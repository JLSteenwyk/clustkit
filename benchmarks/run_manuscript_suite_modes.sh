#!/usr/bin/env bash
set -euo pipefail

THREADS=${1:-4}
RESULTS_ROOT=${2:-"benchmarks/data/manuscript_results_modes"}
PARALLEL_JOBS=${3:-2}
MODES=(balanced accurate fast)

mkdir -p "$RESULTS_ROOT"

echo "============================================================"
echo "ClustKIT Manuscript Suite"
echo "Threads: $THREADS"
echo "Results root: $RESULTS_ROOT"
echo "Modes: ${MODES[*]}"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "============================================================"
echo

if [ ! -d "benchmarks/data/pfam_families" ]; then
    echo ">>> Step 0: Downloading Pfam families..."
    python benchmarks/download_pfam_families.py
else
    echo ">>> Step 0: Pfam data already present, skipping download."
fi
echo

run_logged_job() {
    local logfile=$1
    shift
    "$@" 2>&1 | tee "$logfile"
}

wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL_JOBS" ]; do
        wait -n
    done
}

for MODE in "${MODES[@]}"; do
    MODE_DIR="$RESULTS_ROOT/$MODE"
    mkdir -p "$MODE_DIR"

    echo "############################################################"
    echo "Clustering mode: $MODE"
    echo "############################################################"
    echo

    echo ">>> Pfam concordance benchmark [$MODE]..."
    run_logged_job "$MODE_DIR/pfam_concordance.log" \
        python benchmarks/benchmark_pfam_concordance.py \
        --threads "$THREADS" \
        --clustkit-mode "$MODE" \
        --thresholds 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    echo

    echo ">>> Thread scaling [$MODE]..."
    run_logged_job "$MODE_DIR/thread_scaling.log" \
        python benchmarks/benchmark_thread_scaling.py \
        --max-threads "$THREADS" \
        --clustkit-mode "$MODE" \
        --threshold 0.5
    echo

    echo ">>> Launching internal mode diagnostics in parallel [$MODE]..."

    echo "    - Ablation [$MODE]"
    wait_for_slot
    run_logged_job "$MODE_DIR/ablation.log" \
        python benchmarks/benchmark_ablation.py \
        --threads "$THREADS" \
        --clustkit-mode "$MODE" \
        --threshold 0.4 &

    echo "    - LSH recall [$MODE]"
    if [ -f "benchmarks/data/pfam_mixed.fasta" ]; then
        wait_for_slot
        run_logged_job "$MODE_DIR/lsh_recall.log" \
            python benchmarks/benchmark_lsh_recall.py \
            -i benchmarks/data/pfam_mixed.fasta \
            -t 0.5 \
            --threads "$THREADS" \
            --clustkit-mode "$MODE" \
            -o "$MODE_DIR/lsh_recall_results.json" &
    else
        echo "  (skipped: pfam_mixed.fasta not found)"
    fi

    echo "    - Phase 3 profiling [$MODE]"
    if [ -f "benchmarks/data/pfam_mixed.fasta" ]; then
        wait_for_slot
        run_logged_job "$MODE_DIR/profile_phase3.log" \
            python benchmarks/profile_phase3.py \
            -i benchmarks/data/pfam_mixed.fasta \
            -t 0.4 \
            --clustkit-mode "$MODE" \
            --max-threads "$THREADS" \
            --skip-timing &
    else
        echo "  (skipped: pfam_mixed.fasta not found)"
    fi
    echo

    wait
done

echo ">>> Correctness validation..."
if [ -f "benchmarks/data/pfam_mixed.fasta" ]; then
    run_logged_job "$RESULTS_ROOT/correctness.log" \
        python benchmarks/benchmark_correctness.py \
        -i benchmarks/data/pfam_mixed.fasta \
        --threads "$THREADS"
else
    echo "  (skipped: pfam_mixed.fasta not found)"
fi
echo

echo "============================================================"
echo "Manuscript suite complete."
echo "============================================================"
