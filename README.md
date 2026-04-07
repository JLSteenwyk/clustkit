# ClustKIT

**Accurate protein sequence clustering via LSH, Smith-Waterman alignment, and Leiden community detection.**

ClustKIT is a bioinformatics tool for protein sequence clustering. It combines MinHash sketching, locality-sensitive hashing (LSH), banded Smith-Waterman alignment, and Leiden community detection to achieve high clustering accuracy at all identity thresholds, including the challenging low-identity regime (30-50%) where greedy heuristic methods lose sensitivity.

## Features

- **Accurate at low identity**: Smith-Waterman alignment with BLOSUM62 scoring and Leiden graph partitioning produce well-connected clusters, especially at thresholds below 50%
- **Fast**: Multi-stage filtering (LSH + Jaccard pre-filter + length-ratio filter) eliminates >98% of candidate pairs before alignment
- **Scalable**: C/OpenMP alignment kernel with configurable thread count; optional GPU acceleration via CuPy
- **Protein and nucleotide**: Supports both sequence types with adaptive k-mer sizing
- **Flexible clustering**: Leiden community detection (default), connected components, or greedy

## Installation

```bash
pip install clustkit
```

With GPU acceleration (requires CUDA 12.x):

```bash
pip install clustkit[gpu]
```

From source:

```bash
git clone https://github.com/JLSteenwyk/ClustKIT.git
cd ClustKIT
pip install -e ".[dev]"
```

## Quick start

```bash
# Cluster proteins at 50% identity using 8 threads
clustkit cluster -i sequences.fasta -o output/ -t 0.5 --threads 8

# Cluster nucleotides at 97% identity
clustkit cluster -i 16s.fasta -o output/ -t 0.97 --mode nucleotide --threads 8

# Use connected components instead of Leiden
clustkit cluster -i sequences.fasta -o output/ -t 0.7 --cluster-method connected --threads 4

# GPU-accelerated alignment
clustkit cluster -i sequences.fasta -o output/ -t 0.3 --device 0 --threads 8
```

Output files:
- `output/clusters.tsv` — Cluster assignments (sequence_id, cluster_id, is_representative)
- `output/representatives.fasta` — Representative sequences
- `output/run_info.json` — Run parameters and statistics

## Pipeline overview

ClustKIT clusters sequences through six phases:

1. **Read** — Parse FASTA/FASTQ, integer-encode sequences
2. **Sketch** — MinHash bottom-s sketches with adaptive k-mer size
3. **LSH** — Locality-sensitive hashing to find candidate pairs
4. **Align** — Banded Smith-Waterman with BLOSUM62 scoring, affine gap penalties, length-ratio and Jaccard pre-filters
5. **Graph** — Sparse similarity graph construction
6. **Cluster** — Leiden community detection (default), connected components, or greedy

## CLI reference

```
clustkit cluster   Cluster sequences by identity threshold
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Input FASTA/FASTQ file | required |
| `-o, --output` | Output directory | required |
| `-t, --threshold` | Identity threshold (0.0-1.0) | 0.9 |
| `--mode` | Sequence type: `protein` or `nucleotide` | `protein` |
| `--threads` | Number of CPU threads | 1 |
| `--device` | `cpu`, `auto`, or GPU device ID (e.g., `0`) | `cpu` |
| `--cluster-method` | `leiden` (default), `connected`, or `greedy` | `leiden` |
| `--alignment` | `align` (SW, accurate) or `kmer` (fast) | `align` |
| `--clustering-mode` | Presets: `balanced`, `accurate`, or `fast` | `balanced` |
| `--sensitivity` | LSH sensitivity: `low`, `medium`, `high` | per mode |
| `--sketch-size` | MinHash sketch size | 128 |
| `-k, --kmer-size` | K-mer size for sketching | 5 (protein), 11 (nucleotide) |
| `--representative` | `longest`, `centroid`, or `most_connected` | `longest` |
| `--format` | Output format: `tsv` or `cdhit` | `tsv` |

## Dependencies

**Core** (installed automatically):

| Package | Purpose |
|---------|---------|
| numpy | Array operations |
| numba | JIT-compiled fallback kernels |
| biopython | FASTA/FASTQ parsing |
| scipy | Sparse graph, connected components |
| leidenalg | Leiden community detection |
| python-igraph | Graph representation for Leiden |
| typer + rich | CLI framework |

**Optional**:

| Package | Install | Purpose |
|---------|---------|---------|
| cupy-cuda12x | `pip install clustkit[gpu]` | GPU-accelerated Smith-Waterman alignment |

## License

MIT License. See [LICENSE](LICENSE) for details.
