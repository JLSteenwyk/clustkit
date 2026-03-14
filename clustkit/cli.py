"""Command-line interface for ClustKIT."""

from pathlib import Path
from typing import Optional

import typer

from clustkit import __version__

app = typer.Typer(
    name="clustkit",
    help="Sequence clustering and search for bioinformatics.",
    add_completion=False,
)


def version_callback(value: bool):
    if value:
        typer.echo(f"clustkit {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """ClustKIT: Accurate sequence clustering and search via adaptive banded alignment."""


@app.command()
def cluster(
    input: Path = typer.Option(
        ...,
        "-i",
        "--input",
        help="Input FASTA/FASTQ file.",
        exists=True,
        readable=True,
    ),
    output: Path = typer.Option(
        ...,
        "-o",
        "--output",
        help="Output directory.",
    ),
    threshold: float = typer.Option(
        0.9,
        "-t",
        "--threshold",
        help="Sequence identity threshold (0.0-1.0).",
        min=0.0,
        max=1.0,
    ),
    mode: str = typer.Option(
        "protein",
        "--mode",
        help="Sequence type: 'protein' or 'nucleotide'.",
    ),
    alignment: str = typer.Option(
        "align",
        "--alignment",
        help="Similarity method: 'align' (default, accurate) or 'kmer' (fast).",
    ),
    sketch_size: int = typer.Option(
        128,
        "--sketch-size",
        help="Number of minimizer hashes per sequence.",
    ),
    kmer_size: Optional[int] = typer.Option(
        None,
        "-k",
        "--kmer-size",
        help="K-mer size (default: 5 for protein, 11 for nucleotide).",
    ),
    sensitivity: str = typer.Option(
        "medium",
        "--sensitivity",
        help="LSH sensitivity: 'low', 'medium', or 'high'.",
    ),
    cluster_method: str = typer.Option(
        "connected",
        "--cluster-method",
        help="Clustering method: 'connected', 'greedy', or 'leiden'.",
    ),
    representative: str = typer.Option(
        "longest",
        "--representative",
        help="Representative selection: 'longest', 'centroid', or 'most_connected'.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Device: 'cpu', 'auto', or GPU device ID (e.g., '0'). "
             "'auto' benchmarks a sample to pick the fastest.",
    ),
    threads: int = typer.Option(
        1,
        "--threads",
        help="Number of CPU threads (CPU mode only).",
        min=1,
    ),
    format: str = typer.Option(
        "tsv",
        "--format",
        help="Output format: 'tsv' or 'cdhit'.",
    ),
):
    """Cluster sequences by identity threshold."""
    from clustkit.pipeline import run_pipeline

    # Resolve k-mer size default
    if kmer_size is None:
        kmer_size = 5 if mode == "protein" else 11

    config = {
        "input": input,
        "output": output,
        "threshold": threshold,
        "mode": mode,
        "alignment": alignment,
        "sketch_size": sketch_size,
        "kmer_size": kmer_size,
        "sensitivity": sensitivity,
        "cluster_method": cluster_method,
        "representative": representative,
        "device": device,
        "threads": threads,
        "format": format,
    }

    run_pipeline(config)


@app.command()
def search(
    query: Path = typer.Option(
        ...,
        "-q",
        "--query",
        help="Query FASTA/FASTQ file.",
        exists=True,
        readable=True,
    ),
    db: Path = typer.Option(
        ...,
        "--db",
        help="Database FASTA/FASTQ file or pre-built index directory.",
    ),
    output: Path = typer.Option(
        ...,
        "-o",
        "--output",
        help="Output TSV file for search results.",
    ),
    threshold: float = typer.Option(
        0.5,
        "-t",
        "--threshold",
        help="Minimum sequence identity to report (0.0-1.0).",
        min=0.0,
        max=1.0,
    ),
    top_k: int = typer.Option(
        10,
        "--top-k",
        help="Maximum number of hits per query.",
        min=1,
    ),
    mode: str = typer.Option(
        "protein",
        "--mode",
        help="Sequence type: 'protein' or 'nucleotide'.",
    ),
    sketch_size: int = typer.Option(
        128,
        "--sketch-size",
        help="Number of minimizer hashes per sequence.",
    ),
    kmer_size: Optional[int] = typer.Option(
        None,
        "-k",
        "--kmer-size",
        help="K-mer size (default: 5 for protein, 11 for nucleotide).",
    ),
    sensitivity: str = typer.Option(
        "high",
        "--sensitivity",
        help="LSH sensitivity: 'low', 'medium', or 'high'.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Device: 'cpu', 'auto', or GPU device ID.",
    ),
    threads: int = typer.Option(
        1,
        "--threads",
        help="Number of CPU threads.",
        min=1,
    ),
):
    """Search query sequences against a database."""
    import numba

    from clustkit.io import read_sequences
    from clustkit.database import load_database
    from clustkit.search import search_sequences, write_search_results_tsv

    numba.set_num_threads(threads)

    # Resolve k-mer size default
    if kmer_size is None:
        kmer_size = 5 if mode == "protein" else 11

    # Read query sequences
    query_dataset = read_sequences(query, mode)

    # Load database: either pre-built index dir or raw FASTA
    db_path = Path(db)
    if (db_path / "params.json").exists():
        # Pre-built index
        db_index = load_database(db_path)
        db_dataset = db_index.dataset
        db_sketches = db_index.sketches
    else:
        # Raw FASTA — sketch on the fly
        db_dataset = read_sequences(db_path, mode)
        db_sketches = None

    results = search_sequences(
        query_dataset=query_dataset,
        db_dataset=db_dataset,
        threshold=threshold,
        top_k=top_k,
        mode=mode,
        kmer_size=kmer_size,
        sketch_size=sketch_size,
        sensitivity=sensitivity,
        device=device,
        db_sketches=db_sketches,
    )

    write_search_results_tsv(output, results)
    typer.echo(
        f"Search complete: {sum(len(h) for h in results.hits)} hits "
        f"for {results.num_queries} queries in {results.runtime_seconds:.2f}s"
    )


@app.command()
def makedb(
    input: Path = typer.Option(
        ...,
        "-i",
        "--input",
        help="Input FASTA/FASTQ file.",
        exists=True,
        readable=True,
    ),
    output: Path = typer.Option(
        ...,
        "-o",
        "--output",
        help="Output directory for the database index.",
    ),
    mode: str = typer.Option(
        "protein",
        "--mode",
        help="Sequence type: 'protein' or 'nucleotide'.",
    ),
    threshold: float = typer.Option(
        0.5,
        "-t",
        "--threshold",
        help="Target identity threshold for LSH parameter tuning.",
        min=0.0,
        max=1.0,
    ),
    sketch_size: int = typer.Option(
        128,
        "--sketch-size",
        help="Number of minimizer hashes per sequence.",
    ),
    kmer_size: Optional[int] = typer.Option(
        None,
        "-k",
        "--kmer-size",
        help="K-mer size (default: 5 for protein, 11 for nucleotide).",
    ),
    sensitivity: str = typer.Option(
        "high",
        "--sensitivity",
        help="LSH sensitivity: 'low', 'medium', or 'high'.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Device: 'cpu', 'auto', or GPU device ID.",
    ),
    threads: int = typer.Option(
        1,
        "--threads",
        help="Number of CPU threads.",
        min=1,
    ),
):
    """Pre-build a database index for fast searching."""
    import numba

    from clustkit.database import build_database, save_database

    numba.set_num_threads(threads)

    # Resolve k-mer size default
    if kmer_size is None:
        kmer_size = 5 if mode == "protein" else 11

    db_index = build_database(
        input_path=input,
        mode=mode,
        kmer_size=kmer_size,
        sketch_size=sketch_size,
        threshold=threshold,
        sensitivity=sensitivity,
        device=device,
    )

    save_database(db_index, output)
    typer.echo(
        f"Database built: {db_index.params['num_sequences']} sequences, "
        f"saved to {output}"
    )
