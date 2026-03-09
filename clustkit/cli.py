"""Command-line interface for ClustKIT."""

from pathlib import Path
from typing import Optional

import typer

from clustkit import __version__

app = typer.Typer(
    name="clustkit",
    help="GPU-accelerated sequence clustering for bioinformatics.",
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
    """ClustKIT: GPU-accelerated sequence clustering."""


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
        help="Device: 'cpu' or GPU device ID (e.g., '0').",
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
