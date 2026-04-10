"""Command-line interface for ClustKIT."""

from pathlib import Path
from typing import Optional

import typer

from clustkit import __version__
from clustkit.clustering_mode import resolve_clustering_mode

app = typer.Typer(
    name="clustkit",
    help="Accurate protein sequence clustering via LSH, Smith-Waterman alignment, and Leiden community detection.",
    add_completion=False,
    invoke_without_command=True,
)


def version_callback(value: bool):
    if value:
        typer.echo(f"clustkit {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    input: Optional[Path] = typer.Option(
        None,
        "-i",
        "--input",
        help="Input FASTA/FASTQ file.",
        exists=True,
        readable=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
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
    alignment: str = typer.Option(
        "align",
        "--alignment",
        help="Similarity method: 'align' (default, accurate) or 'kmer' (fast).",
    ),
    clustering_mode: str = typer.Option(
        "balanced",
        "--clustering-mode",
        help="Threshold-aware clustering mode: 'balanced', 'accurate', or 'fast'.",
    ),
    sketch_size: Optional[int] = typer.Option(
        None,
        "--sketch-size",
        help="Number of minimizer hashes per sequence. Overrides --clustering-mode.",
    ),
    kmer_size: Optional[int] = typer.Option(
        None,
        "-k",
        "--kmer-size",
        help="K-mer size for sketching (default: 5).",
    ),
    sensitivity: Optional[str] = typer.Option(
        None,
        "--sensitivity",
        help="LSH sensitivity: 'low', 'medium', or 'high'. Overrides --clustering-mode.",
    ),
    cluster_method: str = typer.Option(
        "leiden",
        "--cluster-method",
        help="Clustering method: 'leiden' (default), 'connected', or 'greedy'.",
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
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """Accurate protein sequence clustering via LSH, Smith-Waterman alignment, and Leiden community detection."""
    if input is None or output is None:
        typer.echo("Error: --input (-i) and --output (-o) are required.")
        raise typer.Exit(code=1)

    from clustkit.pipeline import run_pipeline

    if kmer_size is None:
        kmer_size = 5
    try:
        sketch_size, sensitivity = resolve_clustering_mode(
            clustering_mode, threshold, sketch_size, sensitivity
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    config = {
        "input": input,
        "output": output,
        "threshold": threshold,
        "clustering_mode": clustering_mode,
        "mode": "protein",
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
