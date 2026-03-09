"""Shared test fixtures and synthetic data generators."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


# --- Synthetic protein sequences ---
# Three clusters of similar sequences:
#   Cluster A: seq1, seq2, seq3 (very similar — differ by ~5%)
#   Cluster B: seq4, seq5 (very similar to each other, different from A)
#   Cluster C: seq6 (singleton, short sequence)

CLUSTER_A_BASE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAATALKKALP"
CLUSTER_B_BASE = "GCDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDE"
CLUSTER_C_SEQ = "MKTAY"  # Very short — should be singleton or edge case

SYNTHETIC_FASTA = f""">seq1 Cluster A member 1
{CLUSTER_A_BASE}
>seq2 Cluster A member 2 (2 mutations)
{CLUSTER_A_BASE[:10]}XX{CLUSTER_A_BASE[12:]}
>seq3 Cluster A member 3 (2 mutations at different positions)
{CLUSTER_A_BASE[:40]}XX{CLUSTER_A_BASE[42:]}
>seq4 Cluster B member 1
{CLUSTER_B_BASE}
>seq5 Cluster B member 2 (2 mutations)
{CLUSTER_B_BASE[:20]}XX{CLUSTER_B_BASE[22:]}
>seq6 Cluster C singleton (short)
{CLUSTER_C_SEQ}
"""

SYNTHETIC_FASTQ = f"""@seq1 Cluster A member 1
{CLUSTER_A_BASE}
+
{"I" * len(CLUSTER_A_BASE)}
@seq2 Cluster A member 2
{CLUSTER_A_BASE[:10]}XX{CLUSTER_A_BASE[12:]}
+
{"I" * len(CLUSTER_A_BASE)}
"""


@pytest.fixture
def synthetic_fasta_path(tmp_path):
    """Write synthetic FASTA to a temp file and return its path."""
    fasta_file = tmp_path / "test.fasta"
    fasta_file.write_text(SYNTHETIC_FASTA)
    return fasta_file


@pytest.fixture
def synthetic_fastq_path(tmp_path):
    """Write synthetic FASTQ to a temp file and return its path."""
    fastq_file = tmp_path / "test.fastq"
    fastq_file.write_text(SYNTHETIC_FASTQ)
    return fastq_file


@pytest.fixture
def output_dir(tmp_path):
    """Provide a temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out
