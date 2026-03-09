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


# --- Synthetic nucleotide sequences ---
# Three clusters of DNA sequences (~100bp, ~90% identity within clusters):
#   Cluster NA: nseq1, nseq2, nseq3 (very similar — differ by ~10%)
#   Cluster NB: nseq4, nseq5, nseq6 (very similar to each other, different from NA)
#   Cluster NC: nseq7, nseq8, nseq9 (very similar to each other, different from NA and NB)

# Cluster NA base: 100bp sequence rich in A/T
NT_CLUSTER_A_BASE = "ATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAG"
# Cluster NB base: 100bp sequence rich in G/C (different composition)
NT_CLUSTER_B_BASE = "GCGCGCGCGCGCAATTAATTAATTGCGCGCGCGCGCAATTAATTAATTGCGCGCGCGCGCAATTAATTAATTGCGCGCGCGCGCAATTAATTAATTGCGCGC"
# Cluster NC base: 100bp different pattern
NT_CLUSTER_C_BASE = "TATATATATAGCGCGCGCGATATATATATAGCGCGCGCGATATATATATAGCGCGCGCGATATATATATAGCGCGCGCGATATATATATAGCGCGCGCGATATAT"

def _mutate_nt(seq, positions):
    """Replace nucleotides at given positions with a different base."""
    complement = {"A": "T", "T": "A", "G": "C", "C": "G"}
    s = list(seq)
    for p in positions:
        s[p] = complement[s[p]]
    return "".join(s)

# ~90% identity: mutate ~10 positions out of 100
NT_CLUSTER_A_MUT1 = _mutate_nt(NT_CLUSTER_A_BASE, [5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
NT_CLUSTER_A_MUT2 = _mutate_nt(NT_CLUSTER_A_BASE, [3, 13, 23, 33, 43, 53, 63, 73, 83, 93])
NT_CLUSTER_B_MUT1 = _mutate_nt(NT_CLUSTER_B_BASE, [4, 14, 24, 34, 44, 54, 64, 74, 84, 94])
NT_CLUSTER_B_MUT2 = _mutate_nt(NT_CLUSTER_B_BASE, [7, 17, 27, 37, 47, 57, 67, 77, 87, 97])
NT_CLUSTER_C_MUT1 = _mutate_nt(NT_CLUSTER_C_BASE, [2, 12, 22, 32, 42, 52, 62, 72, 82, 92])
NT_CLUSTER_C_MUT2 = _mutate_nt(NT_CLUSTER_C_BASE, [6, 16, 26, 36, 46, 56, 66, 76, 86, 96])

SYNTHETIC_NT_FASTA = f""">nseq1 Cluster NA member 1
{NT_CLUSTER_A_BASE}
>nseq2 Cluster NA member 2 (10 mutations)
{NT_CLUSTER_A_MUT1}
>nseq3 Cluster NA member 3 (10 mutations at different positions)
{NT_CLUSTER_A_MUT2}
>nseq4 Cluster NB member 1
{NT_CLUSTER_B_BASE}
>nseq5 Cluster NB member 2 (10 mutations)
{NT_CLUSTER_B_MUT1}
>nseq6 Cluster NB member 3 (10 mutations)
{NT_CLUSTER_B_MUT2}
>nseq7 Cluster NC member 1
{NT_CLUSTER_C_BASE}
>nseq8 Cluster NC member 2 (10 mutations)
{NT_CLUSTER_C_MUT1}
>nseq9 Cluster NC member 3 (10 mutations)
{NT_CLUSTER_C_MUT2}
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
def synthetic_nt_fasta_path(tmp_path):
    """Write synthetic nucleotide FASTA to a temp file and return its path."""
    fasta_file = tmp_path / "test_nt.fasta"
    fasta_file.write_text(SYNTHETIC_NT_FASTA)
    return fasta_file


@pytest.fixture
def output_dir(tmp_path):
    """Provide a temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out
