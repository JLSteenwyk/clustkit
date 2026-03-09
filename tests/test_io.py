"""Tests for sequence I/O: reading, encoding, and output writing."""

import numpy as np
import pytest

from clustkit.io import (
    NUCLEOTIDE_ALPHABET,
    PROTEIN_ALPHABET,
    encode_sequence,
    read_sequences,
    write_clusters_tsv,
    write_representatives_fasta,
)


class TestEncodeSequence:
    def test_protein_basic(self):
        seq = "ACDEF"
        encoded = encode_sequence(seq, "protein")
        assert len(encoded) == 5
        assert encoded.dtype == np.uint8
        # First char 'A' should map to index 0
        assert encoded[0] == PROTEIN_ALPHABET.index("A")
        assert encoded[4] == PROTEIN_ALPHABET.index("F")

    def test_nucleotide_basic(self):
        seq = "ACGT"
        encoded = encode_sequence(seq, "nucleotide")
        assert list(encoded) == [0, 1, 2, 3]

    def test_unknown_residue(self):
        seq = "AXZ"  # X and Z are not in protein alphabet
        encoded = encode_sequence(seq, "protein")
        assert encoded[0] == PROTEIN_ALPHABET.index("A")
        assert encoded[1] == len(PROTEIN_ALPHABET)  # unknown sentinel
        assert encoded[2] == len(PROTEIN_ALPHABET)

    def test_lowercase_handled(self):
        seq = "acgt"
        encoded = encode_sequence(seq, "nucleotide")
        assert list(encoded) == [0, 1, 2, 3]

    def test_empty_sequence(self):
        encoded = encode_sequence("", "protein")
        assert len(encoded) == 0


class TestReadSequences:
    def test_read_fasta(self, synthetic_fasta_path):
        dataset = read_sequences(synthetic_fasta_path, "protein")
        assert dataset.num_sequences == 6
        assert dataset.mode == "protein"
        assert dataset.ids[0] == "seq1"
        assert dataset.ids[5] == "seq6"
        assert dataset.encoded_sequences.shape[0] == 6
        assert dataset.lengths[0] > 0

    def test_read_fastq(self, synthetic_fastq_path):
        dataset = read_sequences(synthetic_fastq_path, "protein")
        assert dataset.num_sequences == 2
        assert dataset.ids[0] == "seq1"

    def test_sequence_lengths(self, synthetic_fasta_path):
        dataset = read_sequences(synthetic_fasta_path, "protein")
        for i, rec in enumerate(dataset.records):
            assert dataset.lengths[i] == len(rec.sequence)

    def test_padding(self, synthetic_fasta_path):
        dataset = read_sequences(synthetic_fasta_path, "protein")
        max_len = dataset.max_length
        assert dataset.encoded_sequences.shape == (6, max_len)

    def test_empty_file(self, tmp_path):
        empty_fasta = tmp_path / "empty.fasta"
        empty_fasta.write_text("")
        with pytest.raises(ValueError, match="Cannot detect format"):
            read_sequences(empty_fasta, "protein")

    def test_format_detection_invalid(self, tmp_path):
        bad_file = tmp_path / "bad.txt"
        bad_file.write_text("not a fasta file\n")
        with pytest.raises(ValueError, match="Cannot detect format"):
            read_sequences(bad_file, "protein")


class TestWriteOutputs:
    def test_write_representatives_fasta(self, synthetic_fasta_path, output_dir):
        dataset = read_sequences(synthetic_fasta_path, "protein")
        reps = np.array([0, 3, 5], dtype=np.int32)

        out_file = output_dir / "reps.fasta"
        write_representatives_fasta(out_file, dataset, reps)

        content = out_file.read_text()
        assert ">seq1" in content
        assert ">seq4" in content
        assert ">seq6" in content
        assert ">seq2" not in content

    def test_write_clusters_tsv(self, synthetic_fasta_path, output_dir):
        dataset = read_sequences(synthetic_fasta_path, "protein")
        labels = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
        reps = np.array([0, 3, 5], dtype=np.int32)

        out_file = output_dir / "clusters.tsv"
        write_clusters_tsv(out_file, dataset, labels, reps)

        lines = out_file.read_text().strip().split("\n")
        assert lines[0] == "sequence_id\tcluster_id\tis_representative"
        assert len(lines) == 7  # header + 6 sequences
        # seq1 should be representative of cluster 0
        assert "seq1\t0\tTrue" in lines[1]
