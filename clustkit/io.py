"""Sequence I/O: FASTA/FASTQ reading, integer encoding, and output writing."""

from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

# Alphabets for integer encoding
PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
NUCLEOTIDE_ALPHABET = "ACGT"

# Mapping from character to integer (unknown residues map to last index + 1)
PROTEIN_CHAR_TO_INT = {c: i for i, c in enumerate(PROTEIN_ALPHABET)}
NUCLEOTIDE_CHAR_TO_INT = {c: i for i, c in enumerate(NUCLEOTIDE_ALPHABET)}

# Unknown character sentinel
PROTEIN_UNKNOWN = len(PROTEIN_ALPHABET)
NUCLEOTIDE_UNKNOWN = len(NUCLEOTIDE_ALPHABET)


@dataclass
class SequenceRecord:
    """A single sequence with its metadata."""

    id: str
    description: str
    sequence: str
    encoded: np.ndarray = field(default=None, repr=False)
    length: int = 0

    def __post_init__(self):
        self.length = len(self.sequence)


@dataclass
class SequenceDataset:
    """Collection of sequences ready for the clustering pipeline."""

    records: list[SequenceRecord]
    mode: str  # "protein" or "nucleotide"
    encoded_sequences: np.ndarray = field(default=None, repr=False)
    lengths: np.ndarray = field(default=None, repr=False)
    ids: list[str] = field(default_factory=list)

    @property
    def num_sequences(self) -> int:
        return len(self.records)

    @property
    def max_length(self) -> int:
        return int(self.lengths.max()) if self.lengths is not None and len(self.lengths) > 0 else 0


def encode_sequence(sequence: str, mode: str) -> np.ndarray:
    """Encode a sequence string as an integer array.

    Args:
        sequence: Amino acid or nucleotide sequence string.
        mode: "protein" or "nucleotide".

    Returns:
        1D numpy array of uint8 integers.
    """
    if mode == "protein":
        char_map = PROTEIN_CHAR_TO_INT
        unknown = PROTEIN_UNKNOWN
    else:
        char_map = NUCLEOTIDE_CHAR_TO_INT
        unknown = NUCLEOTIDE_UNKNOWN

    seq_upper = sequence.upper()
    encoded = np.array(
        [char_map.get(c, unknown) for c in seq_upper],
        dtype=np.uint8,
    )
    return encoded


def _detect_format(filepath: Path) -> str:
    """Detect whether a file is FASTA or FASTQ from the first character."""
    with open(filepath) as f:
        first_char = f.read(1)
    if first_char == ">":
        return "fasta"
    elif first_char == "@":
        return "fastq"
    else:
        raise ValueError(
            f"Cannot detect format of {filepath}: "
            f"expected '>' (FASTA) or '@' (FASTQ) as first character, got '{first_char}'"
        )


def _parse_fasta(filepath: Path) -> list[SequenceRecord]:
    """Parse a FASTA file into SequenceRecords."""
    records = []
    current_id = None
    current_desc = ""
    current_seq_parts = []

    with open(filepath) as f:
        for line in f:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                # Save previous record
                if current_id is not None:
                    records.append(
                        SequenceRecord(
                            id=current_id,
                            description=current_desc,
                            sequence="".join(current_seq_parts),
                        )
                    )
                # Parse header
                header = line[1:].strip()
                parts = header.split(None, 1)
                current_id = parts[0] if parts else ""
                current_desc = parts[1] if len(parts) > 1 else ""
                current_seq_parts = []
            elif current_id is not None:
                current_seq_parts.append(line.strip())

    # Last record
    if current_id is not None:
        records.append(
            SequenceRecord(
                id=current_id,
                description=current_desc,
                sequence="".join(current_seq_parts),
            )
        )

    return records


def _parse_fastq(filepath: Path) -> list[SequenceRecord]:
    """Parse a FASTQ file into SequenceRecords (quality scores are discarded)."""
    records = []

    with open(filepath) as f:
        while True:
            header = f.readline().rstrip("\n\r")
            if not header:
                break
            if not header.startswith("@"):
                raise ValueError(f"Expected FASTQ header starting with '@', got: {header}")

            sequence = f.readline().rstrip("\n\r")
            f.readline()  # '+' separator line
            f.readline()  # quality scores line

            parts = header[1:].split(None, 1)
            seq_id = parts[0] if parts else ""
            desc = parts[1] if len(parts) > 1 else ""

            records.append(
                SequenceRecord(id=seq_id, description=desc, sequence=sequence)
            )

    return records


def read_sequences(filepath: Path, mode: str) -> SequenceDataset:
    """Read sequences from a FASTA/FASTQ file and return an encoded dataset.

    Args:
        filepath: Path to input FASTA or FASTQ file.
        mode: "protein" or "nucleotide".

    Returns:
        SequenceDataset with encoded sequences ready for the pipeline.
    """
    filepath = Path(filepath)
    fmt = _detect_format(filepath)

    if fmt == "fasta":
        records = _parse_fasta(filepath)
    else:
        records = _parse_fastq(filepath)

    if not records:
        return SequenceDataset(
            records=[],
            mode=mode,
            encoded_sequences=np.empty((0, 0), dtype=np.uint8),
            lengths=np.array([], dtype=np.int32),
            ids=[],
        )

    # Encode sequences
    for rec in records:
        rec.encoded = encode_sequence(rec.sequence, mode)

    lengths = np.array([rec.length for rec in records], dtype=np.int32)
    ids = [rec.id for rec in records]

    # Pad to uniform length for batch processing
    max_len = int(lengths.max())
    n = len(records)

    if mode == "protein":
        pad_value = PROTEIN_UNKNOWN
    else:
        pad_value = NUCLEOTIDE_UNKNOWN

    encoded_matrix = np.full((n, max_len), pad_value, dtype=np.uint8)
    for i, rec in enumerate(records):
        encoded_matrix[i, : rec.length] = rec.encoded

    return SequenceDataset(
        records=records,
        mode=mode,
        encoded_sequences=encoded_matrix,
        lengths=lengths,
        ids=ids,
    )


def write_representatives_fasta(
    filepath: Path,
    dataset: SequenceDataset,
    representative_indices: np.ndarray,
):
    """Write representative sequences to a FASTA file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        for idx in representative_indices:
            rec = dataset.records[idx]
            f.write(f">{rec.id} {rec.description}\n")
            # Write sequence in 80-character lines
            seq = rec.sequence
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")


def write_clusters_tsv(
    filepath: Path,
    dataset: SequenceDataset,
    cluster_labels: np.ndarray,
    representative_indices: np.ndarray,
):
    """Write cluster assignments to a TSV file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    rep_set = set(int(i) for i in representative_indices)

    with open(filepath, "w") as f:
        f.write("sequence_id\tcluster_id\tis_representative\n")
        for i, rec in enumerate(dataset.records):
            is_rep = i in rep_set
            f.write(f"{rec.id}\t{cluster_labels[i]}\t{is_rep}\n")


def write_clusters_cdhit(
    filepath: Path,
    dataset: SequenceDataset,
    cluster_labels: np.ndarray,
    representative_indices: np.ndarray,
):
    """Write cluster assignments in CD-HIT .clstr format."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    rep_set = set(int(i) for i in representative_indices)
    unit = "nt" if dataset.mode == "nucleotide" else "aa"

    # Group sequences by cluster
    clusters: dict[int, list[int]] = {}
    for i, label in enumerate(cluster_labels):
        label = int(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    with open(filepath, "w") as f:
        for cluster_idx, (label, members) in enumerate(sorted(clusters.items())):
            f.write(f">Cluster {cluster_idx}\n")
            for member_idx, seq_idx in enumerate(members):
                rec = dataset.records[seq_idx]
                length = rec.length
                if seq_idx in rep_set:
                    f.write(f"{member_idx}\t{length}{unit}, >{rec.id}... *\n")
                else:
                    f.write(f"{member_idx}\t{length}{unit}, >{rec.id}...\n")
