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
    """Collection of sequences ready for the clustering pipeline.

    Supports two storage formats:
    - **Padded matrix** (``encoded_sequences``): (N, max_len) uint8 — used by
      GPU kernels and legacy code.  Created lazily from compact format when
      first accessed if not stored explicitly.
    - **Compact flat** (``flat_sequences`` + ``offsets``): concatenated 1D
      uint8 array with per-sequence start offsets.  ~25x smaller than the
      padded matrix when outlier sequences inflate max_len.
    """

    records: list[SequenceRecord]
    mode: str  # "protein" or "nucleotide"
    # Padded (N, max_len) matrix — may be None if compact format is used
    _encoded_sequences: np.ndarray = field(default=None, repr=False)
    lengths: np.ndarray = field(default=None, repr=False)
    ids: list[str] = field(default_factory=list)
    # Compact storage: flat 1D array + offsets
    flat_sequences: np.ndarray = field(default=None, repr=False)
    offsets: np.ndarray = field(default=None, repr=False)

    @property
    def num_sequences(self) -> int:
        return len(self.records)

    @property
    def max_length(self) -> int:
        return int(self.lengths.max()) if self.lengths is not None and len(self.lengths) > 0 else 0

    @property
    def encoded_sequences(self) -> np.ndarray:
        """Return padded (N, max_len) matrix, building it lazily from compact format."""
        if self._encoded_sequences is not None:
            return self._encoded_sequences
        if self.flat_sequences is not None and self.offsets is not None:
            self._encoded_sequences = self._build_padded_matrix()
            return self._encoded_sequences
        return None

    @encoded_sequences.setter
    def encoded_sequences(self, value):
        self._encoded_sequences = value

    def _build_padded_matrix(self) -> np.ndarray:
        """Reconstruct the padded (N, max_len) matrix from compact storage."""
        n = len(self.lengths)
        max_len = self.max_length
        pad_value = PROTEIN_UNKNOWN if self.mode == "protein" else NUCLEOTIDE_UNKNOWN
        matrix = np.full((n, max_len), pad_value, dtype=np.uint8)
        for i in range(n):
            start = int(self.offsets[i])
            length = int(self.lengths[i])
            matrix[i, :length] = self.flat_sequences[start:start + length]
        return matrix

    def drop_padded_matrix(self):
        """Free the padded matrix to reclaim memory (compact format must exist)."""
        if self.flat_sequences is not None:
            self._encoded_sequences = None

    def subset(self, indices: np.ndarray) -> "SequenceDataset":
        """Create a new SequenceDataset containing only the specified sequences.

        Args:
            indices: 1D int array of sequence indices to keep (original indexing).

        Returns:
            New SequenceDataset with re-indexed compact storage.
        """
        indices = np.asarray(indices, dtype=np.int64)
        sub_records = [self.records[i] for i in indices]
        sub_lengths = self.lengths[indices]
        sub_ids = [self.ids[i] for i in indices]

        # Build new compact storage
        total_len = int(np.sum(sub_lengths))
        sub_flat = np.empty(total_len, dtype=np.uint8)
        sub_offsets = np.empty(len(indices), dtype=np.int64)
        pos = 0
        for j, orig_i in enumerate(indices):
            length = int(self.lengths[orig_i])
            start = int(self.offsets[orig_i])
            sub_offsets[j] = pos
            sub_flat[pos:pos + length] = self.flat_sequences[start:start + length]
            pos += length

        return SequenceDataset(
            records=sub_records,
            mode=self.mode,
            _encoded_sequences=None,
            lengths=sub_lengths,
            ids=sub_ids,
            flat_sequences=sub_flat,
            offsets=sub_offsets,
        )


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
            _encoded_sequences=np.empty((0, 0), dtype=np.uint8),
            lengths=np.array([], dtype=np.int32),
            ids=[],
            flat_sequences=np.empty(0, dtype=np.uint8),
            offsets=np.array([], dtype=np.int64),
        )

    # Encode sequences
    for rec in records:
        rec.encoded = encode_sequence(rec.sequence, mode)

    lengths = np.array([rec.length for rec in records], dtype=np.int32)
    ids = [rec.id for rec in records]
    n = len(records)

    # Build compact storage: flat 1D array + offsets
    total_len = int(lengths.sum())
    flat_sequences = np.empty(total_len, dtype=np.uint8)
    offsets = np.empty(n, dtype=np.int64)
    pos = 0
    for i, rec in enumerate(records):
        offsets[i] = pos
        flat_sequences[pos:pos + rec.length] = rec.encoded
        pos += rec.length

    return SequenceDataset(
        records=records,
        mode=mode,
        _encoded_sequences=None,  # built lazily from compact format when needed
        lengths=lengths,
        ids=ids,
        flat_sequences=flat_sequences,
        offsets=offsets,
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
