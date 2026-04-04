"""Build random subsamples of TrEMBL for scaling benchmarks.

Creates 100K, 500K, 1M, 5M subsets from TrEMBL.
Also counts UniRef50 sequences.
"""
import random
import sys
import time
from pathlib import Path

TREMBL = Path("/mnt/85740f55-8e9a-4214-9500-be446866627e/uniprot/uniprot_trembl.fasta")
UNIREF50 = Path("/mnt/85740f55-8e9a-4214-9500-be446866627e/uniref50/uniref50.fasta")
OUT_DIR = Path("/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/ClustKIT/benchmarks/data/scaling")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIZES = [100_000, 500_000, 1_000_000, 5_000_000]


def count_sequences(fasta_path):
    """Count sequences in FASTA (streaming)."""
    count = 0
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count


def reservoir_sample(fasta_path, n, seed=42):
    """Reservoir sampling of n sequences from a FASTA file.
    Returns list of (header, sequence) tuples.
    """
    rng = random.Random(seed)
    reservoir = []
    idx = 0
    current_header = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                if current_header is not None:
                    record = (current_header, "".join(current_seq))
                    if idx < n:
                        reservoir.append(record)
                    else:
                        j = rng.randint(0, idx)
                        if j < n:
                            reservoir[j] = record
                    idx += 1
                    if idx % 1_000_000 == 0:
                        print(f"    ...processed {idx/1e6:.0f}M sequences", flush=True)
                current_header = line.rstrip()
                current_seq = []
            else:
                current_seq.append(line.rstrip())

        # Last record
        if current_header is not None:
            record = (current_header, "".join(current_seq))
            if idx < n:
                reservoir.append(record)
            else:
                j = rng.randint(0, idx)
                if j < n:
                    reservoir[j] = record

    return reservoir


def write_fasta(records, output_path):
    """Write records to FASTA file."""
    with open(output_path, "w") as f:
        for header, seq in records:
            f.write(header + "\n")
            # Write sequence in 80-char lines
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")


if __name__ == "__main__":
    max_size = max(SIZES)
    print(f"Sampling {max_size:,} sequences from TrEMBL (202.6M)...", flush=True)
    t0 = time.time()
    samples = reservoir_sample(str(TREMBL), max_size)
    elapsed = time.time() - t0
    print(f"  Sampled {len(samples):,} sequences in {elapsed:.0f}s\n", flush=True)

    # Shuffle deterministically then write subsets
    rng = random.Random(42)
    rng.shuffle(samples)

    for size in SIZES:
        subset = samples[:size]
        out_path = OUT_DIR / f"trembl_{size//1000}k.fasta"
        print(f"Writing {out_path.name} ({size:,} sequences)...", end=" ", flush=True)
        write_fasta(subset, out_path)
        import os
        fsize = os.path.getsize(out_path) / 1e6
        print(f"{fsize:.0f} MB", flush=True)

    # Also count UniRef50
    print(f"\nCounting UniRef50 sequences...", flush=True)
    n_uniref = count_sequences(str(UNIREF50))
    print(f"  UniRef50: {n_uniref:,} sequences")

    print("\nDone!")
