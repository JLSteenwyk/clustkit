"""Prepare 16S rRNA benchmark dataset from SILVA 138.1 NR99.

Filters for bacteria, subsamples, removes gaps (SILVA uses aligned sequences),
and creates taxonomy ground truth file.
"""

import random
import sys
from pathlib import Path

SILVA_FASTA = Path(__file__).resolve().parent / "data" / "16s_silva" / "SILVA_138.1_SSURef_NR99_tax_silva.fasta"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "16s_silva"

MAX_SEQS = 50000
MIN_LENGTH = 1200  # Filter short fragments
MAX_LENGTH = 2000  # Filter abnormally long sequences
SEED = 42
TAX_LEVEL = 5  # 0=Domain, 1=Phylum, 2=Class, 3=Order, 4=Family, 5=Genus


def main():
    print(f"Reading SILVA FASTA: {SILVA_FASTA}")

    # Parse all bacterial sequences
    bacteria = []
    header = None
    seq_parts = []

    with open(SILVA_FASTA) as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if header is not None and "Bacteria;" in header:
                    seq = "".join(seq_parts).replace(".", "").replace("-", "").upper()
                    seq = seq.replace("U", "T")  # SILVA uses RNA alphabet
                    if MIN_LENGTH <= len(seq) <= MAX_LENGTH:
                        # Parse taxonomy from header
                        parts = header.split(" ", 1)
                        seq_id = parts[0]
                        if len(parts) > 1:
                            tax_str = parts[1]
                            tax_levels = [t.strip() for t in tax_str.split(";") if t.strip()]
                            if len(tax_levels) > TAX_LEVEL:
                                genus = tax_levels[TAX_LEVEL]
                                if genus and "uncultured" not in genus.lower() and "Unknown" not in genus and "metagenome" not in genus.lower():
                                    bacteria.append((seq_id, seq, genus))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line.strip())

    # Handle last sequence
    if header is not None and "Bacteria;" in header:
        seq = "".join(seq_parts).replace(".", "").replace("-", "").upper()
        seq = seq.replace("U", "T")
        if MIN_LENGTH <= len(seq) <= MAX_LENGTH:
            parts = header.split(" ", 1)
            seq_id = parts[0]
            if len(parts) > 1:
                tax_str = parts[1]
                tax_levels = [t.strip() for t in tax_str.split(";") if t.strip()]
                if len(tax_levels) > TAX_LEVEL:
                    genus = tax_levels[TAX_LEVEL]
                    if genus and genus != "uncultured" and "Unknown" not in genus:
                        bacteria.append((seq_id, seq, genus))

    print(f"  Found {len(bacteria)} bacterial sequences with genus annotation (length {MIN_LENGTH}-{MAX_LENGTH})")

    # Count genera
    genus_counts = {}
    for _, _, genus in bacteria:
        genus_counts[genus] = genus_counts.get(genus, 0) + 1

    print(f"  {len(genus_counts)} distinct genera")

    # Filter genera with at least 5 members (meaningful clusters)
    valid_genera = {g for g, c in genus_counts.items() if c >= 5}
    bacteria = [(sid, seq, genus) for sid, seq, genus in bacteria if genus in valid_genera]
    print(f"  After filtering genera with >=5 members: {len(bacteria)} sequences, {len(valid_genera)} genera")

    # Subsample if needed
    if len(bacteria) > MAX_SEQS:
        random.seed(SEED)
        bacteria = random.sample(bacteria, MAX_SEQS)
        print(f"  Subsampled to {MAX_SEQS} sequences")

    # Recount after subsampling
    genus_counts = {}
    for _, _, genus in bacteria:
        genus_counts[genus] = genus_counts.get(genus, 0) + 1
    print(f"  Final: {len(bacteria)} sequences, {len(genus_counts)} genera")

    # Length stats
    lengths = [len(seq) for _, seq, _ in bacteria]
    print(f"  Length: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.0f}")

    # Write output FASTA
    output_fasta = OUTPUT_DIR / "silva_16s_bacteria.fasta"
    with open(output_fasta, "w") as fh:
        for seq_id, seq, genus in bacteria:
            fh.write(f">{seq_id}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i + 80] + "\n")
    print(f"  Written: {output_fasta}")

    # Write taxonomy file
    output_tax = OUTPUT_DIR / "silva_16s_taxonomy.tsv"
    with open(output_tax, "w") as fh:
        for seq_id, seq, genus in bacteria:
            fh.write(f"{seq_id}\t{genus}\n")
    print(f"  Written: {output_tax}")

    # Summary
    top_genera = sorted(genus_counts.items(), key=lambda x: -x[1])[:20]
    print(f"\n  Top 20 genera:")
    for genus, count in top_genera:
        print(f"    {genus}: {count}")


if __name__ == "__main__":
    main()
