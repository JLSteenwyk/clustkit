"""Download Pfam family sequences from UniProt for benchmarking.

Downloads reviewed (SwissProt) sequences for diverse Pfam families.
"""

import os
import sys
import time
from pathlib import Path
import urllib.request
import urllib.error

DATA_DIR = Path(__file__).resolve().parent / "data" / "pfam_families"

# Diverse Pfam families with moderate sizes (100-2000 reviewed seqs each)
# Existing 15 families are already downloaded
EXISTING = {
    "PF00042", "PF00067", "PF00072", "PF00076", "PF00089",
    "PF00106", "PF00115", "PF00155", "PF00171", "PF00226",
    "PF00244", "PF00389", "PF00501", "PF00561", "PF00753",
}

# Additional families to download (diverse protein functions)
NEW_FAMILIES = {
    # Enzymes
    "PF00069": "Protein_kinase",
    "PF00085": "Thioredoxin",
    "PF00112": "Papain_cysteine_protease",
    "PF00128": "Alpha_amylase",
    "PF00175": "NAD_dependent_oxidoreductase",
    "PF00179": "UQ_dehydrogenase",
    "PF00195": "Chalcone_isomerase",
    "PF00248": "Aldo_keto_reductase",
    "PF00291": "Pyridoxal_phosphate_enzyme",
    "PF00462": "Glutaredoxin",
    "PF00487": "FA_desaturase",
    "PF00702": "Haloacid_dehalogenase",
    # Binding/transport
    "PF00001": "7tm_receptor",
    "PF00005": "ABC_transporter",
    "PF00023": "Ankyrin_repeat",
    "PF00046": "Homeobox",
    "PF00047": "Immunoglobulin",
    "PF00096": "Zinc_finger_C2H2",
    "PF00153": "Mitochondrial_carrier",
    "PF00169": "PH_domain",
    "PF00178": "Ets_domain",
    "PF00249": "Myb_DNA_binding",
    "PF00307": "Calponin_homology",
    "PF00400": "WD40_repeat",
    "PF00412": "LIM_domain",
    "PF00435": "Spectrin_repeat",
    # Structural/signaling
    "PF00008": "EGF_like_domain",
    "PF00013": "KH_domain",
    "PF00014": "Kunitz_BPTI",
    "PF00018": "SH3_domain",
    "PF00022": "Actin",
    "PF00028": "Cadherin_domain",
    "PF00036": "EF_hand",
    "PF00041": "Fibronectin_type_III",
    "PF00048": "Small_cytokines_IL8",
    "PF00071": "Ras_family",
    "PF00078": "Reverse_transcriptase",
    "PF00102": "Protein_tyrosine_phosphatase",
    "PF00125": "Histone_core",
    "PF00168": "C2_domain",
    "PF00240": "Ubiquitin",
}


def download_family(pfam_id, name, max_seqs=1000):
    """Download SwissProt sequences for a Pfam family from UniProt."""
    out_path = DATA_DIR / f"{pfam_id}_{name}.fasta"
    if out_path.exists():
        n = sum(1 for line in open(out_path) if line.startswith(">"))
        print(f"  {pfam_id} ({name}): already exists ({n} seqs)")
        return n

    url = (
        f"https://rest.uniprot.org/uniprotkb/stream?"
        f"query=(xref:pfam-{pfam_id})+AND+(reviewed:true)"
        f"&format=fasta&size={max_seqs}"
    )

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"Accept": "text/plain"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read().decode("utf-8")
            break
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 2:
                print(f"  {pfam_id}: retry {attempt+1} ({e})")
                time.sleep(5)
            else:
                print(f"  {pfam_id}: FAILED ({e})")
                return 0

    if not data.strip():
        print(f"  {pfam_id} ({name}): no sequences found")
        return 0

    with open(out_path, "w") as f:
        f.write(data)

    n = data.count("\n>") + (1 if data.startswith(">") else 0)
    print(f"  {pfam_id} ({name}): {n} seqs")
    time.sleep(1)  # Rate limiting
    return n


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    total = 0

    print(f"Downloading {len(NEW_FAMILIES)} Pfam families to {DATA_DIR}")
    print()

    for pfam_id, name in sorted(NEW_FAMILIES.items()):
        n = download_family(pfam_id, name, max_seqs=1000)
        total += n

    print(f"\nDownloaded {total} new sequences")

    # Count all sequences
    all_count = 0
    for f in sorted(DATA_DIR.glob("PF*.fasta")):
        n = sum(1 for line in open(f) if line.startswith(">"))
        all_count += n
    print(f"Total across all families: {all_count} sequences in {len(list(DATA_DIR.glob('PF*.fasta')))} families")


if __name__ == "__main__":
    main()
