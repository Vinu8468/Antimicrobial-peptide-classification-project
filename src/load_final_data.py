"""Build a cleaned AMP/non-AMP dataset from the project's source FASTA files.

Run from any directory with:
    python /path/to/project/src/load_final_data.py

This script deliberately writes final_dataset_v3.csv rather than overwriting v2.
"""

from pathlib import Path
from collections import Counter
import argparse

import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Keep the same biological sources used by the original loader.
AMP_FASTA = RAW_DIR / "naturalAMPs_APD2024a.fasta"
NON_AMP_FASTA = RAW_DIR / "peptide_10_50_uniprot_no_toxin_antimicrobial_secreted.fasta"
OUTPUT_CSV = PROCESSED_DIR / "final_dataset_v3.csv"
AUDIT_FILE = PROCESSED_DIR / "final_dataset_v3_audit.txt"

MIN_LEN = 10
MAX_LEN = 50
RANDOM_SEED = 42
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def load_fasta(file_path: Path, label: int, source_name: str) -> tuple[pd.DataFrame, dict]:
    """Read, normalize, length-filter, and alphabet-filter a FASTA file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required FASTA file not found: {file_path}\n"
            "Check the filename and the data/raw directory."
        )

    rows = []
    counts = Counter()
    for record in SeqIO.parse(str(file_path), "fasta"):
        counts["records_read"] += 1
        sequence = str(record.seq).strip().upper()

        if not sequence:
            counts["empty"] += 1
            continue
        if not (MIN_LEN <= len(sequence) <= MAX_LEN):
            counts["outside_length_range"] += 1
            continue
        if any(aa not in VALID_AA for aa in sequence):
            counts["noncanonical_residue"] += 1
            continue

        rows.append(
            {
                "id": record.id,
                "sequence": sequence,
                "label": int(label),
                "length": len(sequence),
                "source": source_name,
            }
        )
        counts["retained"] += 1

    return pd.DataFrame(
        rows, columns=["id", "sequence", "label", "length", "source"]
    ), dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-imbalance", action="store_true",
        help="Keep every cleaned sequence instead of undersampling to equal class sizes.",
    )
    args = parser.parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {ROOT}")
    print("Loading positive AMP sequences...")
    amp_df, amp_stats = load_fasta(AMP_FASTA, label=1, source_name=AMP_FASTA.name)

    print("Loading candidate negative sequences...")
    non_amp_df, non_amp_stats = load_fasta(
        NON_AMP_FASTA, label=0, source_name=NON_AMP_FASTA.name
    )

    # Combine before deduplication so that exact sequences assigned both labels
    # can be identified and excluded rather than assigned an arbitrary label.
    combined = pd.concat([amp_df, non_amp_df], ignore_index=True)
    if combined.empty:
        raise ValueError("No sequences remained after filtering the source FASTA files.")

    before_dedup = len(combined)
    label_count_per_sequence = combined.groupby("sequence")["label"].nunique()
    conflicting_sequences = set(label_count_per_sequence[label_count_per_sequence > 1].index)
    conflict_rows = int(combined["sequence"].isin(conflicting_sequences).sum())
    combined = combined.loc[~combined["sequence"].isin(conflicting_sequences)].copy()

    # After conflicts are removed, each exact sequence has one label. Keep one
    # representative row per sequence before considering class balance.
    combined = (
        combined.sort_values(["sequence", "source", "id"])
        .drop_duplicates(subset="sequence", keep="first")
        .reset_index(drop=True)
    )
    combined["length"] = combined["sequence"].str.len()
    cleaned_class_counts = combined["label"].value_counts().sort_index().to_dict()

    # Preserve the project's original balanced-class design, but balance only
    # AFTER invalid sequences, conflicting labels, and exact duplicates are removed.
    if not args.keep_imbalance:
        class_sizes = combined["label"].value_counts()
        if len(class_sizes) != 2:
            raise ValueError("Cannot balance dataset: one class is missing after cleaning.")
        target_n = int(class_sizes.min())
        sampled_groups = [
            group.sample(n=target_n, random_state=RANDOM_SEED)
            for _, group in combined.groupby("label")
        ]
        combined = (
            pd.concat(sampled_groups, ignore_index=True)
            .sort_values(["label", "sequence"])
            .reset_index(drop=True)
        )
    combined["length"] = combined["sequence"].str.len()

    if combined["label"].nunique() != 2:
        raise ValueError(
            "The cleaned dataset does not contain both labels. Review the source FASTA files."
        )

    combined.to_csv(OUTPUT_CSV, index=False)

    lines = [
        "AMP dataset v3 audit",
        "=====================",
        f"AMP FASTA: {AMP_FASTA}",
        f"Negative-candidate FASTA: {NON_AMP_FASTA}",
        f"Length range: {MIN_LEN}-{MAX_LEN} amino acids",
        f"Allowed residues: {''.join(sorted(VALID_AA))}",
        "",
        f"AMP input statistics: {amp_stats}",
        f"Negative-candidate input statistics: {non_amp_stats}",
        f"Rows combined after sequence filtering: {before_dedup}",
        f"Distinct sequences with conflicting labels removed: {len(conflicting_sequences)}",
        f"Rows removed because their sequence had conflicting labels: {conflict_rows}",
        f"Unique class counts after deduplication, before balancing: {cleaned_class_counts}",
        f"Balancing applied: {not args.keep_imbalance}",
        f"Random seed for balancing: {RANDOM_SEED if not args.keep_imbalance else 'not used'}",
        f"Final unique sequences: {len(combined)}",
        f"Final class counts: {combined['label'].value_counts().sort_index().to_dict()}",
        f"Final duplicate sequence rows: {int(combined['sequence'].duplicated().sum())}",
        "",
        "Caution: label 0 means a sequence selected from the negative-candidate source file;",
        "it does not by itself prove experimental absence of antimicrobial activity.",
        "",
    ]
    AUDIT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print("\nInput filtering summary")
    print(f"AMP: {amp_stats}")
    print(f"Negative candidates: {non_amp_stats}")
    print(f"Conflicting sequences removed: {len(conflicting_sequences)}")
    print(f"Rows removed due to those conflicts: {conflict_rows}")
    print("Class counts after deduplication, before balancing:", cleaned_class_counts)
    print("Balancing applied:", not args.keep_imbalance)
    print("\nFinal dataset class counts:")
    print(combined["label"].value_counts().sort_index().rename(index={0: "negative candidate", 1: "AMP"}))
    print(f"\nSaved cleaned dataset: {OUTPUT_CSV}")
    print(f"Saved audit report: {AUDIT_FILE}")
    print("No existing CSV was overwritten; final_dataset.csv and v2 are preserved.")


if __name__ == "__main__":
    main()
